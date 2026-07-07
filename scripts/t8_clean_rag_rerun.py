"""T8 clean re-run — corrects T7's two boundary violations flagged by owner (2026-07-07):

 (#2 modality)  retrieval query = the omni's OWN ASR of the AUDIO question (deployable at inference),
                NOT the gold question text (which the deployment never has).
 (#3 leakage)   the gold answer string is SCRUBBED from every injected passage, so the KB cannot hand
                the model the answer -> tests external-knowledge AUGMENTATION, not answer lookup.

Arms (greedy temp0; the spoken-question AUDIO is in every arm):
  base           : audio only (parametric)
  inject_k_raw   : inject top-k passages UNSCRUBBED  (leaky reference = T7 inject_k; sizes the lookup)
  inject_k_scrub : inject top-k passages, gold answer SCRUBBED  (the CLEAN knowledge signal)
  oracle_scrub   : inject the item's own passage, gold SCRUBBED (clean oracle)

Read-outs:
  clean_H0 = acc(inject_k_scrub) - acc(base)               (does external knowledge, sans the answer, help?)
  lookup   = acc(inject_k_raw)  - acc(inject_k_scrub)      (how much of T7's 'gain' was answer-copy)
Boundary: query = omni-own-ASR of audio; injected passages have the gold answer removed; the gold is used
ONLY to score (held out, never injected). directional-only, small n.

reproduce: SPEECHRL_DATA_DIR=<repo>/speechrl-data python -u scripts/t8_clean_rag_rerun.py
"""
import json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2
import t7_rag_gate_probe as t7

N_EVAL = int(os.environ.get("T8_N", "60"))
TOPK = int(os.environ.get("T8_K", "5"))
SEED = 20260707
OUT = p2.DATA / "_repro" / "t8_clean_rag_rerun.json"
ASR_INSTR = "请把这段音频里的问题逐字转写成文字,只输出转写文本,不要作答。"
ANS_INSTR = "请只用一个简短答案回答音频里的问题(只给答案,不要解释)。"
REF = "参考资料(检索得到,可能相关也可能无关):\n{docs}\n\n"


def scrub(passage, golds):
    p = passage
    for g in golds:
        g = g.strip()
        if g:
            p = re.sub(re.escape(g), "[…]", p, flags=re.IGNORECASE)
    return p


def main():
    rng = np.random.default_rng(SEED)
    pool = t7.load_heysquad(int(os.environ.get("T7_KB_ROWS", "3000")))
    kb_texts, seen = [], set()
    for r in pool:
        if r["ctx"] not in seen:
            seen.add(r["ctx"]); kb_texts.append(r["ctx"])
    rname, kb_vec, embed_q = t7.build_retriever(kb_texts)
    eval_items = [pool[int(j)] for j in rng.permutation(len(pool))[:N_EVAL]]
    print(f"server={p2.BASE} n={N_EVAL} retriever={rname} KB={len(kb_texts)}", flush=True)

    arms = {"base": [], "inject_k_raw": [], "inject_k_scrub": [], "oracle_scrub": []}
    asr_hit, scrub_leak = [], []
    t0 = time.time()
    for i, it in enumerate(eval_items):
        try:
            wav = t7._wav(it)
            tr = p2.gen(wav, ASR_INSTR, 42, 0.0)                       # omni's own ASR -> retrieval query
        except Exception as e:
            print(f"  skip #{i} asr: {type(e).__name__}", flush=True); continue
        order = np.argsort(-(embed_q([tr]) @ kb_vec.T)[0])[:TOPK]
        topk = [kb_texts[int(j)] for j in order]
        topk_scrub = [scrub(d, it["golds"]) for d in topk]
        own_scrub = scrub(it["ctx"], it["golds"])
        asr_hit.append(int(it["ctx"] in topk))
        scrub_leak.append(int(any(t7.any_gold_in(it["golds"], d) for d in topk_scrub)))
        try:
            a_base = t7.any_gold_in(it["golds"], p2.gen(wav, ANS_INSTR, 42, 0.0))
            a_raw = t7.any_gold_in(it["golds"], p2.gen(wav, REF.format(docs="\n---\n".join(topk)) + ANS_INSTR, 42, 0.0))
            a_scr = t7.any_gold_in(it["golds"], p2.gen(wav, REF.format(docs="\n---\n".join(topk_scrub)) + ANS_INSTR, 42, 0.0))
            a_osc = t7.any_gold_in(it["golds"], p2.gen(wav, REF.format(docs=own_scrub) + ANS_INSTR, 42, 0.0))
        except Exception as e:
            print(f"  skip #{i} gen: {type(e).__name__}", flush=True); continue
        arms["base"].append(a_base); arms["inject_k_raw"].append(a_raw)
        arms["inject_k_scrub"].append(a_scr); arms["oracle_scrub"].append(a_osc)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(eval_items)}] base={np.mean(arms['base']):.3f} raw={np.mean(arms['inject_k_raw']):.3f} "
                  f"scrub={np.mean(arms['inject_k_scrub']):.3f} osc={np.mean(arms['oracle_scrub']):.3f} "
                  f"asr_hit={np.mean(asr_hit):.2f} leak={np.mean(scrub_leak):.2f} ({time.time()-t0:.0f}s)", flush=True)

    def m(x):
        return round(float(np.mean(x)), 3) if x else None

    def boot(a, b, iters=2000):
        a, b = np.array(a, float), np.array(b, float); d = a - b
        bs = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(iters)]
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    n = len(arms["base"])
    res = {"stage": "1-directional", "grade": "small-n, single-touch, boundary-clean (audio-query + answer-scrubbed)",
           "model": "qwen3-omni-30b Q8_0 GGUF", "retriever": rname, "n": n,
           "asr_query_retrieval_hit@k": m(asr_hit), "post_scrub_answer_still_present": m(scrub_leak),
           "acc": {k: m(v) for k, v in arms.items()},
           "clean_H0_scrub_minus_base": (round(m(arms["inject_k_scrub"]) - m(arms["base"]), 3) if n else None),
           "clean_H0_CI": (boot(arms["inject_k_scrub"], arms["base"]) if n else None),
           "lookup_component_raw_minus_scrub": (round(m(arms["inject_k_raw"]) - m(arms["inject_k_scrub"]), 3) if n else None),
           "boundary": "query=omni-own-ASR of AUDIO (deployable); injected passages have gold answer SCRUBBED; "
                       "gold used ONLY for scoring (held out, never injected)",
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python -u scripts/t8_clean_rag_rerun.py"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T8 CLEAN RAG RERUN ===\n" + json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    print("wrote", OUT, "\nT8_DONE", flush=True)


if __name__ == "__main__":
    main()

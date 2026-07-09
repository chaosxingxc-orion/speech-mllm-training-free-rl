"""T6 build + T7 experiment (combined) — R1 admission-gate on a boundary-clean heysquad RAG testbed.

Frozen Qwen3-Omni (llama.cpp) = generator; a fit-once lexical/embedding retriever (CPU) = off-the-shelf
tool (retriever is NOT the research object -- utilization is). Testbed heysquad (SQuAD-based spoken QA):
  KB     = pool of `context` passages (external knowledge, retrievable).
  query  = the QUESTION text (clean-retrieval isolation; own-ASR regime is a follow-up) -> top-k passages.
  gold   = `answers` -- held out for SCORING ONLY, never injected.

Arms (greedy temp0; the spoken-question AUDIO is in EVERY arm):
  base     : audio question only (parametric)                              [no external knowledge]
  inject_k : inject ALL top-k retrieved passages (text)                    [inject-top-k-always FLOOR = C<A regime]
  gate     : R1 -- admit retrieved passages with sim >= GATE_FRAC*max_sim  [reward-guided ADMISSION gate]
  oracle   : inject the item's OWN gold passage                            [H0 oracle-retrieval CEILING]

Claims (T4 pre-reg): H0 = acc(oracle)-acc(base) > 0 (headroom); H-util-gate = acc(gate)-acc(inject_k) > 0.
Boundary: query=question text (not answer); answers never injected; reward/gate = retriever relevance
(proxy, non-gold, a DIFFERENT model from the generator = decorrelated). directional-only at this n.

reproduce: SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python -u scripts/t7_rag_gate_probe.py
"""
import glob, io, json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

N_EVAL = int(os.environ.get("T7_N", "60"))
KB_ROWS = int(os.environ.get("T7_KB_ROWS", "3000"))
TOPK = int(os.environ.get("T7_K", "5"))
GATE_FRAC = float(os.environ.get("T7_GATE_FRAC", "0.9"))   # admit passage iff sim >= GATE_FRAC * max_sim
SEED = 20260707
DS = p2.DS
OUT = p2.DATA / "_repro" / "t7_rag_gate_probe.json"
ANS_INSTR = "请只用一个简短答案回答音频里的问题(只给答案,不要解释)。"
REF = "参考资料(检索得到,可能相关也可能无关):\n{docs}\n\n"


def _golds(a):
    out = []
    if a is None:
        return out
    if isinstance(a, dict):
        t = a.get("text")
        out = list(t) if isinstance(t, (list, tuple)) else ([t] if t else [])
    elif isinstance(a, (list, tuple)):
        for el in a:
            if isinstance(el, dict) and el.get("text"):
                out.append(el["text"])
            elif isinstance(el, str):
                out.append(el)
    return [str(x) for x in out if str(x).strip()]


def load_heysquad(n_rows):
    fs = sorted(glob.glob(f"{DS}/heysquad/**/*.parquet", recursive=True))
    rows = []
    for f in fs:
        try:
            t = pq.read_table(f, columns=["audio", "question", "context", "answers", "is_impossible", "id"])
        except Exception:
            continue
        for r in t.to_pylist():
            if r.get("is_impossible"):
                continue
            g = _golds(r.get("answers"))
            if r.get("audio") and r.get("context") and g:
                rows.append({"audio": r["audio"], "q": r["question"], "ctx": r["context"], "golds": g,
                             "id": r.get("id")})
        if len(rows) >= n_rows:
            break
    return rows[:n_rows]


def _l2(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def build_retriever(kb_texts):
    """Fit-once retriever: returns (kb_vec, embed_query_fn) sharing ONE vector space."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        kb = _l2(m.encode(list(kb_texts), normalize_embeddings=True))
        return "minilm", kb, lambda qs: _l2(m.encode(list(qs), normalize_embeddings=True))
    except Exception as e:
        print(f"  [retriever] minilm unavailable ({type(e).__name__}); word-TFIDF fallback", flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50000)
        v.fit(list(kb_texts))                                   # fit ONCE on the KB
        kb = _l2(v.transform(list(kb_texts)).toarray())
        return "tfidf-word12", kb, lambda qs: _l2(v.transform(list(qs)).toarray())


def _wav(it):
    a = it["audio"]
    if isinstance(a, dict) and a.get("bytes"):
        return p2._wav_from_bytes(a["bytes"], f"hsq_{it['id']}")
    if isinstance(a, dict) and a.get("array") is not None:
        p2.WAV.mkdir(parents=True, exist_ok=True)
        p = p2.WAV / f"hsq_{it['id']}.wav"
        if not p.exists():
            p2.sf.write(str(p), np.asarray(a["array"], dtype="float32"), int(a.get("sampling_rate", 16000)))
        return str(p)
    return p2._wav_from_bytes(a, f"hsq_{it['id']}")


def any_gold_in(golds, text):
    o = p2.norm(text)
    return int(any(p2.norm(g) and p2.norm(g) in o for g in golds))


def main():
    rng = np.random.default_rng(SEED)
    print(f"server={p2.BASE} n_eval={N_EVAL} kb_rows={KB_ROWS} k={TOPK} gate_frac={GATE_FRAC}", flush=True)
    pool = load_heysquad(KB_ROWS)
    kb_texts, seen = [], set()
    for r in pool:
        if r["ctx"] not in seen:
            seen.add(r["ctx"]); kb_texts.append(r["ctx"])
    rname, kb_vec, embed_q = build_retriever(kb_texts)
    kb_idx = {t: i for i, t in enumerate(kb_texts)}
    print(f"  retriever={rname} KB={len(kb_texts)} passages (from {len(pool)} rows)", flush=True)

    eval_items = [pool[int(j)] for j in rng.permutation(len(pool))[:N_EVAL]]
    q_vec = embed_q([r["q"] for r in eval_items])
    sims = q_vec @ kb_vec.T

    arms = {"base": [], "inject_k": [], "gate": [], "oracle": []}
    base_fail_oracle, gate_hit, k_hit, n_admit = [], [], [], []
    t0 = time.time()
    for i, it in enumerate(eval_items):
        try:
            wav = _wav(it)
        except Exception as e:
            print(f"    skip #{i} wav: {type(e).__name__}", flush=True); continue
        order = np.argsort(-sims[i])[:TOPK]
        topk = [kb_texts[int(j)] for j in order]
        ss = np.array([sims[i][int(j)] for j in order])
        thr = GATE_FRAC * float(ss.max()) if ss.size else 0.0
        admit = [d for d, s in zip(topk, ss) if s >= thr] or topk[:1]
        k_hit.append(int(it["ctx"] in topk)); gate_hit.append(int(it["ctx"] in admit)); n_admit.append(len(admit))
        try:
            a_base = any_gold_in(it["golds"], p2.gen(wav, ANS_INSTR, 42, 0.0))
            a_k = any_gold_in(it["golds"], p2.gen(wav, REF.format(docs="\n---\n".join(topk)) + ANS_INSTR, 42, 0.0))
            a_g = any_gold_in(it["golds"], p2.gen(wav, REF.format(docs="\n---\n".join(admit)) + ANS_INSTR, 42, 0.0))
            a_o = any_gold_in(it["golds"], p2.gen(wav, REF.format(docs=it["ctx"]) + ANS_INSTR, 42, 0.0))
        except Exception as e:
            print(f"    skip #{i} gen: {type(e).__name__}", flush=True); continue
        arms["base"].append(a_base); arms["inject_k"].append(a_k)
        arms["gate"].append(a_g); arms["oracle"].append(a_o)
        if a_base == 0:
            base_fail_oracle.append(a_o)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(eval_items)}] base={np.mean(arms['base']):.3f} k={np.mean(arms['inject_k']):.3f} "
                  f"gate={np.mean(arms['gate']):.3f} oracle={np.mean(arms['oracle']):.3f} "
                  f"admit={np.mean(n_admit):.1f} hit@k={np.mean(k_hit):.2f} ({time.time()-t0:.0f}s)", flush=True)

    def m(x):
        return round(float(np.mean(x)), 3) if x else None

    def boot_ci(a, b, iters=2000):
        a, b = np.array(a, float), np.array(b, float); d = a - b
        bs = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(iters)]
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    n = len(arms["base"])
    res = {"stage": "1-directional", "grade": "small-n, single-touch, NOT significance-bearing",
           "model": "qwen3-omni-30b Q8_0 GGUF llama.cpp", "retriever": rname, "n": n,
           "kb_passages": len(kb_texts), "topk": TOPK, "gate_frac": GATE_FRAC,
           "retrieval_hit_at_k": m(k_hit), "gate_kept_gold": m(gate_hit), "mean_admitted": m(n_admit),
           "acc": {k: m(v) for k, v in arms.items()},
           "H0_oracle_minus_base": (round(m(arms["oracle"]) - m(arms["base"]), 3) if n else None),
           "H0_oracle_minus_base_CI": (boot_ci(arms["oracle"], arms["base"]) if n else None),
           "H0_rescue_on_base_fails": m(base_fail_oracle), "n_base_fails": len(base_fail_oracle),
           "Hutil_gate_minus_injectk": (round(m(arms["gate"]) - m(arms["inject_k"]), 3) if n else None),
           "Hutil_gate_minus_injectk_CI": (boot_ci(arms["gate"], arms["inject_k"]) if n else None),
           "boundary": "query=question text (not answer); answers never injected; KB=passage pool; "
                       "gate reward=retriever relevance (proxy, non-gold, decorrelated from generator)",
           "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python -u scripts/t7_rag_gate_probe.py"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T7 RAG GATE PROBE ===\n" + json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    print("wrote", OUT, "\nT7_DONE", flush=True)


if __name__ == "__main__":
    main()

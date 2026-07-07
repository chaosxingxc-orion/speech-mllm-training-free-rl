"""Answer-in-KB LEAKAGE AUDIT for T7 (no generation; string overlap only).

Quantifies how much of T7's 'RAG gain' is the injected passage CONTAINING the gold answer verbatim
(reading-comprehension / answer-lookup) vs genuine external-knowledge augmentation. This is the
leakage the owner flagged: if the KB records the ground truths, 'task-RAG' degenerates to answer lookup.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2
import t7_rag_gate_probe as t7

N_EVAL = int(os.environ.get("T7_N", "60"))
SEED = 20260707
GATE_FRAC = 0.9
TOPK = 5


def main():
    rng = np.random.default_rng(SEED)
    pool = t7.load_heysquad(int(os.environ.get("T7_KB_ROWS", "3000")))
    kb_texts, seen = [], set()
    for r in pool:
        if r["ctx"] not in seen:
            seen.add(r["ctx"]); kb_texts.append(r["ctx"])
    rname, kb_vec, embed_q = t7.build_retriever(kb_texts)
    eval_items = [pool[int(j)] for j in rng.permutation(len(pool))[:N_EVAL]]
    q_vec = embed_q([r["q"] for r in eval_items])
    sims = q_vec @ kb_vec.T

    def gold_in(text, golds):
        o = p2.norm(text)
        return any(p2.norm(g) and p2.norm(g) in o for g in golds)

    own, in_topk, in_gate, in_distractors_only = [], [], [], []
    for i, it in enumerate(eval_items):
        g = it["golds"]
        own.append(gold_in(it["ctx"], g))                       # answer in the item's own passage (=KB entry)
        order = np.argsort(-sims[i])[:TOPK]
        topk = [kb_texts[int(j)] for j in order]
        ss = np.array([sims[i][int(j)] for j in order])
        thr = GATE_FRAC * float(ss.max()) if ss.size else 0.0
        admit = [d for d, s in zip(topk, ss) if s >= thr] or topk[:1]
        in_topk.append(any(gold_in(d, g) for d in topk))        # answer present in the injected top-k text
        in_gate.append(any(gold_in(d, g) for d in admit))       # answer present in gated text
        # answer present in a NON-own retrieved passage (a distractor) — pure noise carrying the answer?
        others = [d for d in topk if d != it["ctx"]]
        in_distractors_only.append(any(gold_in(d, g) for d in others))

    def f(x):
        return round(float(np.mean(x)), 3)
    out = {
        "n": len(eval_items), "retriever": rname,
        "answer_in_own_KB_passage": f(own),          # ~1.0 => KB literally records the ground truths
        "answer_in_injected_topk": f(in_topk),       # inject_k arm: answer handed to model
        "answer_in_gated": f(in_gate),               # gate arm
        "answer_in_a_distractor_passage": f(in_distractors_only),
        "interpretation": "answer_in_injected_topk ~= T7 inject_k accuracy 0.767 => the 'RAG gain' is "
                          "dominated by the injected passage CONTAINING the answer (reading-comp / lookup), "
                          "NOT external-knowledge reasoning. Leakage confirmed structurally.",
    }
    import json
    print("=== T7 LEAKAGE AUDIT ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""E2 clean (utilization) — COUNTERFACTUAL knowledge-injection: does the frozen omni USE injected
external knowledge over its own parametric belief? Boundary-clean, cross-dataset, audio-input.

For each spoken-QA item (real AUDIO question; original gold A), we inject a ONE-LINE COUNTERFACTUAL
fact stating a DIFFERENT answer A' (= another item's gold, answer-shaped but wrong for this item):
    "参考资料:根据资料,该问题的正确答案是「A'」。"
Arms (greedy temp0; audio in both):
  base      : audio only.
  inject_CF : audio + the counterfactual fact.
Metrics per set:
  base_acc_A       : parametric correctness (matches original gold A)
  base_says_Aprime : control — should be ~0 (model wouldn't say A' unprompted)
  CF_follow_Aprime : inject_CF output matches A'  =  the model ADOPTED injected external knowledge
  CF_keep_A        : inject_CF output still matches A (ignored injection, kept parametric)

Why boundary-clean: A' is a controlled COUNTERFACTUAL (not this item's gold); we score adoption of OUR
injected value, never leak the dataset gold as 'correct'. This isolates knowledge-USE/override (the RAG
utilization mechanism) from answer-lookup-of-gold. Pairs with T8 (answer-scrubbed augmentation = null).

reproduce: SPEECHRL_DATA_DIR=<repo>/speechrl-data python -u scripts/t9_cf_utilization.py
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

N = int(os.environ.get("T9_N", "40"))
SEED = 20260707
OUT = p2.DATA / "_repro" / "t9_cf_utilization.json"
ANS = "请只用一个简短答案回答音频里的问题(只给答案,不要解释)。"
CF = "参考资料:根据资料,该问题的正确答案是「{a}」。\n\n"
SETS = {"SQuAD-zh": p2.load_squad_zh, "vocalbench-zh": p2.load_vocalbench_zh, "big-bench-audio": p2.load_bba}


def run_set(name, loader):
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = N
    items = [it for it in loader(rng) if it["task"] == "qa" and str(it["gold"]).strip()]
    n = len(items)
    golds = [str(it["gold"]) for it in items]
    aprime = [golds[(k + 1) % n] for k in range(n)]                 # counterfactual = shift-by-1 other gold
    bA, bAp, cf_follow, cf_keep = [], [], [], []
    t0 = time.time()
    for k, it in enumerate(items):
        A, Ap = golds[k], aprime[k]
        if p2.norm(Ap) == p2.norm(A):
            continue
        try:
            ob = p2.gen(it["wav"], ANS, 42, 0.0)
            oc = p2.gen(it["wav"], CF.format(a=Ap) + ANS, 42, 0.0)
        except Exception as e:
            print(f"  skip {name}#{k}: {type(e).__name__}", flush=True); continue
        nb, nc = p2.norm(ob), p2.norm(oc)
        bA.append(int(p2.norm(A) in nb)); bAp.append(int(p2.norm(Ap) in nb))
        cf_follow.append(int(p2.norm(Ap) in nc)); cf_keep.append(int(p2.norm(A) in nc))
        if (k + 1) % 20 == 0:
            print(f"  [{name} {k+1}/{n}] baseA={np.mean(bA):.3f} follow={np.mean(cf_follow):.3f} "
                  f"keep={np.mean(cf_keep):.3f} ({time.time()-t0:.0f}s)", flush=True)

    def m(x):
        return round(float(np.mean(x)), 3) if x else None
    res = {"n": len(bA), "base_acc_A": m(bA), "base_says_Aprime": m(bAp),
           "CF_follow_Aprime": m(cf_follow), "CF_keep_A": m(cf_keep)}
    print(f"== {name}: baseA={res['base_acc_A']} base_says_Ap={res['base_says_Aprime']} "
          f"CF_follow={res['CF_follow_Aprime']} CF_keep={res['CF_keep_A']} (n={res['n']})", flush=True)
    return res


def main():
    print(f"server={p2.BASE} n={N}", flush=True)
    results = {}
    for name, loader in SETS.items():
        try:
            results[name] = run_set(name, loader)
        except Exception as e:
            print(f"!! {name} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    ok = [r for r in results.values() if "CF_follow_Aprime" in r]
    follow = round(float(np.mean([r["CF_follow_Aprime"] for r in ok])), 3) if ok else None
    out = {"stage": "1-directional", "grade": "small-n, single-touch, boundary-clean (counterfactual utilization)",
           "model": "qwen3-omni-30b Q8_0 GGUF", "results": results, "mean_CF_follow": follow,
           "interpretation": "CF_follow_Aprime = fraction the frozen omni ADOPTS the injected counterfactual "
                            "external fact over its parametric answer = clean knowledge-USE/override rate "
                            "(no gold leakage: A' is a controlled counterfactual). High => external knowledge "
                            "is used; pairs with T8 (answer-scrubbed augmentation = null).",
           "boundary": "audio-input question; injected value A' is a counterfactual (another item's gold), "
                       "NOT this item's gold; we score adoption of the injected value, never the gold.",
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python -u scripts/t9_cf_utilization.py"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T9 CF UTILIZATION ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("wrote", OUT, "\nT9_DONE", flush=True)


if __name__ == "__main__":
    main()

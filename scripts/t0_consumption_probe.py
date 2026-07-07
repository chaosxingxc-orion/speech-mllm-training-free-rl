"""T0 — model-evolution VALIDITY PROBE (WS-0): can the FROZEN Qwen3-Omni CONSUME injected external
text knowledge in-context (audio-query regime), WITHOUT training?

Why: the surveyed field gap (MARS/RASST/VoxMind) — "speech-LLMs need fine-tuning to consume retrieved
knowledge" — was observed on 2025->2026-04 models. Our base is Qwen3-Omni (2026-04/05, stronger). This
re-checks that premise ON OUR MODEL. Stage-1 directional (small n, single-touch, NOT significance-bearing).

Design (3 arms, greedy temp 0, audio query in every arm, boundary-aware):
  A no-inject       : baseline greedy on the spoken-QA item.
  B inject-correct  : prepend the item's GOLD answer as a "reference note"  -> CONSUMPTION CEILING
                      (can the frozen omni use handed-in text at all, in the audio regime?).
  C inject-mismatch : prepend a DIFFERENT item's gold as reference          -> PLACEBO control.

Read-out (go/shift, NOT kill):
  B>>A and B>>C  -> frozen omni CONSUMES injected knowledge in-context -> the "must-train-to-consume"
                    gap is model-evolved-away -> thesis SHIFTS to optimizing WHAT/WHEN to retrieve+inject.
  B ~ A          -> can't use even a handed answer -> consumption gap confirmed -> thesis stands as-is.

Boundary: B/C inject text as a reference; this is a CEILING probe (gold-as-ref), directional-only, NOT a
deployable gain. Real imperfect-retrieval utilization is H0/H-util at T7. No test-item leakage beyond the
deliberate ceiling arm (which exists precisely to bound consumption, and is never used as a deployable lever).

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python -u scripts/t0_consumption_probe.py
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

N = int(os.environ.get("T0_N", "40"))
SEED = 20260706
SETS = {"big-bench-audio": p2.load_bba, "vocalbench-zh": p2.load_vocalbench_zh}
OUT = p2.DATA / "_repro" / "t0_consumption_probe.json"
REF = "参考资料(供参考,可能相关也可能无关,请结合音频作答):\n{fact}\n\n"


def gen_inject(wav, instr, fact, seed):
    return p2.gen(wav, REF.format(fact=fact) + instr, seed, 0.0)


def run_set(name, loader):
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = N
    items = [it for it in loader(rng) if it["task"] == "qa"]
    golds = [it["gold"] for it in items]
    n_items = len(golds)
    mism = [golds[(k + 1) % n_items] for k in range(n_items)]  # shift-by-1 derangement
    A, B, C, resc = [], [], [], []
    t0 = time.time()
    for k, it in enumerate(items):
        try:
            a = p2.reward(it, p2.gen(it["wav"], it["instr"], 42, 0.0))
            b = p2.reward(it, gen_inject(it["wav"], it["instr"], it["gold"], 42))
            c = p2.reward(it, gen_inject(it["wav"], it["instr"], mism[k], 42))
        except Exception as e:
            print(f"    skip {name}#{k}: {type(e).__name__}", flush=True)
            continue
        A.append(a); B.append(b); C.append(c)
        if a == 0:
            resc.append(b)
        if (k + 1) % 20 == 0:
            print(f"  [{name} {k+1}/{len(items)}] A={np.mean(A):.3f} B={np.mean(B):.3f} "
                  f"C={np.mean(C):.3f} ({time.time()-t0:.0f}s)", flush=True)
    n = len(A)
    mA, mB, mC = float(np.mean(A)), float(np.mean(B)), float(np.mean(C))
    res = {"n": n, "A_no_inject": round(mA, 3), "B_inject_correct_ceiling": round(mB, 3),
           "C_inject_mismatch_placebo": round(mC, 3), "B_minus_A": round(mB - mA, 3),
           "B_minus_C": round(mB - mC, 3),
           "rescue_rate_on_A_fails": round(float(np.mean(resc)) if resc else 0.0, 3),
           "n_A_fails": len(resc)}
    print(f"== {name}: A={res['A_no_inject']} B={res['B_inject_correct_ceiling']} "
          f"C={res['C_inject_mismatch_placebo']} B-A={res['B_minus_A']:+} B-C={res['B_minus_C']:+} "
          f"rescue={res['rescue_rate_on_A_fails']} (n={n}, A-fails={len(resc)})", flush=True)
    return res


def main():
    print(f"server={p2.BASE} n={N} seed={SEED}", flush=True)
    results = {}
    for name, loader in SETS.items():
        try:
            results[name] = run_set(name, loader)
        except Exception as e:
            print(f"!! {name} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    ok = [r for r in results.values() if "B_inject_correct_ceiling" in r]
    verdict = "inconclusive"
    if ok:
        mBA = float(np.mean([r["B_minus_A"] for r in ok]))
        mBC = float(np.mean([r["B_minus_C"] for r in ok]))
        if mBA > 0.15 and mBC > 0.10:
            verdict = ("SHIFT: frozen omni consumes injected knowledge in-context (audio regime) -> "
                       "thesis = optimize WHAT/WHEN to retrieve+inject (selection), not enable consumption")
        elif mBA < 0.05:
            verdict = "GO: consumption gap confirmed on our model -> thesis = enable consumption stands"
        else:
            verdict = "MIXED: partial consumption -> inspect per-set; refine at T7"
    out = {"stage": "1-directional", "grade": "small-n, single-touch, NOT significance-bearing",
           "model": "qwen3-omni-30b Q8_0 GGUF llama.cpp",
           "design": "A no-inject / B inject-correct-gold (CEILING) / C inject-mismatch (PLACEBO); "
                     "greedy temp0; audio-query in every arm",
           "boundary_note": "B/C inject gold-as-reference = CEILING probe, directional-only, NOT a "
                            "deployable gain; real imperfect-retrieval utilization = H0/H-util at T7",
           "results": results, "verdict": verdict,
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python -u scripts/t0_consumption_probe.py"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T0 CONSUMPTION PROBE ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("wrote", OUT, "\nT0_DONE", flush=True)


if __name__ == "__main__":
    main()

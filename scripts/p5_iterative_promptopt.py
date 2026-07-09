"""T3 (E8′) — LEGITIMATE iterative, feedback-driven instruction optimization (not a 4-candidate pick).

The last prompt-space read-out lever. Offline dev-set optimization (dev labels are legitimately available
at optimize time, like training a prompt), then the optimized instruction is FROZEN and evaluated greedily
on a held-out TEST split — test items are audio-only, no per-item leakage (Information-Boundary-Guard).
This is APE/OPRO/TextGrad-style *iterative refinement*, distinct from E8 (candidate pick) and T2 (few-shot).

Loop (R rounds): eval current instruction on dev -> collect wrong cases (model answer vs correct, as text)
-> ask the model (text-only meta-prompt) to propose an improved GENERAL instruction fixing those errors ->
keep it iff dev accuracy improves. Report frozen test greedy(best) - greedy(base) with paired-bootstrap CI.

reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/p5_iterative_promptopt.py --only mmau-mini,SQuAD-zh
"""
import json, os, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

BASE = p2.BASE
ROUNDS = int(os.environ.get("T3_ROUNDS", "3"))
N_DEV = int(os.environ.get("T3_NDEV", "25"))
N_TEST = int(os.environ.get("T3_NTEST", "40"))
SEED = 20260705
NBOOT = 10000
OUT = p2.DATA / "_repro" / "t3_iterative_promptopt.json"


def _b64(p):
    import base64
    return base64.b64encode(open(p, "rb").read()).decode()


def call_audio(wav, instr, temp=0.0):
    content = [{"type": "input_audio", "input_audio": {"data": _b64(wav), "format": "wav"}},
               {"type": "text", "text": instr}]
    return _post(content, temp)


def call_text(prompt, temp=0.0, maxtok=220):
    return _post([{"type": "text", "text": prompt}], temp, maxtok)


def _post(content, temp, maxtok=None):
    payload = {"messages": [{"role": "user", "content": content}],
               "max_tokens": maxtok or p2.MAXTOK, "seed": 42, "temperature": temp}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as z:
        return json.loads(z.read().decode())["choices"][0]["message"]["content"].strip()


def acc_on(items, instr):
    r = []
    for it in items:
        try:
            r.append(p2.reward(it, call_audio(it["wav"], instr)))
        except Exception:
            r.append(0)
    return np.array(r, float)


def gold_text(it):
    if it["task"] == "mcq":
        return p2.LETTERS[it["gold"]] + ". " + it["opts"][it["gold"]]
    if it["task"] == "slu":
        return f"{it['gold']}. {p2.MINDS14[it['gold']]}"
    return str(it["gold"])


def refine(instr, dev, dev_rewards):
    """Text-only meta-prompt: propose an improved general instruction from dev error patterns."""
    wrong = [dev[i] for i in range(len(dev)) if dev_rewards[i] == 0][:6]
    if not wrong:
        return instr
    errs = "\n".join(f"- model answered wrongly; the correct answer was: {gold_text(w)}" for w in wrong)
    meta = (f"You are optimizing the INSTRUCTION given to a speech model for a task.\n"
            f"Current instruction:\n\"\"\"{instr}\"\"\"\n\n"
            f"On a dev set it got these items wrong (correct answers shown; you cannot hear the audio):\n{errs}\n\n"
            f"Propose ONE improved, GENERAL instruction (not specific to these items) that would reduce such "
            f"errors. Keep it concise. Output ONLY the new instruction text, nothing else.")
    try:
        new = call_text(meta)
        return new.strip().strip('"') or instr
    except Exception:
        return instr


def boot(d):
    rng = np.random.default_rng(1)
    return [round(float(np.quantile([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(NBOOT)], q)), 4)
            for q in (.025, .975)]


def run(name, loader):
    p2.N_UTTS = N_DEV + N_TEST + 20
    items = loader(np.random.default_rng(SEED))
    if len(items) < N_DEV + 10:
        return {"error": "too few"}
    dev, test = items[:N_DEV], items[N_DEV:N_DEV + N_TEST]
    base = dev[0]["instr"]
    t0 = time.time()
    best_instr, best_dev = base, acc_on(dev, base).mean()
    cur = base
    trace = [round(float(best_dev), 3)]
    for r in range(ROUNDS):
        dr = acc_on(dev, cur)
        cur = refine(cur, dev, dr)
        da = acc_on(dev, cur).mean()
        trace.append(round(float(da), 3))
        if da > best_dev:
            best_dev, best_instr = da, cur
        print(f"  [{name} r{r+1}] dev_acc={da:.3f} best={best_dev:.3f} ({time.time()-t0:.0f}s)", flush=True)
    base_test = acc_on(test, base)
    opt_test = acc_on(test, best_instr)
    res = {"task": test[0]["task"], "n_test": len(test), "rounds": ROUNDS,
           "base_test": round(float(base_test.mean()), 4), "opt_test": round(float(opt_test.mean()), 4),
           "opt_minus_base": round(float((opt_test - base_test).mean()), 4),
           "delta_CI": boot(opt_test - base_test),
           "dev_acc_trace": trace, "best_dev": round(float(best_dev), 4),
           "best_instruction": best_instr[:400]}
    lo = res["delta_CI"][0]; hi = res["delta_CI"][1]
    res["sig"] = "SIG(+)" if lo > 0 else ("SIG(-)" if hi < 0 else "n.s.")
    print(f"  == {name}: base={res['base_test']} opt={res['opt_test']} d={res['opt_minus_base']:+.3f} "
          f"CI{res['delta_CI']} [{res['sig']}] devtrace={trace}", flush=True)
    return res


def main():
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    print(f"server={BASE} rounds={ROUNDS} n_dev={N_DEV} n_test={N_TEST}", flush=True)
    out = json.load(open(OUT)).get("results", {}) if OUT.exists() else {}
    for name, loader in p2.LOADERS.items():
        if only and name not in only:
            continue
        try:
            out[name] = run(name, loader)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "lever": "iterative feedback-driven prompt-opt (E8′)",
                               "boundary": "dev-set offline optimization (dev labels ok), frozen test greedy, test audio-only",
                               "rounds": ROUNDS, "n_dev": N_DEV, "n_test": N_TEST, "seed": SEED},
                   "results": out, "reproduce": "python scripts/p5_iterative_promptopt.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== T3 ITERATIVE PROMPT-OPT ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("T3_DONE", flush=True)


if __name__ == "__main__":
    main()

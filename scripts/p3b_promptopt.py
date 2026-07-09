"""E8 — in-fence global prompt optimization (dev-label-scored system-prompt search) + transfer.

Training-free, no per-instance selector: score a pool of candidate system-prompts (part of the
conditioning A) on a DEV split with labels, pick A* = argmax dev-acc, apply the FIXED A* to a held-out
TEST split. Reports greedy(A*) − greedy(A_base) on test and the dev→test transfer ratio. On MCQ/QA/SLU
with dev labels this is a legitimate OPRO-flavoured search that needs no label-free selection at deploy.

reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/p3b_promptopt.py --only mmau-mini
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2
import base64, urllib.request

BASE = p2.BASE
N_DEV = int(os.environ.get("E8_NDEV", "60"))
N_TEST = int(os.environ.get("E8_NTEST", "120"))
SEED = 20260705
OUT = p2.DATA / "_repro" / "e8_promptopt.json"

CANDS = [
    "",  # A_base (no system prompt)
    "You are an expert audio analyst. Answer precisely and only what the audio supports.",
    "Listen very carefully to the entire audio before answering. Then give the single best answer.",
    "You are a careful, accurate assistant. Do not guess; commit to the most defensible answer.",
    "Reason about the audio content step by step internally, then output only the final answer.",
    "You are a domain expert. Identify the key evidence in the audio, then answer decisively.",
    "Be concise and correct. If unsure, pick the option best supported by the audio.",
    "Focus on what is actually said/heard, not on surface cues. Answer accurately.",
]
CANDS = CANDS[:int(os.environ.get("E8_NCANDS", str(len(CANDS))))]  # keep "" base + first N-1


def gen_sys(wav, instr, system, seed, temp):
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instr}]})
    payload = {"messages": msgs, "max_tokens": p2.MAXTOK, "seed": seed, "temperature": temp}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def acc_on(items, system):
    c = 0
    for it in items:
        try:
            c += p2.reward(it, gen_sys(it["wav"], it["instr"], system, 42, 0.0))
        except Exception:
            pass
    return c / max(1, len(items))


def run_ds(name, loader):
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = N_DEV + N_TEST + 40
    items = loader(rng)
    if len(items) < N_DEV + 20:
        return {"error": f"too few items ({len(items)})"}
    dev, test = items[:N_DEV], items[N_DEV:N_DEV + N_TEST]
    t0 = time.time()
    dev_scores = []
    for ci, s in enumerate(CANDS):
        a = acc_on(dev, s)
        dev_scores.append(a)
        print(f"  [{name} cand{ci}] dev={a:.3f} ({time.time()-t0:.0f}s)", flush=True)
    best = int(np.argmax(dev_scores))
    test_base = acc_on(test, CANDS[0])
    test_best = test_base if best == 0 else acc_on(test, CANDS[best])
    res = {"task": dev[0]["task"], "n_dev": len(dev), "n_test": len(test),
           "dev_scores": [round(x, 4) for x in dev_scores], "best_cand": best,
           "dev_best": round(dev_scores[best], 4), "test_base": round(test_base, 4),
           "test_best": round(test_best, 4), "test_gain": round(test_best - test_base, 4),
           "rel_gain": round((test_best - test_base) / max(1e-9, test_base), 4),
           "transfer": round(test_best / max(1e-9, dev_scores[best]), 4)}
    print(f"  == {name}: best=cand{best} test_base={test_base:.3f} test_best={test_best:.3f} "
          f"gain={res['test_gain']:+.3f} (rel {res['rel_gain']:+.1%}) transfer={res['transfer']:.2f}", flush=True)
    return res


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print(f"server={BASE} ndev={N_DEV} ntest={N_TEST} cands={len(CANDS)}", flush=True)
    out = json.load(open(OUT)).get("results", {}) if OUT.exists() else {}
    for name, loader in p2.LOADERS.items():
        if only and name not in only:
            continue
        try:
            out[name] = run_ds(name, loader)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "lever": "global prompt (system) optimization",
                               "n_dev": N_DEV, "n_test": N_TEST, "n_cands": len(CANDS), "seed": SEED},
                   "results": out,
                   "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/p3b_promptopt.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== E8 PROMPT-OPT ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("E8_DONE", flush=True)


if __name__ == "__main__":
    main()

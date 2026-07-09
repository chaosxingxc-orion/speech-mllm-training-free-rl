"""P3/E7 — multimodal few-shot ICL harness (the owner's key lever) + b1/b2 floor + leakage guard.

Adjusts the conditioning A by prepending k (audio, question, ANSWER) demonstrations drawn from a TRAIN
pool (disjoint from the query set — leakage guard), then measures whether greedy accuracy on the query
rises toward the oracle ceiling as k grows (the good answer becoming modal via A, no per-instance
selector). b1/b2 control: a shuffled-label arm shows the same demos with RANDOM answers — a gain that
survives over this floor is genuine task-learning (b2), not format alignment (b1).

Reuses p2_baselines loaders/reward/gen helpers. Logs prompt-token cost per shot (context budget).

reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/p3_fewshot.py --only mmau-mini [--shots 0,1,2,4]
"""
import base64, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2  # gen helpers, loaders, reward, norm, LETTERS, MINDS14

BASE = p2.BASE
SHOTS = [int(x) for x in os.environ.get("E7_SHOTS", "0,1,2,4").split(",")]
K_POOL = int(os.environ.get("E7_POOL", "20"))       # train exemplar pool size
N_QUERY = int(os.environ.get("E7_NQUERY", "120"))   # query items
SEED = 20260705
OUT = p2.DATA / "_repro" / "e7_fewshot.json"
import urllib.request


def answer_text(it):
    if it["task"] == "mcq":
        return f"{p2.LETTERS[it['gold']]}. {it['opts'][it['gold']]}"
    if it["task"] == "slu":
        return f"{it['gold']}. {p2.MINDS14[it['gold']]}"
    return str(it["gold"])  # qa


def _b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def gen_fewshot(exemplars, q, seed, temp):
    """exemplars: list of items shown as (audio, instr, answer). Returns (text, prompt_tokens)."""
    content = []
    for ex, ans in exemplars:
        content.append({"type": "input_audio", "input_audio": {"data": _b64(ex["wav"]), "format": "wav"}})
        content.append({"type": "text", "text": f"{ex['instr']}\nCorrect answer: {ans}"})
    content.append({"type": "input_audio", "input_audio": {"data": _b64(q["wav"]), "format": "wav"}})
    content.append({"type": "text", "text": q["instr"]})
    payload = {"messages": [{"role": "user", "content": content}],
               "max_tokens": p2.MAXTOK, "seed": seed, "temperature": temp}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"].strip(), d.get("usage", {}).get("prompt_tokens", -1)


def run_ds(name, loader):
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = K_POOL + N_QUERY + 40
    items = loader(rng)
    if len(items) < K_POOL + 10:
        return {"error": f"too few items ({len(items)})"}
    pool, query = items[:K_POOL], items[K_POOL:K_POOL + N_QUERY]
    # precompute exemplar answers (true) and a shuffled-label floor (b1)
    pool_ans = [answer_text(x) for x in pool]
    shuf = list(pool_ans)
    rng.shuffle(shuf)
    res = {"task": pool[0]["task"], "n_query": len(query), "shots": {}, "tok": {}}
    t0 = time.time()
    for k in SHOTS:
        exsel = lambda qi: [(pool[(qi + j) % K_POOL], pool_ans[(qi + j) % K_POOL]) for j in range(k)]
        exsel_b1 = lambda qi: [(pool[(qi + j) % K_POOL], shuf[(qi + j) % K_POOL]) for j in range(k)]
        acc, accb1, toks = [], [], []
        for qi, q in enumerate(query):
            try:
                if k == 0:
                    txt, tk = gen_fewshot([], q, 42, 0.0)
                    a = p2.reward(q, txt); ab1 = a
                else:
                    txt, tk = gen_fewshot(exsel(qi), q, 42, 0.0)
                    a = p2.reward(q, txt)
                    txtb1, _ = gen_fewshot(exsel_b1(qi), q, 42, 0.0)
                    ab1 = p2.reward(q, txtb1)
            except Exception as e:
                continue
            acc.append(a); accb1.append(ab1); toks.append(tk)
        res["shots"][k] = {"greedy_b2": round(float(np.mean(acc)), 4),
                           "greedy_b1_floor": round(float(np.mean(accb1)), 4),
                           "b2_minus_b1": round(float(np.mean(acc) - np.mean(accb1)), 4), "n": len(acc)}
        res["tok"][k] = int(np.median(toks)) if toks else -1
        print(f"  [{name} k={k}] b2={np.mean(acc):.3f} b1={np.mean(accb1):.3f} "
              f"med_tok={res['tok'][k]} ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    if "--shots" in sys.argv:
        global SHOTS
        SHOTS = [int(x) for x in sys.argv[sys.argv.index("--shots") + 1].split(",")]
    print(f"server={BASE} shots={SHOTS} pool={K_POOL} nquery={N_QUERY}", flush=True)
    out = json.load(open(OUT)).get("results", {}) if OUT.exists() else {}
    for name, loader in p2.LOADERS.items():
        if only and name not in only:
            continue
        try:
            out[name] = run_ds(name, loader)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "lever": "multimodal few-shot ICL",
                               "shots": SHOTS, "pool": K_POOL, "nquery": N_QUERY, "seed": SEED},
                   "results": out,
                   "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/p3_fewshot.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== E7 FEW-SHOT ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("E7_DONE", flush=True)


if __name__ == "__main__":
    main()

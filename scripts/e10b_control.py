"""E10b — the on-surface self-selection CONTROL the strict review demanded (de-confounds E10).

On the SAME surfaces/pools, compare, all label-free, at matched budget, WITH paired-bootstrap CIs:
  greedy · oracle-over-N · MAJORITY-vote (the self-selection control) · two-system ISOLATED verifier.
This answers the confound "is the two-system verifier's realization real, or would on-surface
self-selection get the same?" — the question E4-vs-E10 (different surfaces) could not.

reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/e10b_control.py
"""
import base64, collections, json, os, re, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

BASE = p2.BASE
NSAMP = int(os.environ.get("E10B_NSAMP", "5"))
N_QUERY = int(os.environ.get("E10B_NQUERY", "48"))
TEMP = 0.7
SEED = 20260705
NBOOT = 10000
OUT = p2.DATA / "_repro" / "e10b_control.json"
ONLY = os.environ.get("E10B_ONLY", "mmau-mini,SQuAD-zh,big-bench-audio").split(",")
SYS_ISO = ("You are a strict, independent verifier. You did NOT write the candidate answer. Judge ONLY "
           "whether the candidate correctly answers the spoken question, using the audio. Respond YES or NO.")


def gen(wav, instr, seed, temp):
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    p = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instr}]}], "max_tokens": p2.MAXTOK, "seed": seed, "temperature": temp}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as z:
        return json.loads(z.read().decode())["choices"][0]["message"]["content"].strip()


def verify(wav, instr, cand, seed=7):
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    m = [{"role": "system", "content": SYS_ISO},
         {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
          {"type": "text", "text": f"{instr}\nCandidate answer: {cand}\nIs it correct? YES or NO."}]}]
    p = {"messages": m, "max_tokens": 4, "seed": seed, "temperature": 0.0}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as z:
        return 1 if json.loads(z.read().decode())["choices"][0]["message"]["content"].strip().upper().startswith("Y") else 0


def ans_key(item, text):
    if item["task"] == "mcq":
        return str(p2._parse_choice(text, item["opts"]))
    return p2.norm(text)[:40]  # qa: normalized short answer


def majority_pick(item, cands):
    keys = [ans_key(item, c) for c in cands]
    common = collections.Counter(keys).most_common(1)[0][0]
    return cands[keys.index(common)]


def boot(diffs, seed=1):
    rng = np.random.default_rng(seed)
    ms = np.array([np.mean(diffs[rng.integers(0, len(diffs), len(diffs))]) for _ in range(NBOOT)])
    return [round(float(np.quantile(ms, .025)), 4), round(float(np.quantile(ms, .975)), 4)]


def run(name, loader):
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = N_QUERY + 20
    items = loader(rng)[:N_QUERY]
    t0 = time.time()
    g, orc, maj, ver = [], [], [], []
    for k, it in enumerate(items):
        try:
            gt = gen(it["wav"], it["instr"], 42, 0.0)
            cands = [gen(it["wav"], it["instr"], 100 + s, TEMP) for s in range(NSAMP)]
            gr = p2.reward(it, gt); rs = [p2.reward(it, c) for c in cands]
            mj = p2.reward(it, majority_pick(it, cands))
            v = [verify(it["wav"], it["instr"], c) for c in cands]
            yes = [i for i, x in enumerate(v) if x]
            vr = p2.reward(it, cands[yes[0]] if yes else cands[0])
        except Exception:
            continue
        g.append(gr); orc.append(int(gr or any(rs))); maj.append(mj); ver.append(vr)
        if (k + 1) % 24 == 0:
            print(f"  [{name} {k+1}] g={np.mean(g):.3f} orc={np.mean(orc):.3f} maj={np.mean(maj):.3f} ver={np.mean(ver):.3f} ({time.time()-t0:.0f}s)", flush=True)
    g, orc, maj, ver = map(np.array, (g, orc, maj, ver))
    n = len(g)
    res = {"n": n, "greedy": round(float(g.mean()), 4), "oracle": round(float(orc.mean()), 4),
           "majority": round(float(maj.mean()), 4), "verifier_iso": round(float(ver.mean()), 4),
           "maj_minus_greedy": round(float((maj - g).mean()), 4), "maj_vs_greedy_CI": boot(maj - g),
           "ver_minus_greedy": round(float((ver - g).mean()), 4), "ver_vs_greedy_CI": boot(ver - g),
           "ver_minus_maj": round(float((ver - maj).mean()), 4), "ver_vs_maj_CI": boot(ver - maj),
           "ver_rel_gain": round(float((ver.mean() - g.mean()) / max(1e-9, g.mean())), 4)}
    print(f"  == {name}: greedy={res['greedy']} oracle={res['oracle']} | majority={res['majority']} "
          f"verifier={res['verifier_iso']} | ver-maj={res['ver_minus_maj']:+.3f} CI{res['ver_vs_maj_CI']}", flush=True)
    return res


def main():
    print(f"server={BASE} N={NSAMP} nquery={N_QUERY} surfaces={ONLY}", flush=True)
    out = json.load(open(OUT)).get("results", {}) if OUT.exists() else {}
    for name in ONLY:
        if name not in p2.LOADERS:
            continue
        try:
            out[name] = run(name, p2.LOADERS[name])
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "control": "on-surface self-selection (majority) vs two-system verifier, with paired-bootstrap CIs",
                               "nsamp": NSAMP, "n_query": N_QUERY, "seed": SEED}, "results": out,
                   "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/e10b_control.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== E10b CONTROL ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("E10B_DONE", flush=True)


if __name__ == "__main__":
    main()

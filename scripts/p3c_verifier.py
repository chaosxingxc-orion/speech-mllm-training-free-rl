"""E10 — generator/verifier as TWO context-differentiated systems (not self-reward); decorrelation.

Per query: generator S_G draws N candidates (temp 0.7). Verifier S_V (SAME frozen weights, DISTINCT
system-prompt + isolated context: only the question+candidate, never the generation chain) scores each
candidate correct/not. Selection = a verifier-endorsed candidate. Measures:
  - greedy (baseline), oracle-over-N (ceiling)
  - verifier-selected accuracy -> rho = realized fraction of oracle-delta
  - ISOLATED vs COUPLED ablation (coupled = verifier told 'you produced this answer') -> does context
    differentiation decorrelate the verifier from the generator (lower tau, higher rho)? (TH2a / C4)
  - catch-rate on the generator's errors (the decorrelation signal).

Training-free, no model training (owner). The residual knowledge-blind-spot floor is PARKED (#37 -> W4).

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/p3c_verifier.py --only mmau-mini
"""
import base64, json, os, re, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

BASE = p2.BASE
NSAMP = int(os.environ.get("E10_NSAMP", "6"))
N_QUERY = int(os.environ.get("E10_NQUERY", "100"))
TEMP = float(os.environ.get("E10_TEMP", "0.7"))
SEED = 20260705
OUT = p2.DATA / "_repro" / "e10_verifier.json"

SYS_ISO = ("You are a strict, independent verifier. You did NOT write the candidate answer. Judge ONLY "
           "whether the candidate correctly answers the spoken question, using the audio as evidence. "
           "Respond with exactly one word: YES or NO.")
SYS_COUP = ("You produced the following candidate answer to the spoken question. Re-check whether YOUR "
            "answer is correct, using the audio. Respond with exactly one word: YES or NO.")


def gen(wav, instr, seed, temp):
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instr}]}], "max_tokens": p2.MAXTOK, "seed": seed, "temperature": temp}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def verify(wav, instr, cand, system, seed=7):
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                {"type": "text", "text": f"{instr}\nCandidate answer: {cand}\nIs the candidate correct? YES or NO."}]}]
    payload = {"messages": msgs, "max_tokens": 4, "seed": seed, "temperature": 0.0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        t = json.loads(r.read().decode())["choices"][0]["message"]["content"].strip().upper()
    return 1 if t.startswith("Y") else 0


def select(cands_txt, votes):
    yes = [i for i, v in enumerate(votes) if v == 1]
    return cands_txt[yes[0]] if yes else cands_txt[0]  # first endorsed, else first candidate


def run_ds(name, loader):
    rng = np.random.default_rng(SEED)
    p2.N_UTTS = N_QUERY + 30
    items = loader(rng)[:N_QUERY]
    t0 = time.time()
    greedy, oracle, sel_iso, sel_coup, caught_iso, ngen_wrong = [], [], [], [], 0, 0
    for k, it in enumerate(items):
        try:
            g_txt = gen(it["wav"], it["instr"], 42, 0.0)
            cands = [gen(it["wav"], it["instr"], 100 + s, TEMP) for s in range(NSAMP)]
            gr = p2.reward(it, g_txt); rs = [p2.reward(it, c) for c in cands]
            v_iso = [verify(it["wav"], it["instr"], c, SYS_ISO) for c in cands]
            v_coup = [verify(it["wav"], it["instr"], c, SYS_COUP) for c in cands]
        except Exception:
            continue
        greedy.append(gr); oracle.append(int(gr or any(rs)))
        sel_iso.append(p2.reward(it, select(cands, v_iso)))
        sel_coup.append(p2.reward(it, select(cands, v_coup)))
        if not gr and any(rs):  # generator (greedy) wrong but a correct candidate exists
            ngen_wrong += 1
            # did the isolated verifier endorse a correct candidate first?
            if p2.reward(it, select(cands, v_iso)) == 1:
                caught_iso += 1
        if (k + 1) % 50 == 0:
            print(f"    [{name} {k+1}] greedy={np.mean(greedy):.3f} oracle={np.mean(oracle):.3f} "
                  f"iso={np.mean(sel_iso):.3f} coup={np.mean(sel_coup):.3f} ({time.time()-t0:.0f}s)", flush=True)
    n = len(greedy); g, o = float(np.mean(greedy)), float(np.mean(oracle))
    si, sc = float(np.mean(sel_iso)), float(np.mean(sel_coup))
    dd = max(1e-9, o - g)
    res = {"task": items[0]["task"], "n": n, "greedy": round(g, 4), "oracle": round(o, 4),
           "oracle_delta": round(o - g, 4),
           "verifier_isolated": round(si, 4), "rho_isolated": round((si - g) / dd, 4),
           "verifier_coupled": round(sc, 4), "rho_coupled": round((sc - g) / dd, 4),
           "decorrelation_catch_rate": round(caught_iso / max(1, ngen_wrong), 4),
           "n_gen_wrong_with_fix": ngen_wrong,
           "isolation_gain_over_coupled": round(si - sc, 4)}
    print(f"  == {name}: greedy={g:.3f} oracle={o:.3f} | iso={si:.3f}(rho={res['rho_isolated']:.2f}) "
          f"coup={sc:.3f}(rho={res['rho_coupled']:.2f}) catch={res['decorrelation_catch_rate']:.2f}", flush=True)
    return res


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print(f"server={BASE} N={NSAMP} nquery={N_QUERY}", flush=True)
    out = json.load(open(OUT)).get("results", {}) if OUT.exists() else {}
    for name, loader in p2.LOADERS.items():
        if only and name not in only:
            continue
        try:
            out[name] = run_ds(name, loader)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "lever": "generator/verifier two-system (decorrelation)",
                               "nsamp": NSAMP, "n_query": N_QUERY, "seed": SEED,
                               "note": "isolated vs coupled verifier; residual knowledge floor PARKED (#37->W4)"},
                   "results": out,
                   "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/p3c_verifier.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== E10 VERIFIER ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("E10_DONE", flush=True)


if __name__ == "__main__":
    main()

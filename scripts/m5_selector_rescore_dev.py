"""M5 selector-accumulation DEV grid (CPU-only; re-scores the committed C1 pools; deterministic).

Pre-registered under wiki/2026-07-03-agentic-tfrl-step1-preregistration.md (freeze b19bff2) §5 M5,
implemented per the architect-hardened protocol spec §C.2 (interpretation log #7/#8):
  - DEV ONLY: the 144-utterance C1 pool is dev-spent; nothing here is confirmatory.
  - B0 axis (b) — estimating R. Ontology pinned: with fixed q0 and fixed oracle R, any selector's
    realized gain <= oracle headroom; M5 never raises the ceiling, it closes the realization gap.
  - Sessions = speakers (31), ordered by (chapter, utt); generation seeds POOLED (the 3 seed slices
    hold disjoint utterances — parallel replicas are impossible on dev). Streaming and label-free:
    at position p the selector sees only the current 8-candidate pool + memory from positions < p,
    built exclusively from its own previous picks. Refs enter only in evaluation.
  - Variants: V1 accumulated rare-token lexicon bonus (lambda x gate grid, 8 configs);
    V2 kNN utility regression over accumulated (raretok, z-utility) tuples (alpha grid, 3);
    V3 smoothed session unigram prior over rare tokens (lambda grid, 3).
  - Winner rule (pre-committed): max mean(mbr_wer_8 - sel_wer_8); ties < 0.002 -> simpler
    (V1 > V3 > V2; smaller lambda/alpha; gate none before consensus2). EXACTLY ONE config
    proceeds to the confirmatory slice (keeps it Holm-free).
  - Diagnostics on the winner: Goodhart by-N trajectory; speaker-derangement shuffled-memory
    (size-matched snapshots, receiver never self-updates) — dev diagnostic only, the binding
    ablation is confirmatory.

reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m5_selector_rescore_dev.py
"""
import collections, copy, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/d/chao_workspace/exploring-l4-intelligence/common/src")
from speechrl_common.rl.reward import wer, _normalize

DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
ART = Path(__file__).resolve().parent.parent / "_repro" / "asr_bon_llamacpp_snr5.json"
FREQ_CACHE = DATA / "_repro" / "librispeech_train960_tokfreq.json"
OUT_LOCAL = DATA / "_repro" / "m5_selector_dev.json"
TRAIN_MAX = 5
TIE_EPS = 0.002
Ns = [1, 2, 4, 8]

_freq = json.load(open(FREQ_CACHE))


def norm_tokens(s: str) -> list[str]:
    toks = []
    for t in _normalize(s).upper().split():
        if t.endswith("'S"):
            t = t[:-2]
        toks.append(t.replace("'", ""))
    return [t for t in toks if t]


def raretok(s: str) -> frozenset:
    return frozenset(t for t in norm_tokens(s) if len(t) >= 4 and _freq.get(t, 0) <= TRAIN_MAX)


def zscore(xs: list[float]) -> list[float]:
    a = np.array(xs, dtype=float)
    sd = a.std()
    return list((a - a.mean()) / sd) if sd > 0 else [0.0] * len(a)


def mbr_utils(pool: list[str]) -> list[float]:
    """U(c) = mean_{c'!=c} -wer(c', c) — exactly the repo mbr utility."""
    return [float(np.mean([-wer(c2, c) for j2, c2 in enumerate(pool) if j2 != j]))
            for j, c in enumerate(pool)]


# ---------------- selector variants (streaming; memory = opaque state dict) ----------------

def v1_new(): return {"L": set()}

def v1_score(mem, pool, z, lam, gate):
    return [z[j] + lam * len(raretok(c) & mem["L"]) / max(1, len(norm_tokens(c)))
            for j, c in enumerate(pool)]

def v1_update(mem, pool, pick, lam, gate):
    rt = raretok(pool[pick])
    if gate == "consensus2":
        support = collections.Counter(t for c in pool for t in raretok(c))
        rt = frozenset(t for t in rt if support[t] >= 2)
    mem["L"] |= rt


def v2_new(): return {"M": []}   # list of (raretok_set, z_utility)

def v2_score(mem, pool, z, alpha, _):
    out = []
    for j, c in enumerate(pool):
        rc = raretok(c)
        if rc and mem["M"]:
            sims = []
            for rs, u in mem["M"]:
                un = len(rc | rs)
                sims.append((len(rc & rs) / un if un else 0.0, u))
            sims.sort(key=lambda t: -t[0])
            top = [t for t in sims[:8] if t[0] > 0]
            uhat = (sum(s * u for s, u in top) / sum(s for s, _ in top)) if top else 0.0
        else:
            uhat = 0.0
        out.append((1 - alpha) * z[j] + alpha * uhat)
    return out

def v2_update(mem, pool, pick, alpha, _):
    z = zscore(mbr_utils(pool))
    mem["M"].extend((raretok(c), z[j]) for j, c in enumerate(pool))


def v3_new(): return {"C": collections.Counter(), "seen": set()}

def v3_score(mem, pool, z, lam, _):
    vocab = set(mem["seen"])
    for c in pool:
        vocab |= raretok(c)
    V = max(1, len(vocab))
    ctot = sum(mem["C"].values())
    priors = []
    for c in pool:
        rc = raretok(c)
        priors.append(sum(math.log((mem["C"][w] + 1) / (ctot + V)) for w in rc) / len(rc)
                      if rc else 0.0)
    zp = zscore(priors)
    return [z[j] + lam * zp[j] for j in range(len(pool))]

def v3_update(mem, pool, pick, lam, _):
    rt = raretok(pool[pick])
    mem["C"].update(rt)
    mem["seen"] |= rt


VARIANTS = {"V1": (v1_new, v1_score, v1_update), "V2": (v2_new, v2_score, v2_update),
            "V3": (v3_new, v3_score, v3_update)}
GRID = ([("V1", lam, gate) for lam in (0.05, 0.1, 0.2, 0.4) for gate in ("none", "consensus2")]
        + [("V2", a, None) for a in (0.1, 0.2, 0.4)]
        + [("V3", lam, None) for lam in (0.25, 0.5, 1.0)])
SIMPLICITY = {"V1": 0, "V3": 1, "V2": 2}


def cfg_key(v, p, g): return f"{v}|{p}|{g or '-'}"


def mem_size(mem) -> int:
    if "L" in mem:
        return len(mem["L"])
    if "M" in mem:
        return len(mem["M"])
    if "C" in mem:
        return int(sum(mem["C"].values()))
    return 0


def run_config(sessions, variant, param, gate, N=8, snapshots=False, donor_snaps=None, donor_map=None):
    """Streaming pass. Returns rows [(id, sel_wer, pick_idx, proxy, mem_items)], optional snapshots.
    donor_snaps/donor_map: shuffled-memory arm — score with the donor speaker's snapshot at
    min(p, donor_len); NEVER update own memory."""
    new, score, update = VARIANTS[variant]
    out, snaps = {}, {}
    for spk, utts in sessions.items():
        mem = new()
        snaps[spk] = []
        for p, u in enumerate(utts):
            pool = u["pool"][:N]
            z = zscore(mbr_utils(pool))
            if donor_snaps is not None:
                d = donor_map[spk]
                ds = donor_snaps[d]
                dmem = ds[min(p, len(ds) - 1)] if ds else new()
                sc = score(dmem, pool, z, param, gate)
            else:
                sc = score(mem, pool, z, param, gate)
            pick = int(np.argmax(sc))  # ties -> lowest index
            out[u["id"] + "|" + str(u["gen_seed"])] = {
                "sel_wer": wer(u["ref"], pool[pick]), "pick": pick,
                "proxy": float(sc[pick]),
                "mem_items": mem_size(mem) if donor_snaps is None else -1}
            if donor_snaps is None:
                update(mem, pool, pick, param, gate)
                if snapshots:
                    snaps[spk].append(copy.deepcopy(mem))
    return out, snaps


def main():
    t0 = time.time()
    art = json.load(open(ART))
    rows = art["per_utt"]
    key = lambda u: u["id"] + "|" + str(u["gen_seed"])
    base = {key(u): u for u in rows}
    sessions = collections.defaultdict(list)
    for u in rows:
        sessions[u["id"].split("-")[0]].append(u)
    for spk in sessions:
        sessions[spk].sort(key=lambda u: (int(u["id"].split("-")[1]), int(u["id"].split("-")[2])))
    print(f"dev pool: {len(rows)} rows, {len(sessions)} speaker sessions "
          f"(depth {min(map(len, sessions.values()))}-{max(map(len, sessions.values()))})", flush=True)

    # sanity: MBR reproduction from pools must match the committed artifact
    mbr_re = []
    for u in rows:
        z = mbr_utils(u["pool"])
        mbr_re.append(wer(u["ref"], u["pool"][int(np.argmax(z))]))
    art_mbr = float(np.mean([u["mbr_wer_8"] for u in rows]))
    assert abs(float(np.mean(mbr_re)) - art_mbr) < 1e-9, \
        f"MBR reproduction mismatch: {np.mean(mbr_re):.6f} vs artifact {art_mbr:.6f}"
    print(f"MBR reproduction check OK ({art_mbr:.4f})", flush=True)

    results = {}
    for variant, param, gate in GRID:
        out, _ = run_config(sessions, variant, param, gate)
        red = float(np.mean([base[k]["mbr_wer_8"] - v["sel_wer"] for k, v in out.items()]))
        results[cfg_key(variant, param, gate)] = {"red_vs_mbr": round(red, 5), "out": out}
        print(f"  {cfg_key(variant, param, gate):22s} red_vs_mbr={red:+.5f}", flush=True)

    # winner rule (pre-committed)
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["red_vs_mbr"])
    best_red = ranked[0][1]["red_vs_mbr"]
    contenders = [k for k, v in ranked if best_red - v["red_vs_mbr"] < TIE_EPS]
    def simplicity(k):
        v, p, g = k.split("|")
        return (SIMPLICITY[v], float(p), 0 if g in ("-", "none") else 1)
    winner = sorted(contenders, key=simplicity)[0]
    wv, wp, wg = winner.split("|")
    wg = None if wg == "-" else wg
    tie_break = winner != ranked[0][0]
    print(f"winner: {winner} (tie_break={tie_break}; contenders={contenders})", flush=True)

    # winner diagnostics: Goodhart by N + dev shuffled-memory (speaker derangement)
    goodhart = {}
    for N in Ns:
        out, _ = run_config(sessions, wv, float(wp), wg, N=N)
        goodhart[str(N)] = {
            "true_wer": round(float(np.mean([v["sel_wer"] for v in out.values()])), 5),
            "proxy": round(float(np.mean([v["proxy"] for v in out.values()])), 5)}
    _, snaps = run_config(sessions, wv, float(wp), wg, snapshots=True)
    spks = sorted(sessions)
    donor = {spks[i]: spks[(i + 1) % len(spks)] for i in range(len(spks))}
    shuf, _ = run_config(sessions, wv, float(wp), wg, donor_snaps=snaps, donor_map=donor)
    wout = results[winner]["out"]
    gain_sel = float(np.mean([base[k]["mbr_wer_8"] - v["sel_wer"] for k, v in wout.items()]))
    gain_shuf = float(np.mean([base[k]["mbr_wer_8"] - v["sel_wer"] for k, v in shuf.items()]))
    retained = gain_shuf / gain_sel if gain_sel > 0 else None

    baselines = {"greedy_wer": round(float(np.mean([u["greedy_wer"] for u in rows])), 4),
                 "mbr_wer_8": round(art_mbr, 4),
                 "oracle_wer_8": round(float(np.mean([u["oracle_wer_8"] for u in rows])), 4)}
    per_utt = []
    for u in rows:
        k = key(u)
        spk_list = sessions[u["id"].split("-")[0]]
        per_utt.append({"id": u["id"], "gen_seed": u["gen_seed"], "speaker": u["id"].split("-")[0],
                        "session_pos": next(i + 1 for i, x in enumerate(spk_list) if key(x) == k),
                        "greedy_wer": u["greedy_wer"], "mbr_wer_8": u["mbr_wer_8"],
                        "oracle_wer_8": u["oracle_wer_8"],
                        "sel_wer_8_by_config": {c: results[c]["out"][k]["sel_wer"] for c in results},
                        "winner_pick_idx": wout[k]["pick"], "winner_shuf_wer_8": shuf[k]["sel_wer"]})
    grid_summary = {}
    for v in ("V1", "V2", "V3"):
        cfgs = {k: r["red_vs_mbr"] for k, r in results.items() if k.startswith(v)}
        bk = max(cfgs, key=cfgs.get)
        grid_summary[v] = {"configs": len(cfgs), "best": {"config": bk, "red_vs_mbr": cfgs[bk]}}
    summary = {
        "lane": "M5", "axis": "b (estimating R)", "status": "DEV-ONLY (spent 144 pool)",
        "dev_pool": "_repro/asr_bon_llamacpp_snr5.json", "n_utts": len(rows),
        "n_sessions": len(sessions),
        "session_def": "speaker; order (chapter,utt); gen-seeds pooled (disjoint utt slices)",
        "baselines": baselines, "grid": grid_summary,
        "winner": {"config": winner, "variant": wv, "param": float(wp), "gate": wg,
                   "dev_red_vs_mbr": round(gain_sel, 5),
                   "dev_red_vs_greedy": round(float(np.mean(
                       [base[k]["greedy_wer"] - v["sel_wer"] for k, v in wout.items()])), 5),
                   "tie_break_applied": tie_break,
                   "rule": "max mean(mbr8-sel8); <0.002 tie -> simpler (V1>V3>V2)"},
        "diagnostics": {"dev_shuffled_gain_retained": round(retained, 4) if retained is not None else None,
                        "dev_gain_sel": round(gain_sel, 5), "dev_gain_shuf": round(gain_shuf, 5),
                        "shuffle_def": "speaker rotation derangement, size-matched snapshots, no self-update",
                        "goodhart_by_N": goodhart},
        "prereg": "wiki/2026-07-03-agentic-tfrl-step1-preregistration.md @ b19bff2 §5 M5",
    }
    out = {"summary": summary, "per_utt": per_utt, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m5_selector_rescore_dev.py "
                        "(CPU-only; re-scores the committed C1 pools; deterministic)"}
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_LOCAL, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT_LOCAL, flush=True)
    print("M5_DEV_DONE", flush=True)


if __name__ == "__main__":
    main()

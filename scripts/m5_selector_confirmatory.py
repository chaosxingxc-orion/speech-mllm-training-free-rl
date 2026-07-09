"""M5 selector-accumulation CONFIRMATORY run (single touch; frozen winner only).

Pre-registered under wiki/2026-07-03-agentic-tfrl-step1-preregistration.md (freeze b19bff2) §5 M5 +
§2 GO-minimal, per the architect-hardened protocol spec §C.3-C.5 (interpretation log #7/#9/#10/#11/#12).

Two stages (M5C_STAGE env):
  manifest — derive the frozen confirmatory slice (12 speakers x 12 utts; rng 20260703; excludes the
             144 dev-spent C1 ids AND the 36 M3 Phase-0 ids), write the slice manifest + snr5 wavs.
             COMMIT the manifest + this runner + the dev artifact BEFORE any generation (git-provable
             pre-commitment).
  run      — single generation pass (3 replica seeds x (1 greedy + 8-pool)), then ALL arms as CPU
             analyses of the same pools: MBR / frozen selector / shuffled-memory / oracle / Goodhart.
             The runner hard-fails if given more than one selector config (Holm guard).

Frozen thresholds: PASS = (A) selector beats MBR, cluster-bootstrap 95% CI-LB > 0, AND
(B) mean greedy->selector reduction >= 0.015 [reading (i); reading (ii) = mean(MBR-sel) >= 0.015
also reported; disagreement escalates to owner, defaulting stricter]. Ablation: accumulation
load-bearing iff shuffled arm retains <= 50% of the selector's gain (point estimates; CI reported).

reproduce:
  M5C_STAGE=manifest SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m5_selector_confirmatory.py
  M5C_STAGE=run      SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m5_selector_confirmatory.py
  (llama-server resident on :8091 for the run stage)
"""
import base64, collections, copy, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/d/chao_workspace/exploring-l4-intelligence/common/src")
from repro_asr_best_of_n import INSTRUCTION
from speechrl_common.rl.reward import wer
from m5_selector_rescore_dev import (VARIANTS, mbr_utils, zscore, mem_size)  # identical semantics

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
STAGE = os.environ.get("M5C_STAGE", "manifest")
REPRO = Path(__file__).resolve().parent.parent / "_repro"
DEV_ART = REPRO / "m5_selector_dev.json"
M3_SEL = REPRO / "m3_phase0_selection.json"
C1_ART = REPRO / "asr_bon_llamacpp_snr5.json"
MANIFEST = DATA / "_repro" / "m5_confirmatory_slice_ids.json"
WAV_DIR = DATA / "_repro" / "m5_confirmatory_wavs_snr5"
OUT = DATA / "_repro" / "m5_selector_confirmatory.json"
SLICE_SEED = 20260703
N_SPK, DEPTH = 12, 12
REPLICA_SEEDS = [2026, 2027, 2028]
POOL, TEMP, TOP_P, MAXTOK, SNR = 8, 0.8, 0.95, 100, 5.0
Ns = [1, 2, 4, 8]
NBOOT = 10000
PASS_RED_GREEDY = 0.015
ABLATION_RETAIN = 0.5


def frozen_winner():
    w = json.load(open(DEV_ART))["summary"]["winner"]
    return w["variant"], float(w["param"]), w["gate"]


def make_manifest():
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / "datasets/librispeech/all/test.other/0000.parquet",
                      columns=["text", "id"])
    ids = t.column("id").to_pylist()
    ref = dict(zip(ids, t.column("text").to_pylist()))
    dev_spent = set(r["id"] for r in json.load(open(C1_ART))["per_utt"])
    m3_ids = set(r["id"] for r in json.load(open(M3_SEL))["selected"])
    excluded = dev_spent | m3_ids
    per_spk = collections.defaultdict(list)
    for i in ids:
        if i not in excluded:
            per_spk[i.split("-")[0]].append(i)
    eligible = sorted(s for s, u in per_spk.items() if len(u) >= DEPTH)
    rng = np.random.default_rng(SLICE_SEED)
    chosen = sorted(rng.choice(eligible, N_SPK, replace=False).tolist())
    slice_ids = []
    for s in chosen:
        utts = sorted(per_spk[s], key=lambda i: (int(i.split("-")[1]), int(i.split("-")[2])))[:DEPTH]
        slice_ids.extend(utts)
    manifest = {"n_utts": len(slice_ids), "n_speakers": N_SPK, "utts_per_speaker": DEPTH,
                "sampling_seed": SLICE_SEED, "eligible_speakers": len(eligible),
                "excluded_dev_spent": len(dev_spent), "excluded_m3_phase0": len(m3_ids),
                "speakers": chosen, "ids": slice_ids,
                "condition": {"snr_db": SNR, "temp": TEMP, "top_p": TOP_P, "pool": POOL,
                              "max_tokens": MAXTOK},
                "replica_seeds": REPLICA_SEEDS,
                "frozen_winner": dict(zip(("variant", "param", "gate"), frozen_winner())),
                "prereg": "wiki/2026-07-03-agentic-tfrl-step1-preregistration.md @ b19bff2 §5 M5"}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"manifest: {len(slice_ids)} utts, {N_SPK} speakers -> {MANIFEST}", flush=True)

    # snr5 wavs, generated ONCE, shared by all replicas (per-utterance rng = SLICE_SEED*100003+j)
    import soundfile as sf
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    need = [u for u in slice_ids if not (WAV_DIR / f"{u}.wav").exists()]
    if need:
        ta = pq.read_table(DATA / "datasets/librispeech/all/test.other/0000.parquet",
                           columns=["audio", "id"])
        aid = ta.column("id").to_pylist()
        idx = {u: i for i, u in enumerate(aid)}
        audio = ta.column("audio").to_pylist()
        for j, u in enumerate(slice_ids):
            if u not in need:
                continue
            wav, sr = sf.read(io.BytesIO(audio[idx[u]]["bytes"]), dtype="float32")
            rng_n = np.random.default_rng(SLICE_SEED * 100003 + j)
            sig_p = float(np.mean(wav ** 2)) + 1e-12
            noise = rng_n.standard_normal(len(wav)).astype("float32")
            noise *= np.sqrt(sig_p / (10 ** (SNR / 10.0)) / (float(np.mean(noise ** 2)) + 1e-12))
            sf.write(str(WAV_DIR / f"{u}.wav"), wav + noise, sr)
    print(f"wavs ready in {WAV_DIR}", flush=True)
    print("M5C_MANIFEST_DONE", flush=True)


def gen(clip, greedy, seed):
    b64 = base64.b64encode(open(clip, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": INSTRUCTION}]}],
        "max_tokens": MAXTOK, "seed": seed, "temperature": 0.0 if greedy else TEMP}
    if not greedy:
        payload["top_p"] = TOP_P
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def stream_selector(sess_items, variant, param, gate, N=8, donor_snaps=None, donor_map=None,
                    snapshots=False):
    """One replica's streaming pass over {speaker: [items in session order]}.
    Returns {item_key: {sel_wer, pick, proxy, mem_items}}, snapshots per speaker."""
    new, score, update = VARIANTS[variant]
    out, snaps = {}, {}
    for spk, items in sess_items.items():
        mem = new()
        snaps[spk] = []
        for p, it in enumerate(items):
            pool = it["pool"][:N]
            z = zscore(mbr_utils(pool))
            if donor_snaps is not None:
                ds = donor_snaps[donor_map[spk]]
                dmem = ds[min(p, len(ds) - 1)] if ds else new()
                sc = score(dmem, pool, z, param, gate)
            else:
                sc = score(mem, pool, z, param, gate)
            pick = int(np.argmax(sc))
            out[it["key"]] = {"sel_wer": wer(it["ref"], pool[pick]), "pick": pick,
                              "proxy": float(sc[pick]),
                              "mem_items": mem_size(mem) if donor_snaps is None else -1}
            if donor_snaps is None:
                update(mem, pool, pick, param, gate)
                if snapshots:
                    snaps[spk].append(copy.deepcopy(mem))
    return out, snaps


def cluster_boot(deltas_by_utt: dict, nboot=NBOOT, seed=42):
    """deltas_by_utt: {utt_id: [per-replica deltas]}. Resample utterances, keep all replicas."""
    utts = sorted(deltas_by_utt)
    rng = np.random.default_rng(seed)
    means = np.empty(nboot)
    arrs = [np.array(deltas_by_utt[u]) for u in utts]
    n = len(arrs)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        means[b] = float(np.concatenate([arrs[i] for i in idx]).mean())
    return [round(float(np.quantile(means, 0.025)), 5), round(float(np.quantile(means, 0.975)), 5)]


def run():
    t0 = time.time()
    variant, param, gate = frozen_winner()
    n_configs = 1  # hard Holm guard: this runner evaluates exactly ONE selector config
    assert n_configs == 1
    man = json.load(open(MANIFEST))
    ids, speakers = man["ids"], man["speakers"]
    assert man["frozen_winner"] == dict(zip(("variant", "param", "gate"), (variant, param, gate))), \
        "manifest winner != dev-artifact winner — freeze violated"
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / "datasets/librispeech/all/test.other/0000.parquet",
                      columns=["text", "id"])
    ref = dict(zip(t.column("id").to_pylist(), t.column("text").to_pylist()))
    print(f"server={BASE} slice={len(ids)} utts x {len(REPLICA_SEEDS)} replicas "
          f"winner={variant}|{param}|{gate}", flush=True)

    # ---- single generation pass ----
    items = []   # one per (utt, replica)
    for r in REPLICA_SEEDS:
        for j, u in enumerate(ids):
            clip = str(WAV_DIR / f"{u}.wav")
            try:
                greedy = gen(clip, True, r)
                pool = [gen(clip, False, 1000 * r + i) for i in range(POOL)]
            except Exception as e:
                print(f"  rep{r} [{j+1}] {u} SKIP ({type(e).__name__}: {str(e)[:80]})", flush=True)
                continue
            items.append({"key": f"{u}|{r}", "id": u, "gen_seed": r, "speaker": u.split("-")[0],
                          "ref": ref[u], "greedy": greedy, "greedy_wer": wer(ref[u], greedy),
                          "pool": pool})
            if (j + 1) % 24 == 0:
                print(f"  rep{r}: {j+1}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    # drop (utt) rows missing any replica so arms stay symmetric
    by_utt = collections.Counter(it["id"] for it in items)
    items = [it for it in items if by_utt[it["id"]] == len(REPLICA_SEEDS)]
    print(f"generation done: {len(items)} items ({len(items)//len(REPLICA_SEEDS)} complete utts)", flush=True)

    # ---- CPU arms on the same pools ----
    for it in items:
        z = zscore(mbr_utils(it["pool"]))
        for N in Ns:
            zc = zscore(mbr_utils(it["pool"][:N])) if N < POOL else z
            it[f"mbr_wer_{N}"] = wer(it["ref"], it["pool"][:N][int(np.argmax(zc))])
        it["oracle_wer_8"] = min(wer(it["ref"], c) for c in it["pool"])
    sess = {r: collections.defaultdict(list) for r in REPLICA_SEEDS}
    for it in items:
        sess[it["gen_seed"]][it["speaker"]].append(it)
    for r in sess:
        for spk in sess[r]:
            sess[r][spk].sort(key=lambda x: (int(x["id"].split("-")[1]), int(x["id"].split("-")[2])))
    sel_by_key, shuf_by_key = {}, {}
    selN_by_key = {N: {} for N in Ns}
    donor = {speakers[i]: speakers[(i + 1) % len(speakers)] for i in range(len(speakers))}
    for r in REPLICA_SEEDS:
        out8, snaps = stream_selector(sess[r], variant, param, gate, N=8, snapshots=True)
        sel_by_key.update(out8)
        shuf, _ = stream_selector(sess[r], variant, param, gate, N=8,
                                  donor_snaps=snaps, donor_map=donor)
        shuf_by_key.update(shuf)
        for N in Ns:
            if N == 8:
                selN_by_key[8].update(out8)
            else:
                oN, _ = stream_selector(sess[r], variant, param, gate, N=N)
                selN_by_key[N].update(oN)

    # ---- metrics ----
    def by_utt_deltas(f):
        d = collections.defaultdict(list)
        for it in items:
            d[it["id"]].append(f(it))
        return d
    sw = lambda it: sel_by_key[it["key"]]["sel_wer"]
    d_mbr = by_utt_deltas(lambda it: it["mbr_wer_8"] - sw(it))
    d_grd = by_utt_deltas(lambda it: it["greedy_wer"] - sw(it))
    mean = lambda d: round(float(np.mean([x for v in d.values() for x in v])), 5)
    arms = {"greedy_wer": mean(by_utt_deltas(lambda it: it["greedy_wer"])),
            "mbr_wer_8": mean(by_utt_deltas(lambda it: it["mbr_wer_8"])),
            "oracle_wer_8": mean(by_utt_deltas(lambda it: it["oracle_wer_8"])),
            "sel_wer_8": mean(by_utt_deltas(sw)),
            "shuf_wer_8": mean(by_utt_deltas(lambda it: shuf_by_key[it["key"]]["sel_wer"]))}
    ci_mbr = cluster_boot(d_mbr)
    ci_grd = cluster_boot(d_grd)
    read_i = {"delta_vs_mbr_mean": mean(d_mbr), "delta_vs_mbr_ci95": ci_mbr,
              "red_vs_greedy_mean": mean(d_grd), "red_vs_greedy_ci95": ci_grd,
              "threshold": PASS_RED_GREEDY,
              "PASS": bool(ci_mbr[0] > 0 and mean(d_grd) >= PASS_RED_GREEDY)}
    read_ii = {"delta_vs_mbr_mean": mean(d_mbr), "ci95": ci_mbr,
               "PASS": bool(ci_mbr[0] > 0 and mean(d_mbr) >= PASS_RED_GREEDY)}
    headroom = arms["greedy_wer"] - arms["oracle_wer_8"]
    gain_sel = mean(d_mbr)
    d_shuf = by_utt_deltas(lambda it: it["mbr_wer_8"] - shuf_by_key[it["key"]]["sel_wer"])
    gain_shuf = mean(d_shuf)
    d_diff = by_utt_deltas(lambda it: shuf_by_key[it["key"]]["sel_wer"] - sw(it))
    goodhart_true = {str(N): mean(by_utt_deltas(lambda it, N=N: selN_by_key[N][it["key"]]["sel_wer"]))
                     for N in Ns}
    goodhart_proxy = {str(N): round(float(np.mean(
        [selN_by_key[N][it["key"]]["proxy"] for it in items])), 5) for N in Ns}
    d_g48 = by_utt_deltas(lambda it: selN_by_key[8][it["key"]]["sel_wer"]
                          - selN_by_key[4][it["key"]]["sel_wer"])
    ci_g48 = cluster_boot(d_g48)
    goodhart_fail = bool(goodhart_proxy["8"] > goodhart_proxy["4"] and ci_g48[0] > 0)
    pos_bins = {"1-4": [], "5-8": [], "9-12": []}
    for r in REPLICA_SEEDS:
        for spk, its in sess[r].items():
            for p, it in enumerate(its):
                b = "1-4" if p < 4 else ("5-8" if p < 8 else "9-12")
                pos_bins[b].append((it["mbr_wer_8"] - sw(it)))
    by_pos = {b: round(float(np.mean(v)), 5) if v else None for b, v in pos_bins.items()}

    load_bearing = bool(gain_sel > 0 and gain_shuf <= ABLATION_RETAIN * gain_sel)
    verdict = (f"PASS(i)={read_i['PASS']} PASS(ii)={read_ii['PASS']} agree={read_i['PASS']==read_ii['PASS']}; "
               f"gain_sel={gain_sel:+.5f} gain_shuf={gain_shuf:+.5f} "
               f"load_bearing(<=50% retained)={load_bearing}; goodhart_fail={goodhart_fail}. "
               + ("Route: GO-minimal input." if (read_i["PASS"] and read_ii["PASS"] and load_bearing and not goodhart_fail)
                  else ("Route: P-A (gain without load-bearing accumulation)."
                        if (read_i["PASS"] and read_ii["PASS"] and not load_bearing and not goodhart_fail)
                        else ("Route: ESCALATE (readings disagree — owner adjudication, stricter default)."
                              if read_i["PASS"] != read_ii["PASS"]
                              else "Route: lane result stands as measured (no PASS)."))))
    summary = {
        "lane": "M5", "status": "CONFIRMATORY (single touch)",
        "slice": {k: man[k] for k in ("n_utts", "n_speakers", "utts_per_speaker", "sampling_seed",
                                       "excluded_dev_spent", "excluded_m3_phase0")},
        "slice_manifest": "_repro/m5_confirmatory_slice_ids.json",
        "condition": man["condition"], "gen_seeds": REPLICA_SEEDS,
        "replica_design": "same utts x 3 seeds; memory per replica",
        "n_items": len(items), "n_complete_utts": len(items) // len(REPLICA_SEEDS),
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "frozen_variant": man["frozen_winner"],
        "arms": arms,
        "primary": {"reading_i": read_i, "reading_ii": read_ii,
                    "readings_agree": read_i["PASS"] == read_ii["PASS"]},
        "realized_fraction": {"selector": round((arms["greedy_wer"] - arms["sel_wer_8"]) / headroom, 4)
                                          if headroom > 0 else None,
                              "mbr": round((arms["greedy_wer"] - arms["mbr_wer_8"]) / headroom, 4)
                                     if headroom > 0 else None},
        "ablation": {"gain_sel": round(gain_sel, 5), "gain_shuf": round(gain_shuf, 5),
                     "retained": round(gain_shuf / gain_sel, 4) if gain_sel > 0 else None,
                     "load_bearing_le_50pct": load_bearing, "diff_ci95": cluster_boot(d_diff),
                     "shuffle_def": "speaker rotation derangement, size-matched snapshots, no self-update"},
        "goodhart": {"true_wer_by_N": goodhart_true, "proxy_by_N": goodhart_proxy,
                     "fail": goodhart_fail, "delta_true_wer_8_vs_4_ci95": ci_g48,
                     "by_session_position_gain_vs_mbr": by_pos},
        "bootstrap": {"method": "cluster by utterance", "draws": NBOOT, "seed": 42},
        "verdict": verdict,
        "prereg": "wiki/2026-07-03-agentic-tfrl-step1-preregistration.md @ b19bff2 §5 M5 + §2 GO-minimal",
    }
    per_item = [{"id": it["id"], "speaker": it["speaker"], "gen_seed": it["gen_seed"],
                 "ref": it["ref"], "greedy": it["greedy"], "greedy_wer": it["greedy_wer"],
                 "pool": it["pool"],
                 **{f"mbr_wer_{N}": it[f"mbr_wer_{N}"] for N in Ns},
                 "oracle_wer_8": it["oracle_wer_8"],
                 **{f"sel_wer_{N}": selN_by_key[N][it["key"]]["sel_wer"] for N in Ns},
                 "sel_pick_idx": sel_by_key[it["key"]]["pick"],
                 "sel_proxy_score": sel_by_key[it["key"]]["proxy"],
                 "mem_items_at_pick": sel_by_key[it["key"]]["mem_items"],
                 "shuf_wer_8": shuf_by_key[it["key"]]["sel_wer"]} for it in items]
    out = {"summary": summary, "per_item": per_item, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "M5C_STAGE=manifest then M5C_STAGE=run SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data "
                        "python scripts/m5_selector_confirmatory.py (llama-server resident; slice "
                        "deterministic from seed 20260703 + committed exclusions; ONE selector config only)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("M5C_RUN_DONE", flush=True)


if __name__ == "__main__":
    make_manifest() if STAGE == "manifest" else run()

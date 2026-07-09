"""M3 Phase-0 zero-support check (GPU; pre-registered kill rule, mechanically applied).

Frozen threshold (prereg §5 M3, freeze b19bff2): sample N=32 without context on utterances
containing recurring rare entities; KILL the M3 lane if the correct entity appears in > 1% of
samples. Binding aggregate = POOLED fraction over all samples (interpretation log #6); matching is
generous/kill-biased (#1: possessive-strip + plural tolerance); clean audio (#3); temp 0.8/top_p
0.95 = the operative q0 of the C1 operator (#4); 4 seed blocks x 8 = exactly 32 (#5, satisfies the
frozen N=32 AND the >=3-seeds floor). Plus one greedy transcript per utterance (free diagnostic)
and a cluster-bootstrap 95% CI on the pooled fraction (descriptive).

B0 axis statement (spec §B.0): M3 occupies axis (a) — changing q0. Phase-0 runs NO selector and
builds NO memory; it only measures whether the entity is already inside q0's sampling support.

Requires scripts/m3_phase0_select_entities.py output + resident llama-server (see
repro_asr_best_of_n_llamacpp.py header).
reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m3_phase0_zero_support.py
"""
import base64, collections, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/d/chao_workspace/exploring-l4-intelligence/common/src")
from repro_asr_best_of_n import INSTRUCTION  # the C1 prompt, verbatim
from speechrl_common.rl.reward import wer, _normalize

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
SEL = DATA / "_repro" / "m3_phase0_selection.json"
WAV_DIR = DATA / "_repro" / "m3_phase0_wavs"
OUT = DATA / "_repro" / "m3_phase0_zero_support.json"
SEED_BLOCKS = [42, 7, 123, 2026]
PER_BLOCK = 8
TEMP, TOP_P, MAXTOK = 0.8, 0.95, 100
KILL_F = 0.01
NBOOT = 10000


def norm_tokens(s: str) -> list[str]:
    toks = []
    for t in _normalize(s).upper().split():
        if t.endswith("'S"):
            t = t[:-2]
        toks.append(t.replace("'", ""))
    return [t for t in toks if t]


def matches(tok: str, ent: str) -> bool:
    return tok == ent or tok == ent + "S" or tok + "S" == ent


def contains(text: str, ent: str) -> bool:
    return any(matches(t, ent) for t in norm_tokens(text))


def gen(clip: str, greedy: bool, seed: int) -> str:
    b64 = base64.b64encode(open(clip, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": INSTRUCTION}]}],
        "max_tokens": MAXTOK, "seed": seed,
        "temperature": 0.0 if greedy else TEMP}
    if not greedy:
        payload["top_p"] = TOP_P
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def main():
    t0 = time.time()
    sel = json.load(open(SEL))
    picked = sel["selected"]
    n_per_utt = len(SEED_BLOCKS) * PER_BLOCK
    print(f"server={BASE} utts={len(picked)} samples/utt={n_per_utt} "
          f"blocks={SEED_BLOCKS}x{PER_BLOCK} temp={TEMP} kill_if_pooled>{KILL_F}", flush=True)

    rows, skipped = [], []
    for k, r in enumerate(picked):
        ent, clip = r["entity"], str(WAV_DIR / f"{r['id']}.wav")
        try:
            greedy = gen(clip, True, 42)
            samples, seeds = [], []
            for b in SEED_BLOCKS:
                for i in range(PER_BLOCK):
                    seeds.append(1000 * b + i)
                    samples.append(gen(clip, False, 1000 * b + i))
        except Exception as e:
            print(f"  [{k+1}] {r['id']} SKIP ({type(e).__name__}: {str(e)[:100]})", flush=True)
            skipped.append(r["id"]); continue
        n_match = sum(contains(s, ent) for s in samples)
        block_match = {str(b): sum(contains(samples[j], ent)
                                   for j in range(len(samples)) if seeds[j] // 1000 == b)
                       for b in SEED_BLOCKS}
        other_rare = [w for w in set(norm_tokens(r["ref"]))
                      if len(w) >= 4 and w != r["entity"]]
        rows.append({"id": r["id"], "speaker": r["id"].split("-")[0], "chapter": r["chapter"],
                     "ref": r["ref"], "entity": ent, "entity_train_freq": r["train_freq"],
                     "other_tokens_in_ref": other_rare[:20],
                     "greedy": greedy, "greedy_contains_entity": contains(greedy, ent),
                     "greedy_wer": wer(r["ref"], greedy),
                     "samples": samples, "sample_seeds": seeds,
                     "n_match": n_match, "match_fraction": round(n_match / n_per_utt, 4),
                     "block_match": block_match})
        print(f"  [{k+1}/{len(picked)}] {r['id']} entity={ent} match={n_match}/{n_per_utt} "
              f"greedy_hit={rows[-1]['greedy_contains_entity']}", flush=True)

    n_tot = len(rows) * n_per_utt
    n_hit = sum(r["n_match"] for r in rows)
    F = n_hit / max(n_tot, 1)
    # cluster bootstrap over utterances (descriptive)
    fr = np.array([r["n_match"] / n_per_utt for r in rows])
    rng = np.random.default_rng(42)
    bs = np.array([fr[rng.integers(0, len(fr), len(fr))].mean() for _ in range(NBOOT)])
    ci = [round(float(np.quantile(bs, 0.025)), 5), round(float(np.quantile(bs, 0.975)), 5)]
    per_entity = collections.defaultdict(lambda: {"n_utts": 0, "n_matches": 0})
    for r in rows:
        per_entity[r["entity"]]["n_utts"] += 1
        per_entity[r["entity"]]["n_matches"] += r["n_match"]
    for e, v in per_entity.items():
        v["fraction"] = round(v["n_matches"] / (v["n_utts"] * n_per_utt), 4)
    per_block = {str(b): round(sum(r["block_match"][str(b)] for r in rows)
                               / max(len(rows) * PER_BLOCK, 1), 5) for b in SEED_BLOCKS}
    kill = F > KILL_F
    summary = {
        "lane": "M3", "phase": 0, "axis": "a (changing q0)",
        "n_utts": len(rows), "n_skipped": len(skipped), "skipped_ids": skipped,
        "n_entities": len(per_entity), "n_chapters": len(set(r["chapter"] for r in rows)),
        "samples_per_utt": n_per_utt, "seed_blocks": SEED_BLOCKS, "samples_per_block": PER_BLOCK,
        "temp": TEMP, "top_p": TOP_P, "max_tokens": MAXTOK, "condition": "clean",
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "n_samples_total": n_tot, "n_matches_total": n_hit,
        "pooled_match_fraction": round(F, 5), "pooled_ci95_cluster_bootstrap": ci,
        "kill_threshold": KILL_F, "KILL": bool(kill),
        "per_seed_block_fraction": per_block,
        "per_entity": dict(per_entity),
        "utts_with_any_match": int(sum(1 for r in rows if r["n_match"] > 0)),
        "greedy_contains_entity_rate": round(float(np.mean(
            [r["greedy_contains_entity"] for r in rows])), 4),
        "greedy_mean_wer": round(float(np.mean([r["greedy_wer"] for r in rows])), 4),
        "verdict": (f"pooled entity-match fraction F={F:.5f} vs kill threshold {KILL_F}: "
                    + ("M3 LANE KILLED (support already present in q0)" if kill else
                       "M3 SURVIVES PHASE-0 (zero/near-zero support in q0 confirmed)")),
        "prereg": "wiki/2026-07-03-agentic-tfrl-step1-preregistration.md @ b19bff2 §5 M3 Phase-0",
        "selection_artifact": "_repro/m3_phase0_selection.json",
    }
    out = {"summary": summary, "per_utt": rows, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/m3_phase0_zero_support.py "
                        "(llama-server resident on :8091; selection deterministic — see m3_phase0_select_entities.py)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("M3_PHASE0_DONE", flush=True)


if __name__ == "__main__":
    main()

"""ASR best-of-N v2 — Gate B / G5.1 clean redo (seed-separated, corpus+macro WER, selector battery).

Ticket #32. The v1 artifacts (`scripts/repro_asr_best_of_n_llamacpp.py`,
`_repro/asr_bon_llamacpp_snr5.json`) are FROZEN EVIDENCE and are neither imported nor modified by this
script — this is a from-scratch, self-contained redo that satisfies the Gate B review demands v1 did
not:

  1. SEED FOUR-WAY SEPARATION (the central fix):
       - COHORT_SEED (env, default 20260711) selects the FIXED utterance cohort (permutation draw
         over librispeech test.<split>, n utterances) — identical across conditions.
       - NOISE_SEED (env, default 20260712) is a SEPARATE constant that drives the additive-noise
         realization per utterance (rng = NOISE_SEED*100003 + cohort_index). Kept apart from
         COHORT_SEED so "which utterances" and "what noise" vary independently and are each
         independently auditable/reproducible.
       - greedy decoding uses temperature=0.0 (argmax, deterministic); GREEDY_SEED is recorded for
         provenance only and has no effect on the output at temp=0.
       - POOL_SEEDS (env, default "1,2,3") — for the SAME cohort and SAME noise realization, THREE
         independent pools of POOL candidates are sampled (temperature=BON2_TEMP default 0.8,
         top_p=BON2_TOP_P default 0.95). Per-candidate seed = pool_seed*100003 + utt_idx*17 + i
         (utt_idx = 0-based cohort draw index, i = 0-based candidate index within the pool). This
         gives THREE independent generation replicates per utterance, feeding the hierarchical
         bootstrap (see below) instead of v1's single-pool-per-utterance design.

  2. CONDITIONS: `clean` and `snr5` (env BON2_CONDITION). SAME cohort (same COHORT_SEED draw), SAME
     code path (`load_cohort` + `materialize_wavs`) — additive white Gaussian noise at BON2_SNR dB
     (default 5) is the ONLY difference for snr5; clean skips the noise-injection step entirely. One
     artifact per condition: `_repro/asr_bon_v2_<condition>.json` under SPEECHRL_DATA_DIR.

  3. METRICS: every hypothesis (greedy + every pool candidate) gets a jiwer word-level alignment
     (S/D/I/H/N) via `word_alignment()`, version-robust to the actually-installed jiwer (see below).
     Aggregates report BOTH corpus WER (sum errors / sum ref words) AND macro utterance-WER (mean of
     per-utterance WER; for pool-seed-replicated selectors, averaged within-utterance across the 3
     pool seeds FIRST, then across utterances, so every utterance carries equal weight regardless of
     replicate count) — for greedy and for every selector at N in {1,2,4,8} (capped at BON2_POOL).

  4. SELECTOR BATTERY (applied post-hoc to the stored pools only, never re-queries the server):
       - oracle   : argmin WER vs reference. Reported and labeled ORACLE UPPER BOUND — this selector
                    is INFORMATION-BOUNDARY-VIOLATING BY DESIGN (uses the reference at selection time,
                    a label a real deployment never has) and is not a deployable candidate.
       - mbr      : Minimum-Bayes-Risk consensus, reference-free — argmin mean pairwise WORD-LEVEL
                    EDIT DISTANCE against the rest of the pool (uses `speechrl_common.rl.decode.mbr`
                    as the selection operator; utility(ci, cj) = -edit_distance(cj, ci)). Deliberately
                    NOT v1's `-wer(cj, ci)` formula — a WER ratio divides by the pseudo-reference's
                    word count, so a short/degenerate candidate standing in as pseudo-reference
                    artificially inflates every other candidate's score against it, biasing MBR toward
                    the degenerate candidate (confirmed empirically in this script's unit test); raw
                    edit distance has no such denominator and is direction-invariant. Deployable (no
                    labels at inference) — see `select_mbr_idx` docstring for the full derivation.
       - random   : seeded uniform pick over the N candidates. Deployable trivial baseline. Seed =
                    RANDOM_SELECT_SEED + utt_idx*1009 + pool_seed*131 + N*7 (documented, independent
                    of iteration order).
       - length   : longest candidate by NORMALIZED WORD COUNT (ties -> first/lowest index). Deployable.
       - logprob  : argmax mean generated-token logprob (llama-server /v1/chat/completions
                    `logprobs`/`top_logprobs` fields — see `gen()`). Included in the active battery
                    ONLY if every stored hypothesis in the run actually got a non-null value; otherwise
                    omitted and the gap is recorded under `provenance.api_gaps`.

  5. STATISTICS: hierarchical (two-level) bootstrap — resample utterances WITH replacement, then
     WITHIN each resampled utterance independently resample its pool-seed replicates WITH replacement
     — 2000 reps (BON2_NBOOT), vectorized in numpy. CIs are reported for the selector-vs-greedy delta
     (greedy WER - selector WER; positive = improvement) on BOTH corpus and macro WER, for every
     selector at every N.

  6. PROVENANCE + ATOMIC WRITE: the output JSON carries git SHA + dirty flag, the llama.cpp pinned
     commit (read live from the umbrella's scripts/env-setup.sh AND cross-checked against a live
     GET /props, both recorded), model file path(s)+size(s) (+ the docs/datasets.lock.json revision
     entry), the cohort manifest (ordered utterance IDs + a sha256 fingerprint), every seed, the
     sampling params actually sent, and env versions (python/jiwer/numpy). Written via
     tempfile-in-same-dir + os.replace (atomic rename), never a direct open(..., "w") on the final path.

  LIMITATIONS (recorded verbatim in the output under "limitations"): only ONE noise realization is
  drawn per utterance per condition (nested noise replicates — i.e. multiple independent noise draws
  per utterance, bootstrapped as a third hierarchy level — are deferred). The hierarchical bootstrap
  here resamples utterances and pool-generation seeds only, not noise realizations.

Requires the resident llama-server (see scripts/gpu_session.sh `serve qwen3-omni-30b-gguf up`).

reproduce (smoke — n=4, pool=2 i.e. N in {1,2}, ONE pool seed, snr5 only):
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data \
  BON2_CONDITION=snr5 BON2_N=4 BON2_POOL=2 POOL_SEEDS=1 BON2_NBOOT=200 \
    python -u scripts/repro_asr_best_of_n_v2.py

reproduce (full — both conditions, run separately):
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data BON2_CONDITION=clean BON2_N=96 \
    python -u scripts/repro_asr_best_of_n_v2.py
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data BON2_CONDITION=snr5  BON2_N=96 \
    python -u scripts/repro_asr_best_of_n_v2.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/d/chao_workspace/exploring-l4-intelligence/common/src")  # match v1 convention
from speechrl_common.rl.decode import mbr as mbr_op  # noqa: E402
from speechrl_common.rl.reward import _normalize  # noqa: E402

INSTRUCTION = "Transcribe the English speech into text. Output only the transcript, no explanation."

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
THIS_FILE = Path(__file__).resolve()
WORK_REPO_ROOT = THIS_FILE.parent.parent
UMBRELLA_ROOT = WORK_REPO_ROOT.parent.parent  # projects/<work> -> projects -> umbrella root

CONDITION = os.environ.get("BON2_CONDITION", "snr5")
assert CONDITION in ("clean", "snr5"), f"BON2_CONDITION must be 'clean' or 'snr5', got {CONDITION!r}"
SPLIT = os.environ.get("BON2_SPLIT", "other")
N_UTTS = int(os.environ.get("BON2_N", "96"))
POOL = int(os.environ.get("BON2_POOL", "8"))
POOL_SEEDS = [int(x) for x in os.environ.get("POOL_SEEDS", "1,2,3").split(",")]
TEMP = float(os.environ.get("BON2_TEMP", "0.8"))
TOP_P = float(os.environ.get("BON2_TOP_P", "0.95"))
TOP_K = 40
REPEAT_PENALTY = 1.0
MAXTOK = int(os.environ.get("BON2_MAXTOK", "100"))
SNR_DB = float(os.environ.get("BON2_SNR", "5"))
NBOOT = int(os.environ.get("BON2_NBOOT", "2000"))
COHORT_SEED = int(os.environ.get("COHORT_SEED", "20260711"))
NOISE_SEED = int(os.environ.get("NOISE_SEED", "20260712"))  # deliberately != COHORT_SEED
GREEDY_SEED = int(os.environ.get("BON2_GREEDY_SEED", "0"))  # temp=0 -> seed is a no-op; recorded only
RANDOM_SELECT_SEED = int(os.environ.get("BON2_RANDOM_SEED", "424242"))
BOOT_SEED = int(os.environ.get("BON2_BOOT_SEED", "42"))
Ns = [n for n in (1, 2, 4, 8) if n <= POOL]
SELECTOR_NAMES = ["oracle", "mbr", "random", "length", "logprob"]
# 2026-07-11 (hardening chain, ticket: GPU-hardening/second-noise-realization): allow overriding the
# output path via BON2_OUT so a second noise realization (or any other reproduction rerun) can never
# clobber a previously-verified artifact at the default path. Default is UNCHANGED (same hardcoded
# path every prior caller relied on) when BON2_OUT is unset.
OUT = Path(os.environ["BON2_OUT"]) if os.environ.get("BON2_OUT") else DATA / "_repro" / f"asr_bon_v2_{CONDITION}.json"
PARTIAL = OUT.with_suffix(".partial.jsonl")  # utterance-level checkpoint (see main); deleted on success
N_GEN_RETRIES = 0  # count of single-retry recoveries (see gen_with_retry); recorded in provenance
# Per-request HTTP timeouts (2026-07-11 stall fix): the first full run livelocked inside llama-server
# ("need to evaluate at least 1 token for each active slot (n_past = task.n_tokens())" repeating --
# the prompt-fully-cached edge that best-of-N's identical-audio-prefix-25x-per-utterance pattern
# triggers) and the client blocked indefinitely in poll() on its socket. requests' (connect, read)
# timeout pair turns such a hang into a Timeout exception -> gen_with_retry's single retry -> at
# worst the utterance is recorded skipped instead of stalling the whole run.
GEN_CONNECT_TIMEOUT_S = float(os.environ.get("BON2_CONNECT_TIMEOUT", "10"))
GEN_READ_TIMEOUT_S = float(os.environ.get("BON2_READ_TIMEOUT", "180"))
# Abort guards (never spin all night on a systemically-broken server): abort if the FIRST 3
# utterances ALL skip (nothing works at all), or if total skips exceed BON2_MAX_SKIP.
MAX_SKIP = int(os.environ.get("BON2_MAX_SKIP", "8"))


# ---------------------------------------------------------------------------------------------
# WER / alignment (jiwer-version-robust: prereg assumed jiwer==2.0 (`compute_measures`); the shared
# venv actually resolves jiwer==4.0.0 (verified 2026-07-11 in WSL: `importlib.metadata.version`),
# which replaced compute_measures with `process_words` (WordOutput objects with the same S/D/I/H
# counts under near-identical attribute names). Support both so the script runs unmodified either way.)
# ---------------------------------------------------------------------------------------------
def word_alignment(reference: str, hypothesis: str) -> dict:
    """Word-level S(ubstitutions)/D(eletions)/I(nsertions)/H(its)/N(ref words) + WER. Normalizes both
    strings with the SAME `_normalize` speechrl_common.rl.reward.wer() uses, so this WER matches the
    project-wide convention exactly (lowercase, strip punctuation, collapse whitespace)."""
    import jiwer

    ref_n, hyp_n = _normalize(reference), _normalize(hypothesis)
    if hasattr(jiwer, "process_words"):  # jiwer >= 3.x
        out = jiwer.process_words(ref_n, hyp_n)
        S, D, I, H = out.substitutions, out.deletions, out.insertions, out.hits
        wer_val = float(out.wer)
    else:  # jiwer == 2.x fallback
        m = jiwer.compute_measures(ref_n, hyp_n)
        S, D, I, H = m["substitutions"], m["deletions"], m["insertions"], m["hits"]
        wer_val = float(m["wer"])
    N = H + S + D
    return {"S": int(S), "D": int(D), "I": int(I), "H": int(H), "N": int(N), "wer": wer_val}


def jiwer_api_info() -> dict:
    import importlib.metadata as im
    import jiwer

    return {"version": im.version("jiwer"),
            "api": "process_words" if hasattr(jiwer, "process_words") else "compute_measures"}


# ---------------------------------------------------------------------------------------------
# Cohort + noise (seed-separated: COHORT_SEED picks utterances, NOISE_SEED picks the noise draw)
# ---------------------------------------------------------------------------------------------
def _add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_p = float(np.mean(wav ** 2)) + 1e-12
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    noise_p = float(np.mean(noise ** 2)) + 1e-12
    noise *= np.sqrt((sig_p / (10 ** (snr_db / 10.0))) / noise_p)
    return (wav + noise).astype(np.float32)


def load_cohort(data_dir: Path, split: str, n: int, cohort_seed: int) -> list[dict]:
    """Fixed utterance cohort (librispeech test.<split>), selected by COHORT_SEED alone — the same
    cohort is reused for BOTH the clean and snr5 conditions (only the noise step differs)."""
    import pyarrow.parquet as pq

    pq_path = data_dir / f"datasets/librispeech/all/test.{split}/0000.parquet"
    t = pq.read_table(str(pq_path), columns=["audio", "text", "id"])
    audio, text, ids = t.column("audio").to_pylist(), t.column("text").to_pylist(), t.column("id").to_pylist()
    idx = np.random.default_rng(cohort_seed).permutation(len(text))[:n]
    return [{"j": j, "row": int(i), "id": ids[int(i)], "ref": text[int(i)], "audio_bytes": audio[int(i)]["bytes"]}
            for j, i in enumerate(idx)]


def materialize_wavs(cohort: list[dict], data_dir: Path, split: str, condition: str,
                     snr_db: float, noise_seed: int) -> list[str]:
    """Write one wav per cohort utterance. clean = untouched audio; snr5 = additive noise via a
    NOISE_SEED-derived rng (independent of COHORT_SEED)."""
    import soundfile as sf

    # 2026-07-11 (hardening chain, second-noise-realization fix): the wav cache was keyed only by
    # utterance id, so a rerun with a DIFFERENT NOISE_SEED would silently hit the `wp.exists()`
    # short-circuit below and REUSE the previous realization's cached noisy wavs -- i.e. not a new
    # noise draw at all. Key the snr5 cache dir by the noise seed so each noise realization
    # materializes its own wavs. `clean` is noise-free and keeps its original directory name.
    # The original (noise1, NOISE_SEED=20260712) run predates this suffix; its wavs remain in the
    # unsuffixed `..._snr5_wavs` directory and are bit-identically regenerable from its seed.
    ns_suffix = f"_ns{noise_seed}" if condition == "snr5" else ""
    wav_dir = data_dir / "_repro" / f"asr_bon_v2_{split}_{condition}{ns_suffix}_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for u in cohort:
        wp = wav_dir / f"{u['id']}.wav"
        if not wp.exists():
            wav, sr = sf.read(io.BytesIO(u["audio_bytes"]), dtype="float32")
            if condition == "snr5":
                rng = np.random.default_rng(noise_seed * 100003 + u["j"])
                wav = _add_noise(wav, snr_db, rng)
            sf.write(str(wp), wav, sr)
        paths.append(str(wp))
    return paths


# ---------------------------------------------------------------------------------------------
# llama-server generation (OpenAI-compatible /v1/chat/completions, input_audio content)
# ---------------------------------------------------------------------------------------------
def gen(wav_path: str, seed: int, temp: float, top_p: float, maxtok: int) -> tuple[str, float | None]:
    import requests  # lazy; carries the (connect, read) timeout pair urllib's single timeout lacked

    b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
                   {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                   {"type": "text", "text": INSTRUCTION}]}],
               "max_tokens": maxtok, "seed": seed, "temperature": temp,
               "top_p": top_p, "top_k": TOP_K, "repeat_penalty": REPEAT_PENALTY,
               "logprobs": True, "top_logprobs": 1}
    r = requests.post(BASE + "/v1/chat/completions", json=payload,
                      timeout=(GEN_CONNECT_TIMEOUT_S, GEN_READ_TIMEOUT_S))
    r.raise_for_status()
    ch = r.json()["choices"][0]
    txt = ch["message"]["content"].strip()
    mean_lp = None
    lp = ch.get("logprobs")
    if lp and lp.get("content"):
        toks = [c["logprob"] for c in lp["content"] if c.get("logprob") is not None]
        if toks:
            mean_lp = float(np.mean(toks))
    return txt, mean_lp


def gen_with_retry(wav_path: str, seed: int, temp: float, top_p: float, maxtok: int) -> tuple[str, float | None]:
    """One retry per failed generation request (full-run resilience directive): pause briefly and
    re-issue the SAME request (same seed/params) once; if the retry also fails, the exception
    propagates and the caller records the whole utterance as skipped — results are never fabricated.
    Successful retries are counted in N_GEN_RETRIES and recorded in provenance."""
    global N_GEN_RETRIES
    try:
        return gen(wav_path, seed, temp, top_p, maxtok)
    except Exception as e:
        print(f"    gen retry 1/1 (seed={seed}) after {type(e).__name__}: {str(e)[:100]}", flush=True)
        time.sleep(5)
        result = gen(wav_path, seed, temp, top_p, maxtok)
        N_GEN_RETRIES += 1
        return result


def query_props() -> dict:
    try:
        req = urllib.request.Request(BASE + "/props")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


# ---------------------------------------------------------------------------------------------
# Selector battery (pure post-hoc selection over stored pools; no server calls)
# ---------------------------------------------------------------------------------------------
def select_oracle_idx(cands: list[dict]) -> int:
    """ORACLE UPPER BOUND: argmin WER vs reference. Information-boundary-violating by design (uses
    the reference at selection time) — reported as a headroom ceiling, never a deployable selector."""
    return int(np.argmin([c["wer"] for c in cands]))


def edit_distance(a: str, b: str) -> int:
    """Word-level Levenshtein distance (S+D+I), direction-invariant (verified: swapping which string
    is passed as jiwer's "reference" swaps insertions<->deletions but leaves the S+D+I TOTAL
    unchanged — the WER *ratio* is asymmetric because its denominator N is the reference's word
    count, but the raw edit distance is not)."""
    al = word_alignment(a, b)
    return al["S"] + al["D"] + al["I"]


def select_mbr_idx(cands: list[dict]) -> int:
    """Deployable, reference-free: MBR consensus by pairwise WORD-LEVEL EDIT DISTANCE among the pool
    members (picks the candidate minimizing mean edit distance to the rest of the pool).

    Uses raw edit distance, NOT a WER ratio (v1's `-wer(cj, ci)` formula): a WER ratio divides by
    N = len(reference), so when a short/degenerate candidate stands in as the pseudo-reference, every
    other (longer, likely-correct) candidate's WER against it is inflated by the small denominator,
    biasing MBR toward picking the degenerate candidate itself. Confirmed empirically (unit test,
    2026-07-11) on a synthetic 3-candidate pool containing one much-shorter outlier: the WER-ratio
    formula picked the outlier; edit distance (symmetric, no such denominator) picked the correct
    consensus candidate instead. This matches the ticket's own definition: "MBR (pairwise edit
    distance)"."""
    texts = [c["hyp"] for c in cands]
    res = mbr_op(texts, lambda ci, cj: -edit_distance(cj, ci))
    return int(res["index"])


def select_random_idx(n_cands: int, utt_idx: int, pool_seed: int, N: int) -> int:
    """Deployable trivial baseline. Seed documented + independent of iteration order."""
    seed = RANDOM_SELECT_SEED + utt_idx * 1009 + pool_seed * 131 + N * 7
    return int(np.random.default_rng(seed).integers(0, n_cands))


def select_length_idx(cands: list[dict]) -> int:
    """Deployable: longest candidate by normalized WORD count (ties -> first/lowest index)."""
    lens = [len(_normalize(c["hyp"]).split()) for c in cands]
    return int(np.argmax(lens))


def select_logprob_idx(cands: list[dict]):
    """Deployable IFF the server actually returned logprobs for every candidate; else None."""
    if any(c.get("logprob_mean") is None for c in cands):
        return None
    return int(np.argmax([c["logprob_mean"] for c in cands]))


def select_all(cands: list[dict], utt_idx: int, pool_seed: int, N: int) -> dict:
    out = {
        "oracle": select_oracle_idx(cands),
        "mbr": select_mbr_idx(cands),
        "random": select_random_idx(len(cands), utt_idx, pool_seed, N),
        "length": select_length_idx(cands),
    }
    out["logprob"] = select_logprob_idx(cands)
    return out


# ---------------------------------------------------------------------------------------------
# Hierarchical bootstrap: resample utterances, THEN within each resampled utterance resample its
# pool-seed replicates (both with replacement). Vectorized over numpy (n_utt x n_pool matrices).
# ---------------------------------------------------------------------------------------------
def hierarchical_boot(wer_mat, S_mat, D_mat, I_mat, Nref_mat, g_wer, g_S, g_D, g_I, g_N,
                      nboot: int, seed: int) -> dict:
    n_utt, n_pool = wer_mat.shape
    rng = np.random.default_rng(seed)
    macro_deltas = np.empty(nboot)
    corpus_deltas = np.empty(nboot)
    for b in range(nboot):
        utt_idx = rng.integers(0, n_utt, n_utt)
        pool_idx = rng.integers(0, n_pool, (n_utt, n_pool))
        sel_wer = wer_mat[utt_idx[:, None], pool_idx]
        macro_sel = float(sel_wer.mean(axis=1).mean())
        macro_grd = float(g_wer[utt_idx].mean())
        macro_deltas[b] = macro_grd - macro_sel
        sel_err = float((S_mat[utt_idx[:, None], pool_idx] + D_mat[utt_idx[:, None], pool_idx]
                        + I_mat[utt_idx[:, None], pool_idx]).sum())
        sel_n = float(Nref_mat[utt_idx[:, None], pool_idx].sum())
        # greedy has no pool-seed dimension; replicate its (single) value n_pool times per resampled
        # utterance occurrence so both corpus ratios are on a comparable (same-count) scale.
        grd_err = float((g_S[utt_idx] + g_D[utt_idx] + g_I[utt_idx]).sum()) * n_pool
        grd_n = float(g_N[utt_idx].sum()) * n_pool
        corpus_deltas[b] = (grd_err / grd_n) - (sel_err / sel_n)

    def ci(arr):
        lo, hi = np.quantile(arr, [0.025, 0.975])
        return [round(float(lo), 5), round(float(hi), 5)]

    return {"macro_delta_mean": round(float(np.mean(macro_deltas)), 5), "macro_delta_ci95": ci(macro_deltas),
            "macro_sig": bool(ci(macro_deltas)[0] > 0),
            "corpus_delta_mean": round(float(np.mean(corpus_deltas)), 5), "corpus_delta_ci95": ci(corpus_deltas),
            "corpus_sig": bool(ci(corpus_deltas)[0] > 0)}


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------
def git_info(repo_dir: Path) -> dict:
    try:
        sha = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except Exception as e:
        return {"sha": None, "dirty": None, "error": f"{type(e).__name__}: {str(e)[:150]}"}


def llamacpp_pinned_commit() -> dict:
    """The llama.cpp commit env-setup.sh checks out (umbrella scripts/env-setup.sh:LLAMACPP_COMMIT),
    read LIVE from that file (not hardcoded) so this stays correct if the pin is bumped later."""
    import re

    p = UMBRELLA_ROOT / "scripts" / "env-setup.sh"
    try:
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("LLAMACPP_COMMIT="):
                m = re.search(r"[0-9a-f]{40}", line)
                return {"source": str(p), "commit": m.group(0) if m else None, "raw_line": line.strip()}
        return {"source": str(p), "commit": None, "note": "LLAMACPP_COMMIT line not found"}
    except Exception as e:
        return {"source": str(p), "commit": None, "error": f"{type(e).__name__}: {str(e)[:150]}"}


def model_provenance() -> dict:
    gguf_dir = DATA / "models" / "qwen3-omni-30b-a3b-instruct-gguf"
    files = {}
    for f in gguf_dir.glob("*.gguf"):
        files[f.name] = {"path": str(f), "size_bytes": f.stat().st_size}
    lock_path = UMBRELLA_ROOT / "docs" / "datasets.lock.json"
    # datasets.lock.json's top-level structure is not a flat list at the root in all schema
    # versions seen in this repo; search recursively for the matching "name" instead of assuming
    # a fixed key path.
    def _find(obj):
        if isinstance(obj, dict):
            if obj.get("name") == "qwen3-omni-30b-a3b-instruct-gguf":
                return obj
            for v in obj.values():
                r = _find(v)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for it in obj:
                r = _find(it)
                if r is not None:
                    return r
        return None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_entry = _find(lock)
    except Exception as e:
        lock_entry = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    return {"gguf_dir": str(gguf_dir), "files": files, "datasets_lock_entry": lock_entry}


def cohort_manifest(cohort: list[dict]) -> dict:
    ids_ordered = [u["id"] for u in cohort]
    digest = hashlib.sha256(json.dumps(ids_ordered, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"split": SPLIT, "n": len(cohort), "cohort_seed": COHORT_SEED,
            "ids_ordered": ids_ordered, "ids_sha256": digest,
            "hash_method": "sha256 of json.dumps(ids_ordered, ensure_ascii=False), UTF-8"}


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic rename, same filesystem (tmp lives in path's own dir)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print(f"server={BASE} condition={CONDITION} split=test.{SPLIT} n={N_UTTS} pool={POOL} "
          f"pool_seeds={POOL_SEEDS} temp={TEMP} top_p={TOP_P} cohort_seed={COHORT_SEED} "
          f"noise_seed={NOISE_SEED} Ns={Ns} nboot={NBOOT}", flush=True)

    cohort = load_cohort(DATA, SPLIT, N_UTTS, COHORT_SEED)
    wav_paths = materialize_wavs(cohort, DATA, SPLIT, CONDITION, SNR_DB, NOISE_SEED)

    # ---- utterance-level checkpoint (2026-07-11 stall fix): every completed utterance's row is
    # appended to PARTIAL (<artifact>.partial.jsonl, first line = config-fingerprint header); on
    # startup, rows already present under a MATCHING fingerprint are reused without re-generation,
    # so a stall/crash only ever costs the un-checkpointed tail. Skipped utterances are NOT
    # checkpointed (a resume retries them). PARTIAL is deleted after the final atomic write. ----
    fingerprint = {"condition": CONDITION, "split": SPLIT, "n_utts": N_UTTS, "pool": POOL,
                   "pool_seeds": POOL_SEEDS, "temp": TEMP, "top_p": TOP_P, "top_k": TOP_K,
                   "repeat_penalty": REPEAT_PENALTY, "max_tokens": MAXTOK,
                   "cohort_seed": COHORT_SEED, "noise_seed": NOISE_SEED,
                   "snr_db": (SNR_DB if CONDITION == "snr5" else None),
                   "ids_sha256": cohort_manifest(cohort)["ids_sha256"]}
    ckpt_rows: dict[str, dict] = {}
    if PARTIAL.exists():
        try:
            with open(PARTIAL, encoding="utf-8") as f:
                header = json.loads(f.readline())
                if header.get("_header") == fingerprint:
                    for line in f:
                        line = line.strip()
                        if line:
                            row = json.loads(line)
                            ckpt_rows[row["id"]] = row
                    print(f"checkpoint: resuming {len(ckpt_rows)} utterances from {PARTIAL}", flush=True)
                else:
                    print("checkpoint: header/config mismatch — ignoring stale partial file", flush=True)
        except Exception as e:
            print(f"checkpoint: unreadable ({type(e).__name__}: {str(e)[:80]}) — starting fresh", flush=True)
            ckpt_rows = {}
    if not ckpt_rows:
        PARTIAL.parent.mkdir(parents=True, exist_ok=True)
        with open(PARTIAL, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_header": fingerprint}, ensure_ascii=False) + "\n")

    rows, skipped = [], []
    for j, (u, wav) in enumerate(zip(cohort, wav_paths)):
        if u["id"] in ckpt_rows:
            rows.append(ckpt_rows[u["id"]])
            if (j + 1) % 10 == 0 or (j + 1) == len(cohort):
                print(f"  [{j+1}/{len(cohort)}] (checkpointed) ({time.time()-t0:.0f}s, "
                      f"{len(rows)} ok / {len(skipped)} skipped)", flush=True)
            continue
        try:
            greedy_txt, greedy_lp = gen_with_retry(wav, GREEDY_SEED, 0.0, TOP_P, MAXTOK)
            g_align = word_alignment(u["ref"], greedy_txt)
            pools = {}
            for ps in POOL_SEEDS:
                cands = []
                for i in range(POOL):
                    seed = ps * 100003 + j * 17 + i
                    txt, lp = gen_with_retry(wav, seed, TEMP, TOP_P, MAXTOK)
                    al = word_alignment(u["ref"], txt)
                    cands.append({"hyp": txt, "logprob_mean": lp, **al})
                pools[str(ps)] = cands
        except Exception as e:
            print(f"  [{j+1}/{len(cohort)}] {u['id']} SKIP ({type(e).__name__}: {str(e)[:120]})", flush=True)
            skipped.append(u["id"])
            # systemic-failure guards: nothing works at all, or too many total skips — abort loudly
            # (partial checkpoint is preserved; a resume retries the skipped utterances).
            if len(rows) == 0 and len(skipped) >= 3:
                raise RuntimeError(f"first {len(skipped)} utterances ALL failed — server looks "
                                   f"systemically broken, aborting (checkpoint kept at {PARTIAL})")
            if len(skipped) > MAX_SKIP:
                raise RuntimeError(f"{len(skipped)} skipped utterances exceeds MAX_SKIP={MAX_SKIP} — "
                                   f"aborting (checkpoint kept at {PARTIAL})")
            continue

        selection = {}
        for N in Ns:
            selection[str(N)] = {}
            for ps in POOL_SEEDS:
                cands_N = pools[str(ps)][:N]
                picks = select_all(cands_N, j, ps, N)
                selection[str(N)][str(ps)] = {
                    sel: ({"idx": idx, "wer": cands_N[idx]["wer"]} if idx is not None else None)
                    for sel, idx in picks.items()
                }

        row = {"id": u["id"], "ref": u["ref"],
               "greedy": {"hyp": greedy_txt, "logprob_mean": greedy_lp, **g_align},
               "pools": pools, "selection": selection}
        with open(PARTIAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        if (j + 1) % 10 == 0 or (j + 1) == len(cohort):
            print(f"  [{j+1}/{len(cohort)}] greedy_wer={g_align['wer']:.3f} "
                  f"({time.time()-t0:.0f}s, {len(rows)} ok / {len(skipped)} skipped, "
                  f"{N_GEN_RETRIES} retries)", flush=True)

    if not rows:
        raise RuntimeError("no utterances survived generation — aborting before writing an empty artifact")

    # ---- logprob availability across the WHOLE run ----
    logprob_ok = all(c["logprob_mean"] is not None for r in rows for cands in r["pools"].values() for c in cands) \
        and all(r["greedy"]["logprob_mean"] is not None for r in rows)
    active_selectors = [s for s in SELECTOR_NAMES if s != "logprob" or logprob_ok]

    # ---- aggregation ----
    n_utt = len(rows)
    n_pool = len(POOL_SEEDS)
    g_wer = np.array([r["greedy"]["wer"] for r in rows], dtype=float)
    g_S = np.array([r["greedy"]["S"] for r in rows], dtype=float)
    g_D = np.array([r["greedy"]["D"] for r in rows], dtype=float)
    g_I = np.array([r["greedy"]["I"] for r in rows], dtype=float)
    g_N = np.array([r["greedy"]["N"] for r in rows], dtype=float)
    greedy_agg = {"corpus_wer": round(float((g_S.sum() + g_D.sum() + g_I.sum()) / g_N.sum()), 5),
                  "macro_wer": round(float(g_wer.mean()), 5)}

    selector_agg = {}
    for N in Ns:
        selector_agg[str(N)] = {}
        for sel in active_selectors:
            wer_mat = np.zeros((n_utt, n_pool))
            S_mat = np.zeros((n_utt, n_pool))
            D_mat = np.zeros((n_utt, n_pool))
            I_mat = np.zeros((n_utt, n_pool))
            Nref_mat = np.zeros((n_utt, n_pool))
            for ui in range(n_utt):
                for k, ps in enumerate(POOL_SEEDS):
                    pick = rows[ui]["selection"][str(N)][str(ps)][sel]
                    idx = pick["idx"]
                    c = rows[ui]["pools"][str(ps)][idx]
                    wer_mat[ui, k] = c["wer"]
                    S_mat[ui, k] = c["S"]
                    D_mat[ui, k] = c["D"]
                    I_mat[ui, k] = c["I"]
                    Nref_mat[ui, k] = c["N"]
            corpus_wer = float((S_mat.sum() + D_mat.sum() + I_mat.sum()) / Nref_mat.sum())
            macro_wer = float(wer_mat.mean(axis=1).mean())
            boot = hierarchical_boot(wer_mat, S_mat, D_mat, I_mat, Nref_mat, g_wer, g_S, g_D, g_I, g_N,
                                     NBOOT, BOOT_SEED)
            label = ("ORACLE UPPER BOUND (information-boundary-violating by design: selection uses "
                     "the reference; NOT deployable)" if sel == "oracle" else "DEPLOYABLE")
            selector_agg[str(N)][sel] = {
                "label": label,
                "corpus_wer": round(corpus_wer, 5), "macro_wer": round(macro_wer, 5),
                "corpus_delta_vs_greedy_mean": boot["corpus_delta_mean"],
                "corpus_delta_vs_greedy_ci95": boot["corpus_delta_ci95"],
                "corpus_sig": boot["corpus_sig"],
                "macro_delta_vs_greedy_mean": boot["macro_delta_mean"],
                "macro_delta_vs_greedy_ci95": boot["macro_delta_ci95"],
                "macro_sig": boot["macro_sig"],
            }

    api_gaps = {"logprobs_available": logprob_ok,
                "note": ("llama-server returned non-null logprobs/top_logprobs for every stored "
                         "hypothesis; logprob-confidence selector is ACTIVE." if logprob_ok else
                         "llama-server did NOT return usable logprobs for at least one hypothesis "
                         "(input_audio + logprobs=True/top_logprobs=1 request); logprob-confidence "
                         "selector is EXCLUDED from the battery for this artifact.")}

    provenance = {
        "git": git_info(WORK_REPO_ROOT),
        "llamacpp_pinned_commit": llamacpp_pinned_commit(),
        "llama_server_props": query_props(),
        "model": model_provenance(),
        "cohort": cohort_manifest(cohort),
        "condition": CONDITION, "snr_db": (SNR_DB if CONDITION == "snr5" else None),
        "noise_seed": NOISE_SEED,
        "pool_seeds": POOL_SEEDS,
        "pool_seed_formula": "seed = pool_seed*100003 + utt_idx*17 + i  (utt_idx=cohort draw index 0-based, i=candidate index 0-based)",
        "greedy_seed": GREEDY_SEED,
        "greedy_seed_note": "temperature=0.0 (argmax decode) -> seed has no effect on output; recorded for completeness only",
        "random_selector_seed_base": RANDOM_SELECT_SEED,
        "random_selector_seed_formula": "seed = RANDOM_SELECT_SEED + utt_idx*1009 + pool_seed*131 + N*7",
        "sampling_params": {"temperature_pool": TEMP, "top_p": TOP_P, "top_k": TOP_K,
                            "repeat_penalty": REPEAT_PENALTY, "max_tokens": MAXTOK,
                            "temperature_greedy": 0.0},
        "env_versions": {"python": sys.version, "numpy": np.__version__, "jiwer": jiwer_api_info()},
        "nboot": NBOOT, "boot_seed": BOOT_SEED,
        "n_generation_retries": N_GEN_RETRIES,
        "n_checkpoint_resumed": len(ckpt_rows),
        "gen_timeouts": {"connect_s": GEN_CONNECT_TIMEOUT_S, "read_s": GEN_READ_TIMEOUT_S,
                         "note": "per-request (connect, read) timeouts added 2026-07-11 after a "
                                 "llama-server prompt-cache livelock hung the first full run; a "
                                 "timed-out request gets ONE retry then the utterance is skipped"},
        "api_gaps": api_gaps,
    }

    limitations = {
        "single_noise_realization_per_condition": (
            "Exactly one additive-noise draw per utterance (NOISE_SEED*100003+utt_idx) is used for "
            "the snr5 condition; nested noise replicates (multiple independent noise draws per "
            "utterance, forming a third hierarchy level for the bootstrap) are DEFERRED. The "
            "hierarchical bootstrap in this artifact resamples utterances and pool-generation seeds "
            "only, not noise realizations."),
    }

    Ns_str = ", ".join(str(n) for n in Ns)
    verdict = (f"condition={CONDITION} n={n_utt} (skipped {len(skipped)}); greedy corpus WER "
              f"{greedy_agg['corpus_wer']:.3f} / macro WER {greedy_agg['macro_wer']:.3f}. "
              f"Selectors active: {active_selectors} at N in {{{Ns_str}}}. See summary.selectors "
              f"for per-N per-selector corpus/macro WER + hierarchical-bootstrap deltas vs greedy "
              f"(2000-rep, utterance-then-pool-seed resampling). Oracle is an information-boundary-"
              f"violating UPPER BOUND, not a deployable result.")

    summary = {
        "problem": "ASR best-of-N v2 (Gate B / G5.1 clean redo) — seed-separated, corpus+macro WER, deployable selector battery",
        "condition": CONDITION, "split": f"test.{SPLIT}", "n_utts": n_utt, "n_skipped": len(skipped),
        "skipped_ids": skipped,
        "pool": POOL, "pool_seeds": POOL_SEEDS, "Ns": Ns,
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL upstream)",
        "greedy": greedy_agg,
        "selectors": selector_agg,
        "active_selectors": active_selectors,
        "bootstrap": {"method": "hierarchical (resample utterances, then within-utterance resample "
                                "pool seeds), both with replacement", "reps": NBOOT, "seed": BOOT_SEED},
        "limitations": limitations,
        "provenance": provenance,
        "verdict": verdict,
    }

    out = {"summary": summary, "per_utt": rows, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": (f"SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data "
                        f"BON2_CONDITION={CONDITION} BON2_N={N_UTTS} BON2_POOL={POOL} "
                        f"POOL_SEEDS={','.join(str(x) for x in POOL_SEEDS)} COHORT_SEED={COHORT_SEED} "
                        f"NOISE_SEED={NOISE_SEED} python -u scripts/repro_asr_best_of_n_v2.py "
                        f"(llama-server resident on {BASE})")}
    atomic_write_json(OUT, out)
    PARTIAL.unlink(missing_ok=True)  # final artifact written atomically; checkpoint no longer needed
    print("\n=== SUMMARY ===\n" + json.dumps({k: v for k, v in summary.items() if k not in ("provenance",)},
                                             indent=2, ensure_ascii=False), flush=True)
    print("wrote", OUT, flush=True)
    print("ASR_BON_V2_DONE", flush=True)


if __name__ == "__main__":
    main()

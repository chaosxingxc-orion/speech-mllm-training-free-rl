"""Operator-B token-level best-of-N ASR validation on LibriSpeech (latest omni model, GPU).

Demonstrates the generative training-free-RL leg: sample N candidate transcripts from a *frozen*
omni model and select with a verifiable reward. Two selectors:
  - oracle best-of-N (reward = -WER vs reference): the verifiable-reward argmax (β->0 tilting, T1);
    its N-scaling is the empirical check of the best-of-N headroom / O(sqrt log N) shape (T2/T6).
  - MBR (reference-free, utility = -WER between candidates): the *deployable* consensus selector (T5).

Reports greedy baseline WER, per-N WER curves, and paired bootstrap CIs on the per-utterance WER
reduction. Validation-only (tiny n); larger locked runs deferred.

reproduce:
  SPEECHRL_DATA_DIR=/path/to/speechrl-data \
  python scripts/repro_asr_best_of_n.py --split other --n-utts 48 --pool 8 --temperature 0.7
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

INSTRUCTION = "Transcribe the English speech into text. Output only the transcript, no explanation."


def _add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at a target SNR (dB) — a standard ASR-robustness perturbation
    that gives a strong model measurable WER, hence best-of-N headroom."""
    sig_p = float(np.mean(wav ** 2)) + 1e-12
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    noise_p = float(np.mean(noise ** 2)) + 1e-12
    target = sig_p / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(target / noise_p)
    return (wav + noise).astype(np.float32)


def load_utts(data_dir: str, split: str, n_utts: int, seed: int, snr_db: float | None) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf

    pq_path = Path(data_dir) / f"datasets/librispeech/all/test.{split}/0000.parquet"
    t = pq.read_table(str(pq_path), columns=["audio", "text", "id"])
    audio = t.column("audio").to_pylist()
    text = t.column("text").to_pylist()
    ids = t.column("id").to_pylist()
    idx = np.random.default_rng(seed).permutation(len(text))[:n_utts]
    tag = "clean" if snr_db is None else f"snr{int(snr_db)}"
    wav_dir = Path(data_dir) / "_repro" / f"librispeech_{split}_{tag}_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    utts = []
    for j, i in enumerate(idx):
        wav, sr = sf.read(io.BytesIO(audio[int(i)]["bytes"]), dtype="float32")
        if snr_db is not None:
            wav = _add_noise(wav, snr_db, np.random.default_rng(seed * 100003 + j))
        wp = wav_dir / f"{ids[int(i)]}.wav"
        sf.write(str(wp), wav, sr)
        utts.append({"id": ids[int(i)], "audio_path": str(wp), "ref": text[int(i)]})
    return utts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="other", choices=["clean", "other"])
    ap.add_argument("--n-utts", type=int, default=48)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--snr", type=float, default=None, help="additive-noise SNR in dB; omit for clean")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "GPU required"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    from speechrl_common.models.generative_omni import generate_candidates, load_generative_omni
    from speechrl_common.rl.decode import best_of_n, mbr
    from speechrl_common.rl.reward import wer

    data = os.environ["SPEECHRL_DATA_DIR"]
    model_path = os.path.join(data, "models/qwen3-omni-30b-a3b-instruct")
    out = Path(data) / "_repro"
    Ns = [n for n in (1, 2, 4, 8) if n <= args.pool]

    utts = load_utts(data, args.split, args.n_utts, args.seed, args.snr)
    tag = "clean" if args.snr is None else f"snr{int(args.snr)}"
    print(f"split=test.{args.split} cond={tag} utts={len(utts)} pool={args.pool} temp={args.temperature}", flush=True)
    model, proc = load_generative_omni(model_path, device="cuda")

    rows = []
    for k, u in enumerate(utts):
        greedy = generate_candidates(model, proc, u["audio_path"], INSTRUCTION, greedy=True,
                                     device="cuda", max_new_tokens=args.max_new_tokens)[0]
        pool = generate_candidates(model, proc, u["audio_path"], INSTRUCTION, n=args.pool,
                                   temperature=args.temperature, seed=args.seed, device="cuda",
                                   max_new_tokens=args.max_new_tokens)
        rec = {"id": u["id"], "ref": u["ref"], "greedy": greedy, "greedy_wer": wer(u["ref"], greedy), "pool": pool}
        for N in Ns:
            cand = pool[:N]
            bo = best_of_n(cand, lambda c: -wer(u["ref"], c))            # oracle (verifiable WER reward)
            mb = mbr(cand, lambda ci, cj: -wer(cj, ci))                   # reference-free consensus
            rec[f"oracle_wer_{N}"] = wer(u["ref"], bo["best"])
            rec[f"mbr_wer_{N}"] = wer(u["ref"], mb["best"])
        rows.append(rec)
        print(f"[{k+1}/{len(utts)}] greedy={rec['greedy_wer']:.3f} "
              f"oracle@{Ns[-1]}={rec[f'oracle_wer_{Ns[-1]}']:.3f} mbr@{Ns[-1]}={rec[f'mbr_wer_{Ns[-1]}']:.3f}", flush=True)

    (out / f"asr_bon_{args.split}_{tag}_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    def paired_ci(akey, bkey, n=1000):
        da = np.array([r[akey] - r[bkey] for r in rows], dtype=float)  # WER reduction if > 0
        rng = np.random.default_rng(args.seed)
        boots = [da[rng.integers(0, len(da), len(da))].mean() for _ in range(n)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return da.mean(), lo, hi

    print(f"\n==== ASR best-of-N (LibriSpeech test.{args.split}, cond={tag}, n={len(rows)}) ====", flush=True)
    print(f"greedy WER = {mean('greedy_wer'):.4f}", flush=True)
    for N in Ns:
        print(f"  N={N}: oracle-BoN WER={mean(f'oracle_wer_{N}'):.4f}  MBR WER={mean(f'mbr_wer_{N}'):.4f}", flush=True)
    for sel in ("oracle", "mbr"):
        m, lo, hi = paired_ci("greedy_wer", f"{sel}_wer_{Ns[-1]}")
        sig = "SIG (CI excludes 0)" if lo > 0 else "n.s."
        print(f"  ΔWER greedy→{sel}@N{Ns[-1]} = {m:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  {sig}", flush=True)
    print("ASR_BON_DONE", flush=True)


if __name__ == "__main__":
    main()

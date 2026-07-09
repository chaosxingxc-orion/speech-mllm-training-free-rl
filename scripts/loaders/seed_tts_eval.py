"""seed_tts_eval — Seed-TTS-Eval loader, RE-PURPOSED as a clean ASR pair (K1/K2, coverage taxonomy).

Recipe: ``wiki/2026-07-09-coverage-dataset-taxonomy.md`` K1/K2 + the "改判" (re-judged) note in
``survey/2026-07-09-datasets-lock-second14.md``: the native TTS task is unusable offline (it needs
the model to SPEAK), so the benchmark is re-purposed as a clean zh(404)+en(363)-scale ASR pair —
audio -> read out its paired transcript.

**Audio-column ambiguity — RESOLVED empirically (verified 2026-07-09), and it resolves the OPPOSITE
of what the survey assumed.** The survey text says "audio/ans pairing ↔ text" i.e. it expected the
embedded ``audio`` column to be the TARGET wav (the one that reads ``text``, referenced by the
``ans`` path). On-disk fact-check of both language parquets says otherwise:

  - Each row has SIX columns: ``filename``, ``prompt_text``, ``WavPath`` (a relative path string,
    e.g. ``"prompt-wavs/10002287-00000094.wav"``), ``text``, ``ans`` (a relative path string, e.g.
    ``"wavs/10002287-00000095.wav"``), and ``audio`` (an embedded ``{"bytes", "path"}`` struct with
    ``path`` e.g. ``"10002287-00000094.wav"``).
  - ``audio["path"]``'s basename matches ``WavPath``'s basename EXACTLY (checked over sampled rows)
    — i.e. the embedded ``audio`` bytes ARE the **prompt** wav (the one that reads ``prompt_text``),
    not the ``ans``/target wav.
  - The ``ans`` and ``WavPath`` columns are bare path STRINGS pointing at a ``wavs/`` /
    ``prompt-wavs/`` directory layout from the original (non-parquet) release. Neither directory
    exists anywhere under ``datasets/seed-tts-eval/`` on this box (only ``en/`` and ``zh/``
    parquet shards + README are present) — so the ``ans``/target wav bytes this dataset would need
    for an (audio=ans-wav, gold=text) pairing are **not materialized on disk at all**, only
    referenced by path string. That pairing is therefore not constructible from what we have.
  - What IS constructible, and is exactly as "clean" an ASR pair: **(wav=audio bytes,
    gold=prompt_text)** — a real recorded utterance paired with its own verified transcript. This
    loader implements that pairing. ``text``/``ans``/``WavPath`` are kept in ``meta`` for
    provenance in case the missing wavs get fetched later and the original (target-wav, text)
    pairing becomes constructible.

On-disk layout: ``datasets/seed-tts-eval/<lang>/train-*.parquet`` (zh: 5 shards / 2020 rows; en: 3
shards / 1088 rows — note these totals are larger than the survey's "zh(404)+en(363)" estimate,
which likely referred to a de-duplicated hard-case subset of the original benchmark not visible from
this repackaged full parquet; not further chased here). ``filename`` is unique per row within each
language (verified).

``split`` selects the language: ``"zh"`` (default) or ``"en"``.

Row shape:
    wav  -> materialized wav path (the prompt recording)
    gold -> str, ``prompt_text`` (what the audio actually says)
    meta -> {"dataset": "seed-tts-eval", "item_id": <filename>, "split": <lang>,
             "text": <target text, unpaired>, "ans": <target wav path string, unmaterialized>,
             "wav_path_col": <WavPath string>}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

_REVISION = None  # see docs/datasets.lock.json
_LANGS = ("zh", "en")


def load_seed_tts_eval(split: str = "zh", n: int | None = None, seed: int = SLICE_SEED) -> list:
    import numpy as np
    import pyarrow.parquet as pq

    if split not in _LANGS:
        raise ValueError(f"seed-tts-eval: unknown split {split!r} (accepts: {_LANGS})")

    lang_dir = datasets_dir() / "seed-tts-eval" / split
    files = sorted(lang_dir.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train-*.parquet under {lang_dir}")

    cols = ["filename", "prompt_text", "WavPath", "text", "ans", "audio"]
    rows = []
    for f in files:
        rows += pq.read_table(f, columns=cols).to_pylist()
    rows.sort(key=lambda r: r["filename"])  # stable enumeration order before sampling

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows)) if n is not None else np.arange(len(rows))
    idx = idx[:n] if n is not None else idx

    cache_dir = wav_cache_dir("seed-tts-eval")
    out = []
    ids = []
    for j in idx:
        r = rows[int(j)]
        item_id = r["filename"]
        ids.append(item_id)
        wav = wav_from_bytes(r["audio"]["bytes"], f"seedtts_{split}_{item_id}", cache_dir)
        out.append({
            "wav": wav,
            "gold": r["prompt_text"],
            "meta": {
                "dataset": "seed-tts-eval",
                "item_id": item_id,
                "split": split,
                "text": r["text"],
                "ans": r["ans"],
                "wav_path_col": r["WavPath"],
            },
        })

    freeze_ids("seed-tts-eval", ids, dataset="seed-tts-eval", revision=_REVISION, seed=seed,
               extra={"split": split, "n_requested": n})
    return out


if __name__ == "__main__":
    from _common import data_root

    os.environ.setdefault("SPEECHRL_DATA_DIR", str(data_root()))
    rows = load_seed_tts_eval(n=3)
    for r in rows:
        print(r["meta"]["item_id"], "gold=", repr(r["gold"])[:60], "wav_exists=", os.path.exists(r["wav"]))

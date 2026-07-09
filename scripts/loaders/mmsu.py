"""loaders/mmsu.py — MMSU audio-reasoning MCQ loader (B6-retrieval-tools).

Gated on task A1's metadata refetch having landed (see
``datasets/mmsu/_metadata_refetch_2026-07-09.txt``): as of 2026-07-09 it HAS landed —
``datasets/mmsu/data/train-0000{0,1,2}-of-00003.parquet`` (5000 rows total, joined-in
question/choices/answer_gt) sit alongside the pre-existing ``datasets/mmsu/audio/*.wav``
(4956 files). Before the refetch, ``datasets/mmsu`` had audio but NO gold labels anywhere in
the tree (a content-level false-COMPLETE, see
`wiki/2026-07-09-coverage-dataset-taxonomy.md` §0 item 2) — this loader implements the full
join now that gold exists; if the metadata parquet is ever absent again, ``load_mmsu`` raises
``NeedsMetadata`` rather than silently returning an empty/broken slice.

DISCREPANCIES from the survey recipe (documented per the loader-contract instructions):

1. **File naming says "train", content is the whole eval benchmark.** The 3 parquet shards are
   named ``train-*-of-00003.parquet`` but there is no separate val/test split anywhere in the
   tree — all 5000 rows are the MCQ eval set (schema: id, task_name, audio, question,
   choice_a..d, answer_gt, category, sub-category, sub-sub-category,
   linguistics_sub_discipline; 47 task_name values, e.g. accent_identification,
   speech_translation, ...). ``load_mmsu`` therefore only accepts ``split="test"`` and reads
   all 3 shards as one pool (mirrors csemotions' "only a train split -> self-split" precedent
   already noted in the taxonomy doc for a different dataset).
2. **19.1% of rows have no on-disk ``audio/<path>.wav`` file** (957 / 5000, field-verified this
   task) despite the metadata's ``audio`` column always carrying non-empty embedded bytes for
   every row. Root cause not diagnosed here (likely a partial/interrupted earlier audio-only
   fetch pass vs. the full metadata-embedded set). ``load_mmsu`` therefore does NOT assume the
   on-disk tree is complete: it prefers the on-disk file (``audio/<path>``, avoids duplicate
   storage) when present, and falls back to materializing from the parquet's embedded
   ``audio.bytes`` (via ``_common.wav_from_bytes``, cached) when it is missing — so every row
   is servable regardless of on-disk gaps.
3. ``answer_gt`` was spot-checked (n=20 random rows) to match exactly one of
   ``choice_a..choice_d`` (case-insensitive, whitespace-stripped) with zero mismatches; a row
   where none/more-than-one choice matches is skipped with a printed warning rather than
   silently mis-scored (should not occur in practice per the spot check, but is a real
   possibility over the full 5000, not exhaustively verified here).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders

from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

CHOICE_COLS = ["choice_a", "choice_b", "choice_c", "choice_d"]


class NeedsMetadata(RuntimeError):
    """Raised when datasets/mmsu/data/*.parquet (the A1 metadata refetch) is not present."""


def _mmsu_dir() -> Path:
    return datasets_dir() / "mmsu"


def _metadata_files() -> list[Path]:
    d = _mmsu_dir() / "data"
    return sorted(d.glob("train-*-of-00003.parquet")) if d.exists() else []


def _select_indices(n_total: int, n: int | None, seed: int) -> list[int]:
    import numpy as np

    if n is None or n >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n, replace=False)
    return sorted(int(i) for i in idx)


def load_mmsu(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list[dict]:
    """Load `n` MMSU MCQ items (joined question/choices/answer_gt <-> audio).

    Row shape: ``wav`` = on-disk ``audio/<path>`` if present, else a materialized path from
    embedded bytes (see module docstring, discrepancy 2); ``gold`` = int index (0-3) of the
    correct choice; ``meta`` = ``{"dataset", "item_id", "split", "task_name", "category",
    "sub_category", "sub_sub_category", "opts", "question"}``.

    Raises:
        NeedsMetadata: if the A1 metadata refetch parquet files are absent (only ``audio/`` on
            disk, no gold) — see module docstring.
    """
    import pyarrow.parquet as pq

    if split != "test":
        raise ValueError(f"mmsu is a single eval pool, only split='test' is supported (got {split!r})")

    files = _metadata_files()
    if not files:
        raise NeedsMetadata(
            "datasets/mmsu/data/train-*-of-00003.parquet not found -- mmsu on this box only has "
            "audio/*.wav with no gold labels (content-level false-COMPLETE; see "
            "wiki/2026-07-09-coverage-dataset-taxonomy.md #0 item 2). Run the A1 metadata "
            "refetch (ddwang2000/MMSU data/ shards, revision "
            "548e2283105825bf908a7db5c09c00dbcf42bd4c via hf-mirror) before using this loader."
        )

    cols = ["id", "task_name", "audio", "question", *CHOICE_COLS, "answer_gt",
            "category", "sub-category", "sub-sub-category"]
    rows = []
    for f in files:
        rows += pq.read_table(f, columns=cols).to_pylist()
    rows_sorted = sorted(rows, key=lambda r: r["id"])

    idx = _select_indices(len(rows_sorted), n, seed)
    audio_dir = _mmsu_dir() / "audio"
    cache_dir = wav_cache_dir("mmsu")

    out = []
    ids = []
    for i in idx:
        r = rows_sorted[i]
        choices = [r[c] for c in CHOICE_COLS]
        ans = str(r["answer_gt"]).strip().lower()
        matches = [k for k, c in enumerate(choices) if str(c).strip().lower() == ans]
        if len(matches) != 1:
            print(f"[load_mmsu] WARNING: skipping {r['id']} -- answer_gt matched "
                  f"{len(matches)} choices (expected exactly 1)", flush=True)
            continue

        on_disk = audio_dir / r["audio"]["path"]
        if on_disk.exists():
            wav = str(on_disk)
        else:
            wav = wav_from_bytes(r["audio"]["bytes"], r["id"], cache_dir)

        item_id = r["id"]
        ids.append(item_id)
        out.append({
            "wav": wav,
            "gold": matches[0],
            "meta": {
                "dataset": "mmsu", "item_id": item_id, "split": split,
                "task_name": r["task_name"], "category": r["category"],
                "sub_category": r["sub-category"], "sub_sub_category": r["sub-sub-category"],
                "opts": choices, "question": r["question"],
            },
        })

    freeze_ids("mmsu", ids, dataset="mmsu",
               revision="548e2283105825bf908a7db5c09c00dbcf42bd4c", seed=seed)
    return out

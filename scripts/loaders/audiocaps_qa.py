"""loaders/audiocaps_qa.py — AudioCaps-QA audio-event-QA loader (B6-retrieval-tools).

File is named with an underscore (``audiocaps_qa.py``, ``load_audiocaps_qa``) because
``audiocaps-qa`` (with a hyphen, matching the on-disk dataset dir name) is not a valid Python
identifier; register it under the hyphenated key ``"audiocaps-qa"`` in ``registry.py`` (see
the ``mmau-mini: load_mmau`` precedent already in that file's comment).

LICENSE: AudioCaps/AudioSet-derived (YouTube clips); the HF card has no license field ->
eval-only prudent (per survey notes).

On-disk layout (confirmed 2026-07-09): a single ready parquet,
``datasets/audiocaps-qa/data/test-00000-of-00001.parquet``, 313 rows, test-only. Schema:
``context`` (dict, embedded audio bytes+path — ~10s AudioCaps/AudioSet environmental-sound
clip), ``instruction`` (str, the question), ``answer`` (str, open-ended free-text gold). No id
column and no stratification field (audio is heterogeneous environmental sound, per the
survey recipe) — item ids are the parquet row index, which is stable since it's a single file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders

from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402


def _parquet_path() -> Path:
    return datasets_dir() / "audiocaps-qa" / "data" / "test-00000-of-00001.parquet"


def _select_indices(n_total: int, n: int | None, seed: int) -> list[int]:
    import numpy as np

    if n is None or n >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n, replace=False)
    return sorted(int(i) for i in idx)


def load_audiocaps_qa(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list[dict]:
    """Load `n` items from AudioCaps-QA (313 rows total, test-only).

    Row shape: ``wav`` = materialized wav path (from embedded ``context`` bytes); ``gold`` =
    ``answer`` (free-text str); ``meta`` = ``{"dataset", "item_id", "split", "instruction"}``.
    """
    import pyarrow.parquet as pq

    if split != "test":
        raise ValueError(f"audiocaps-qa is test-only (got split={split!r})")

    rows = pq.read_table(_parquet_path()).to_pylist()  # single file, row order is stable on disk
    idx = _select_indices(len(rows), n, seed)

    cache_dir = wav_cache_dir("audiocaps-qa")
    out = []
    ids = []
    for i in idx:
        r = rows[i]
        item_id = f"row{i}"
        ids.append(item_id)
        wav = wav_from_bytes(r["context"]["bytes"], item_id, cache_dir)
        out.append({
            "wav": wav,
            "gold": r["answer"],
            "meta": {"dataset": "audiocaps-qa", "item_id": item_id, "split": split,
                      "instruction": r["instruction"]},
        })

    freeze_ids("audiocaps-qa", ids, dataset="audiocaps-qa", seed=seed)
    return out

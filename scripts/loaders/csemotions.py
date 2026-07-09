"""csemotions — CSEMOTIONS Mandarin emotional-speech loader (task B2-paralinguistic).

Layout on disk (SPEECHRL_DATA_DIR/datasets/csemotions/data/):
    train-0000{0..7}-of-00008.parquet -- ONE logical split ("train"), 4160 rows total
    columns: audio={bytes, path}, text, emotion, speaker

Confirmed empirically 2026-07-09 (schema + sample rows across shard 0):
    7 emotions: angry, fearful, happy, neutral, playfulness, sad, surprise
    10 speakers total (e.g. "female001", "female002", ...) across all 8 shards
    license: apache-2.0 (README.md front-matter) -- no NC restriction, unlike speech-massive

Self-split with frozen ids: the dataset ships only a "train" split (no held-out test), so
this loader treats the WHOLE table as the sampling pool for any ``split`` value and relies
entirely on ``freeze_ids`` (deterministic seeded slice) to make a given "test"-style call
reproducible -- there is no dataset-defined train/test boundary to honor or violate. Audio
comes embedded as parquet ``{bytes, path}`` structs, so every row is materialized to a cached
wav via ``_common.wav_from_bytes`` (first call decodes+caches; repeats are free).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402


def load_csemotions(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load CSEMOTIONS rows via a seeded slice of the (single, "train") parquet pool.

    ``split`` is accepted for interface-contract uniformity but does not change WHICH rows
    are eligible (see module docstring) -- only the RNG draw (seed) determines the sample,
    so callers wanting disjoint "train"/"test" pools should pass different ``seed`` values
    (or draw a large ``n`` once and partition themselves) rather than relying on ``split``.
    """
    import numpy as np
    import pyarrow.parquet as pq

    root = datasets_dir() / "csemotions" / "data"
    shard_paths = sorted(root.glob("train-*-of-*.parquet"))  # sorted() per README determinism rule
    if not shard_paths:
        raise FileNotFoundError(f"csemotions parquet shards not found under: {root}")

    # Build a stable (shard_idx, row_idx) -> row-id enumeration WITHOUT loading audio bytes
    # for the whole table first (cheap columns only), matching the "enumerate before sampling"
    # discipline; audio bytes are pulled per-row only for the rows actually selected.
    row_counts = [pq.ParquetFile(p).metadata.num_rows for p in shard_paths]
    total = sum(row_counts)

    rng = np.random.default_rng(seed)
    if n is not None and n < total:
        flat_idx = rng.choice(total, size=n, replace=False)
        flat_idx.sort()
    else:
        flat_idx = np.arange(total)

    # map flat index -> (shard, local_row)
    boundaries = np.cumsum([0] + row_counts)
    by_shard: dict[int, list[int]] = {}
    for fi in flat_idx:
        shard_i = int(np.searchsorted(boundaries, fi, side="right") - 1)
        local_i = int(fi - boundaries[shard_i])
        by_shard.setdefault(shard_i, []).append(local_i)

    cache_dir = wav_cache_dir("csemotions")
    rows = []
    sampled_ids = []
    for shard_i in sorted(by_shard):
        local_idx = sorted(by_shard[shard_i])
        table = pq.read_table(shard_paths[shard_i], columns=["audio", "text", "emotion", "speaker"])
        for li in local_idx:
            r = table.slice(li, 1).to_pylist()[0]
            item_id = f"csemotions/shard{shard_i}_row{li}"
            wav_key = f"shard{shard_i}_row{li}"
            wav_path = wav_from_bytes(r["audio"]["bytes"], wav_key, cache_dir)
            gold = {"emotion": r["emotion"], "speaker": r["speaker"], "text": r["text"]}
            meta = {"dataset": "csemotions", "item_id": item_id, "split": split, "shard": shard_i}
            rows.append({"wav": wav_path, "gold": gold, "meta": meta})
            sampled_ids.append(item_id)

    freeze_ids("csemotions", sampled_ids, dataset="csemotions", seed=seed, extra={"split": split})
    return rows

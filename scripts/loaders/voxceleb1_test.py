"""voxceleb1-test-split — VoxCeleb1 test-split speaker-ID loader (task B2-paralinguistic).

Layout on disk (SPEECHRL_DATA_DIR/datasets/voxceleb1-test-split/data/):
    test-0000{0..2}-of-00003.parquet -- ONE split ("test"), 4874 rows, 40 speakers
    columns: audio={bytes, path}, id, speaker_id

Confirmed empirically 2026-07-09: ``id`` looks like
"id10270+5r0dWxy17C8+00001.wav" (speaker_id + "+" + youtube-video-id + "+" + utt-idx +
".wav") and IS the natural item key (unique per row, human-legible, stable) -- this loader
keys ``meta["item_id"]`` off ``id`` directly rather than a shard/row-index pair, per the B2
spec ("rows {wav, gold=speaker_id} keyed by the 'id' field").
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402


def load_voxceleb1_test(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load VoxCeleb1-test rows via a seeded slice, keyed by the dataset's own ``id`` field.

    Only "test" is meaningful (it's the only split the dataset ships); other ``split`` values
    raise NotImplementedError rather than silently returning the same pool under a different
    label.
    """
    import numpy as np
    import pyarrow.parquet as pq

    if split != "test":
        raise NotImplementedError(
            f"voxceleb1-test-split split={split!r} not available -- this dataset ships only "
            "a 'test' split (hence the dataset name); pass split='test'."
        )

    root = datasets_dir() / "voxceleb1-test-split" / "data"
    shard_paths = sorted(root.glob("test-*-of-*.parquet"))  # sorted() per README determinism rule
    if not shard_paths:
        raise FileNotFoundError(f"voxceleb1-test-split parquet shards not found under: {root}")

    # Read the light columns (id, speaker_id) across all shards first to build a stable,
    # id-sorted enumeration before sampling -- audio bytes are pulled per-row only for the
    # rows actually selected (mirrors csemotions.py's discipline).
    id_index = []  # list of (shard_i, local_i, id, speaker_id), sorted by id
    for shard_i, p in enumerate(shard_paths):
        table = pq.read_table(p, columns=["id", "speaker_id"])
        ids = table.column("id").to_pylist()
        spks = table.column("speaker_id").to_pylist()
        for local_i, (item_id, spk) in enumerate(zip(ids, spks)):
            id_index.append((shard_i, local_i, item_id, spk))
    id_index.sort(key=lambda t: t[2])  # sort by 'id' string, not file/row order

    rng = np.random.default_rng(seed)
    if n is not None and n < len(id_index):
        idx = rng.choice(len(id_index), size=n, replace=False)
        idx.sort()
        id_index = [id_index[int(i)] for i in idx]

    # group selections by shard for efficient per-shard reads
    by_shard: dict[int, list[int]] = {}
    for shard_i, local_i, _item_id, _spk in id_index:
        by_shard.setdefault(shard_i, []).append(local_i)

    cache_dir = wav_cache_dir("voxceleb1-test-split")
    wav_by_key: dict[tuple, str] = {}
    for shard_i, local_idx in by_shard.items():
        local_idx_sorted = sorted(set(local_idx))
        table = pq.read_table(shard_paths[shard_i], columns=["audio", "id"])
        for li in local_idx_sorted:
            r = table.slice(li, 1).to_pylist()[0]
            wav_key = r["id"].rsplit(".", 1)[0].replace("+", "_")  # filesystem-safe cache key
            wav_by_key[(shard_i, li)] = wav_from_bytes(r["audio"]["bytes"], wav_key, cache_dir)

    rows = []
    sampled_ids = []
    for shard_i, local_i, item_id, spk in id_index:
        wav_path = wav_by_key[(shard_i, local_i)]
        meta = {"dataset": "voxceleb1-test-split", "item_id": item_id, "split": split}
        rows.append({"wav": wav_path, "gold": spk, "meta": meta})
        sampled_ids.append(item_id)

    freeze_ids("voxceleb1-test-split", sampled_ids, dataset="voxceleb1-test-split", seed=seed,
               extra={"split": split})
    return rows

"""aishell_1 — AISHELL-1 ASR-zh loader (K2 content/ASR-zh, coverage taxonomy).

Recipe: ``wiki/2026-07-09-coverage-dataset-taxonomy.md`` K2 + ``survey/2026-07-09-datasets-lock-first14.md``
K2 note: "aishell-1 (parquet ready; train=memory pool / dev+test=natural eval split)".

On-disk layout (verified 2026-07-09, matches the recipe): ``datasets/aishell-1/data/<prefix>-*.parquet``
with prefixes ``test`` (3 files, 7176 rows), ``validation`` (5 files, 14326 rows; this is the "dev"
split — HF calls it ``validation``), ``train`` (35 files, 120098 rows; the memory-pool split). Each
row has ``audio`` (``{"bytes", "path"}``; ``path`` e.g. ``"BAC009S0764W0121.wav"``, used as item_id)
and ``transcription`` — space-segmented Chinese words with a leading run of spaces, e.g.
``"    甚至  出现  交易  几乎  停滞  的  情况"`` (word-segmentation markers, not real silence/pauses).

Per task spec, ``gold`` strips ALL whitespace (the word-segmentation spaces carry no CER-relevant
information and jiwer/CER scoring must not count them as edits) so ``gold`` is a bare character
string, e.g. ``"甚至出现交易几乎停滞的情况"``. The untouched original is kept at
``meta["gold_raw"]`` for provenance/debugging.

``split`` accepts ``"test"`` (default), ``"dev"``/``"validation"`` (both map to the HF ``validation``
prefix), or ``"train"`` (the pool split).

Row shape:
    wav  -> materialized wav path
    gold -> str, whitespace-stripped transcript (CER-ready)
    meta -> {"dataset": "aishell-1", "item_id": <wav stem>, "split": <split>, "gold_raw": <original>}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

_REVISION = None  # see docs/datasets.lock.json
_SPLIT_PREFIX = {"test": "test", "dev": "validation", "validation": "validation", "train": "train"}


def load_aishell_1(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    import numpy as np
    import pyarrow.parquet as pq

    prefix = _SPLIT_PREFIX.get(split)
    if prefix is None:
        raise ValueError(f"aishell-1: unknown split {split!r} (accepts: {sorted(_SPLIT_PREFIX)})")

    data_dir = datasets_dir() / "aishell-1" / "data"
    files = sorted(data_dir.glob(f"{prefix}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {prefix}-*.parquet under {data_dir}")

    rows = []
    for f in files:
        rows += pq.read_table(f, columns=["audio", "transcription"]).to_pylist()
    rows.sort(key=lambda r: r["audio"]["path"])  # stable enumeration order before sampling

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows)) if n is not None else np.arange(len(rows))
    idx = idx[:n] if n is not None else idx

    cache_dir = wav_cache_dir("aishell-1")
    out = []
    ids = []
    for j in idx:
        r = rows[int(j)]
        item_id = os.path.splitext(r["audio"]["path"])[0]
        ids.append(item_id)
        wav = wav_from_bytes(r["audio"]["bytes"], f"aishell1_{split}_{item_id}", cache_dir)
        gold_raw = r["transcription"]
        gold = "".join(gold_raw.split())  # strip all whitespace (word-segmentation markers) for CER
        out.append({
            "wav": wav,
            "gold": gold,
            "meta": {
                "dataset": "aishell-1",
                "item_id": item_id,
                "split": split,
                "gold_raw": gold_raw,
            },
        })

    freeze_ids("aishell-1", ids, dataset="aishell-1", revision=_REVISION, seed=seed,
               extra={"split": split, "n_requested": n})
    return out


if __name__ == "__main__":
    from _common import data_root

    os.environ.setdefault("SPEECHRL_DATA_DIR", str(data_root()))
    rows = load_aishell_1(n=3)
    for r in rows:
        print(r["meta"]["item_id"], "gold=", repr(r["gold"]), "wav_exists=", os.path.exists(r["wav"]))

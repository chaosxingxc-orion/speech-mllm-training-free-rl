"""librispeech — LibriSpeech ASR-en loader (K1 content/ASR-en, coverage taxonomy).

Recipe: ``wiki/2026-07-09-coverage-dataset-taxonomy.md`` K1 + ``survey/2026-07-09-datasets-lock-first14.md``
("librispeech: include, eval=test.clean n40/60 length-stratified; memory-pool=train960 (pool != eval,
a natural split); 1-WER reward; CLAP-must-die negative control").

On-disk layout (verified 2026-07-09, matches the recipe): ``datasets/librispeech/all/<split>/*.parquet``
— an HF-audio layout with one directory per split (``test.clean``, ``test.other``,
``validation.clean``, ``validation.other``, ``train.clean.100``, ``train.clean.360``,
``train.other.500``). Each row has ``audio`` (``{"bytes", "path"}``, embedded FLAC bytes),
``text`` (upper-case transcript, no punctuation — native LibriSpeech convention), ``speaker_id``,
``chapter_id``, ``id`` (e.g. ``"6930-75918-0000"``, globally unique within a split).

``split='test.clean'`` (default) and ``split='validation.clean'`` are the two eval-shaped splits
this task calls for; ``split='train.clean.100'`` exposes the 100h pool for memory/pool-vs-eval work
(``pool != eval`` is structural here — disjoint speakers/chapters by LibriSpeech design, not just a
different sample). Other split dirs on disk (``test.other``, ``validation.other``,
``train.clean.360``, ``train.other.500``) work too since the loader is split-name-generic.

Row shape:
    wav  -> materialized wav path (FLAC bytes decoded via ``_common.wav_from_bytes``)
    gold -> str, the upper-case transcript (``text``)
    meta -> {"dataset": "librispeech", "item_id": <id>, "split": <split>, "speaker_id": int,
             "chapter_id": int}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

_REVISION = None  # HF snapshot revision not separately pinned for this local layout; see docs/datasets.lock.json


def load_librispeech(split: str = "test.clean", n: int | None = None, seed: int = SLICE_SEED) -> list:
    import numpy as np
    import pyarrow.parquet as pq

    split_dir = datasets_dir() / "librispeech" / "all" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"librispeech split dir not found: {split_dir} "
            f"(available: {sorted(p.name for p in (datasets_dir() / 'librispeech' / 'all').iterdir())})"
        )
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {split_dir}")

    rows = []
    for f in files:
        rows += pq.read_table(f, columns=["id", "audio", "text", "speaker_id", "chapter_id"]).to_pylist()
    rows.sort(key=lambda r: r["id"])  # stable enumeration order before sampling (contract step 2)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows)) if n is not None else np.arange(len(rows))
    idx = idx[:n] if n is not None else idx

    cache_dir = wav_cache_dir("librispeech")
    out = []
    ids = []
    for j in idx:
        r = rows[int(j)]
        item_id = r["id"]
        ids.append(item_id)
        wav = wav_from_bytes(r["audio"]["bytes"], f"librispeech_{split}_{item_id}", cache_dir)
        out.append({
            "wav": wav,
            "gold": r["text"],
            "meta": {
                "dataset": "librispeech",
                "item_id": item_id,
                "split": split,
                "speaker_id": r["speaker_id"],
                "chapter_id": r["chapter_id"],
            },
        })

    freeze_ids("librispeech", ids, dataset="librispeech", revision=_REVISION, seed=seed,
               extra={"split": split, "n_requested": n})
    return out


if __name__ == "__main__":
    from _common import data_root

    os.environ.setdefault("SPEECHRL_DATA_DIR", str(data_root()))
    rows = load_librispeech(n=3)
    for r in rows:
        print(r["meta"]["item_id"], "gold=", repr(r["gold"])[:80], "wav_exists=", os.path.exists(r["wav"]))

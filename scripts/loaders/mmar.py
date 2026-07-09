"""mmar — MMAR (Massive Multi-disciplinary Audio Reasoning) loader (task B4-qa-mcq).

Recipe: ``survey/2026-07-09-datasets-lock-second14.md`` names MMAR-meta.json rows as
``{wav=audio_path, gold={answer, choices, category}}``.

On-disk layout (verified 2026-07-09, after the A2 extraction step already ran):
    datasets/mmar/MMAR-meta.json   -- a JSON **list** of 1000 row dicts (not JSONL), keys:
        id, audio_path (relative "./audio/<file>.wav"), question, choices (list[str]),
        answer (str, verbatim member of choices), modality, category, sub-category,
        language, source, url, timestamp
    datasets/mmar/audio/<file>.wav -- 1000 wav files, already extracted from
        mmar-audio.tar.gz (A2 done; the .tar.gz is left alongside, unused here)

No bytes-decoding is needed here (unlike the parquet-embedded-bytes datasets elsewhere in
this package) — MMAR ships one wav file per row directly on disk, so the loader only
resolves ``audio_path`` to an absolute path and checks it exists.

MMAR is a single held-out set (no train/val split on disk); only ``split="test"`` is wired
up, matching every other MCQ-family loader's default.

Row shape:
    wav  -> absolute path to the on-disk wav (no materialization/caching needed)
    gold -> {"answer": <str, verbatim choice text>, "choices": [str, ...], "category": <str>}
    meta -> {"dataset": "mmar", "item_id": <id>, "split": "test", "modality": <str>,
             "sub_category": <str>, "language": <str>}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402


def load_mmar(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load MMAR rows. Only ``split="test"`` is implemented (MMAR has no other split)."""
    import json

    import numpy as np

    if split != "test":
        raise NotImplementedError(f"mmar split={split!r} not wired up -- MMAR ships one set only.")

    root = datasets_dir() / "mmar"
    meta_path = root / "MMAR-meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"mmar MMAR-meta.json not found: {meta_path}")

    rows = json.load(open(meta_path, encoding="utf-8"))
    rows.sort(key=lambda r: r["id"])  # stable enumeration order before sampling (contract step 2)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows)) if n is not None else np.arange(len(rows))
    idx = idx[:n] if n is not None else idx

    out, ids = [], []
    for j in idx:
        r = rows[int(j)]
        item_id = r["id"]
        wav_path = root / r["audio_path"].lstrip("./")
        if not wav_path.exists():
            raise FileNotFoundError(f"mmar audio missing: {wav_path} (row id={item_id})")
        out.append({
            "wav": str(wav_path),
            "gold": {"answer": r["answer"], "choices": list(r["choices"]), "category": r["category"]},
            "meta": {
                "dataset": "mmar",
                "item_id": item_id,
                "split": split,
                "modality": r.get("modality"),
                "sub_category": r.get("sub-category"),
                "language": r.get("language"),
            },
        })
        ids.append(item_id)

    freeze_ids("mmar", ids, dataset="mmar", seed=seed, extra={"split": split, "n_requested": n})
    return out


if __name__ == "__main__":
    rows = load_mmar(n=3)
    for r in rows:
        print(r["meta"]["item_id"], "gold=", repr(r["gold"])[:120], "wav_exists=", os.path.exists(r["wav"]))

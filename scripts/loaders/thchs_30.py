"""thchs_30 — THCHS-30 ASR-zh loader (K2 content/ASR-zh "reading diversity pool", coverage taxonomy).

Recipe: ``wiki/2026-07-09-coverage-dataset-taxonomy.md`` K2 ("thchs-30 (reading-diversity pool,
needs extraction)") + task spec: ``.wav`` + ``.wav.trn`` sidecar, first line = char transcript, test
split.

**Discrepancy #1 vs the recipe** (documented per the task's "adapt and document" instruction): the
task text expects this dataset to need extraction "after A2". On disk (verified 2026-07-09) it is
ALREADY extracted with the canonical THCHS-30 layout at ``datasets/thchs-30/data_thchs30/``:
``train/`` (10000 wavs), ``dev/`` (893 wavs), ``test/`` (2495 wavs) — each a flat directory of real
(non-symlink) ``<utt_id>.wav`` files. (There is also a flat ``data/`` dir with all 13388 pairs
undivided by split; the loader reads the pre-split ``train/dev/test`` dirs directly for the WAV so
it never has to re-derive the split membership.) No extraction step was needed — this loader is NOT
blocked.

**Discrepancy #2, found only by actually reading the sidecar content** (not assumable from the file
listing alone): inside ``train/``, ``dev/``, ``test/`` the ``<utt_id>.wav.trn`` files are NOT the
transcript — they are tiny (~23-byte) POINTER files whose first line is a relative path back to the
real transcript, e.g. ``test/D7_952.wav.trn`` contains the single line ``"../data/D7_952.wav.trn"``.
Verified this pointer shape is 100% consistent across all 2495 ``test/*.wav.trn`` files (and spot
checked in ``train/``/``dev/``). The REAL 3-line transcript (char line, pinyin-with-tone line,
pinyin-split line) lives only under the flat ``data/`` dir. This loader follows the pointer
automatically (detects a sidecar whose first line looks like a relative ``../data/....wav.trn`` path
rather than a Chinese transcript, and re-reads from the resolved target) so callers never see the
pointer text as ``gold`` — but it is recorded here since a shallower loader (or an ``ls``-only
audit) that assumed "sidecar first line = transcript" as usual would have silently ingested "../data/
D7_952.wav.trn" as SER/ASR gold, which is why the loader's own sanity check (reject any first line
containing ``.wav.trn``) matters, not just a docstring note.

``data/<utt_id>.wav.trn`` format (3 lines): line 1 = character transcript (Chinese words separated
by spaces, e.g. ``"他 仅 凭 腰部 的 力量 ..."``), line 2 = pinyin-with-tone, line 3 = pinyin split
further into initials/finals. Only line 1 is used as ``gold`` here (kept as-is, spaces included —
unlike aishell-1 this loader does NOT strip the inter-word spaces per the task spec, which only
asked for "first line = char transcript"; strip at the caller/CER-metric layer if desired).

``split`` accepts ``"train"``, ``"dev"``, or ``"test"`` (default) for the WAV directory; the
transcript is always resolved through to ``data/`` regardless of split.

Row shape:
    wav  -> path to the on-disk wav (no materialization needed; THCHS-30 ships real per-split wavs)
    gold -> str, first line of the ``.wav.trn`` sidecar
    meta -> {"dataset": "thchs-30", "item_id": <utt_id>, "split": <split>}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402

_REVISION = None  # see docs/datasets.lock.json
_SPLITS = ("train", "dev", "test")


def load_thchs_30(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    import numpy as np

    if split not in _SPLITS:
        raise ValueError(f"thchs-30: unknown split {split!r} (accepts: {_SPLITS})")

    split_dir = datasets_dir() / "thchs-30" / "data_thchs30" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"thchs-30 split dir not found: {split_dir}")

    wavs = sorted(split_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no .wav files under {split_dir}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(wavs)) if n is not None else np.arange(len(wavs))
    idx = idx[:n] if n is not None else idx

    out = []
    ids = []
    for j in idx:
        wp = wavs[int(j)]
        trn = wp.with_suffix(wp.suffix + ".trn")  # <utt>.wav -> <utt>.wav.trn
        if not trn.exists():
            raise FileNotFoundError(f"thchs-30: missing sidecar transcript {trn}")
        with open(trn, encoding="utf-8") as fh:
            gold = fh.readline().rstrip("\n")
        if gold.endswith(".wav.trn"):  # pointer sidecar (see docstring discrepancy #2) -- follow it
            target = (trn.parent / gold).resolve()
            if not target.exists():
                raise FileNotFoundError(f"thchs-30: sidecar {trn} points at missing {target}")
            with open(target, encoding="utf-8") as fh:
                gold = fh.readline().rstrip("\n")
        item_id = wp.stem
        ids.append(item_id)
        out.append({
            "wav": str(wp),
            "gold": gold,
            "meta": {"dataset": "thchs-30", "item_id": item_id, "split": split},
        })

    freeze_ids("thchs-30", ids, dataset="thchs-30", revision=_REVISION, seed=seed,
               extra={"split": split, "n_requested": n})
    return out


if __name__ == "__main__":
    from _common import data_root

    os.environ.setdefault("SPEECHRL_DATA_DIR", str(data_root()))
    rows = load_thchs_30(n=3)
    for r in rows:
        print(r["meta"]["item_id"], "gold=", repr(r["gold"]), "wav_exists=", os.path.exists(r["wav"]))

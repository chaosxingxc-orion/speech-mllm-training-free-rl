"""slurp — SLURP spoken-language-understanding (intent + slots) loader (task B3-slu).

Layout on disk:
    gold jsonl : SPEECHRL_DATA_DIR/repos/slurp/dataset/slurp/{train,devel,test}.jsonl
    audio      : SPEECHRL_DATA_DIR/datasets/slurp/audio/slurp_real/<recordings[i].file>
                 (a byte-identical duplicate tree also lives at .../datasets/slurp/slurp_real/ —
                 both got populated by the fetch campaign; audio/slurp_real is used here since it
                 matches the audio/slurp_real.tar.gz extraction path in docs/datasets.lock.json)
    revision   : 8eb16545762be97ace75334109d73824217311f1 (pswietojanski/slurp, pinned in
                 docs/datasets.lock.json)

Recording selection: each jsonl line is ONE utterance (``slurp_id``) with MULTIPLE
``recordings`` — the same sentence captured by different devices/sessions. Filenames follow
``audio-<session>[-headset].flac``; despite the suffix, "-headset" recordings are the
headset-mic capture and the bare filename is the close-talk/far-field device capture. Per the
recipe (wiki/survey/2026-07-09-datasets-lock-first14.md, "slurp" row) this loader keeps exactly
ONE recording per ``slurp_id`` — the first non-headset file that actually exists on disk, else
the first headset file that exists (rare: 24/500 test utterances in a spot check had headset
recordings only) — so ``load_slurp`` returns (at most) one Row per ``slurp_id``, not one per
recording. A ``slurp_id`` with NO recording file present on disk is skipped (not raised), so one
missing audio file never sinks a whole split.

Splits: "test" -> test.jsonl (2,974 lines, eval, 18 scenarios per the survey recipe); "devel" ->
devel.jsonl (2,033); "train" -> train.jsonl (11,514, exposed as the memory/few-shot POOL split —
SLURP's own train/test is already a natural pool/eval split, mirroring the
librispeech train960/test.clean pattern in scripts/p2_baselines.py). ``train_synthetic.jsonl``
(TTS-synthesized utterances, audio under slurp_synth/) is deliberately NOT exposed — the recipe
only calls for the three real-recording splits.

gold = {"intent": <"<scenario>_<action>", e.g. "calendar_set">, "scenario": str, "action": str,
        "entities": [{"span": [tok_idx, ...], "type": "..."}, ...], "sentence": str}
``intent`` is SLURP's own ``intent`` field, already shipped in ``<scenario>_<action>`` form —
nothing is built/derived here, only carried through.

Scoring is NOT implemented here (loader contract) — for a caller that needs it:
  - Intent: exact-match on gold["intent"].
  - Slot F1: the SLURP-native scorer lives at
    SPEECHRL_DATA_DIR/repos/slurp/scripts/evaluation/evaluate.py (depends on sibling
    metrics.py/util.py in that same directory; expects a SLURP-format gold jsonl + a predictions
    file, i.e. is a CLI tool, not an importable function as-is).
  - A generic span-level slot-filling F1 ("the slue-toolkit slot scorer") also ships at
    SPEECHRL_DATA_DIR/repos/slue-toolkit/slue_toolkit/eval/eval_utils_ner.py
    (``get_ner_scores`` — precision/recall/F1 over (label, phrase, id) tuples). It is not
    SLURP-specific but can score (entity type, entity phrase) pairs derived from
    gold["entities"] + gold["sentence"]/tokens the same way it scores SLUE's own slot task.

A caller wanting a scenario-stratified eval slice should call this loader with ``n=None`` and
stratify/subsample itself by gold["scenario"] — this loader's own ``n`` is a plain seeded random
draw over slurp_ids (per the README contract), not stratified.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, data_root, datasets_dir, freeze_ids  # noqa: E402

_SPLIT_FILES = {"test": "test.jsonl", "devel": "devel.jsonl", "train": "train.jsonl"}
_REVISION = "8eb16545762be97ace75334109d73824217311f1"  # pswietojanski/slurp, docs/datasets.lock.json


def _pick_recording(recordings: list[dict], audio_dir: Path) -> str | None:
    """Pick one on-disk recording file: prefer non-headset, else headset; None if neither exists."""
    non_headset = [r["file"] for r in recordings if "-headset" not in r["file"]]
    headset = [r["file"] for r in recordings if "-headset" in r["file"]]
    for fname in non_headset + headset:
        if (audio_dir / fname).exists():
            return fname
    return None


def load_slurp(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load SLURP rows for ``split`` ("test" | "devel" | "train").

    Determinism: jsonl lines are sorted by ``slurp_id`` before sampling (never raw file order);
    ``n`` then draws a seeded, sorted-index slice via ``numpy.random.default_rng(seed)``, matching
    the README contract. Slurp_ids with no on-disk recording are dropped BEFORE the ``n`` draw, so
    a requested ``n`` is satisfied from ids that are actually loadable (when available).
    """
    import json

    import numpy as np

    fname = _SPLIT_FILES.get(split)
    if fname is None:
        raise ValueError(f"load_slurp: unknown split {split!r}; expected one of {sorted(_SPLIT_FILES)}")

    jsonl_path = data_root() / "repos" / "slurp" / "dataset" / "slurp" / fname
    if not jsonl_path.exists():
        raise FileNotFoundError(f"load_slurp: gold jsonl not found: {jsonl_path}")
    audio_dir = datasets_dir() / "slurp" / "audio" / "slurp_real"
    if not audio_dir.exists():
        raise FileNotFoundError(f"load_slurp: audio dir not found: {audio_dir}")

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda d: d["slurp_id"])  # sorted() per README's determinism rule

    items = []  # (slurp_id, record, recording_filename)
    for d in records:
        rec_file = _pick_recording(d["recordings"], audio_dir)
        if rec_file is not None:
            items.append((d["slurp_id"], d, rec_file))

    if n is not None and n < len(items):
        idx = np.random.default_rng(seed).choice(len(items), size=n, replace=False)
        items = [items[int(i)] for i in sorted(idx)]

    rows = []
    sampled_ids = []
    for slurp_id, d, rec_file in items:
        gold = {
            "intent": d["intent"],
            "scenario": d["scenario"],
            "action": d["action"],
            "entities": d["entities"],
            "sentence": d["sentence"],
        }
        rows.append({
            "wav": str(audio_dir / rec_file),
            "gold": gold,
            "meta": {
                "dataset": "slurp",
                "item_id": slurp_id,
                "split": split,
                "recording_file": rec_file,
                "non_headset": "-headset" not in rec_file,
            },
        })
        sampled_ids.append(slurp_id)

    freeze_ids("slurp", sampled_ids, dataset="slurp", revision=_REVISION, seed=seed,
               extra={"split": split, "n_requested": n})
    return rows


if __name__ == "__main__":
    os.environ.setdefault("SPEECHRL_DATA_DIR", str(data_root()))
    rows = load_slurp(n=3)
    for r in rows:
        print(r["meta"]["item_id"], "gold=", repr(r["gold"])[:120], "wav_exists=", os.path.exists(r["wav"]))

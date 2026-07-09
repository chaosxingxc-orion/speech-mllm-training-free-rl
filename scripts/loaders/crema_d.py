"""crema-d — CREMA-D emotion-recognition loader (task B2-paralinguistic).

Layout on disk (SPEECHRL_DATA_DIR/datasets/crema-d/):
    audios/{spk}_{sent}_{EMO}_{intensity}.wav   -- 7442 wavs
    train.csv, test.csv                          -- columns: path,classname

**CRITICAL documented gotcha** (wiki/survey/2026-07-09-datasets-lock-first14.md, "crema-d"
row): the ``classname`` column in train.csv/test.csv is UNRELIABLE — it disagrees with the
filename's own emotion code (``{EMO}``) on ~54% of rows. Verified empirically while writing
this loader, e.g. in test.csv:
    audios/1085_TSI_SAD_XX.wav,neutral    <- filename says SAD, csv says "neutral"  (CONFLICT)
    audios/1046_MTI_NEU_XX.wav,neutral    <- filename says NEU, csv says "neutral"  (agrees)
So this loader NEVER reads ``classname`` for the emotion gold label. The CSVs are used
ONLY to determine split membership (which ``path`` values belong to train vs. test); the
emotion label always comes from decoding the filename's own ``{EMO}`` code.

Filename code (CREMA-D original convention): ``{spk}_{sent}_{EMO}_{intensity}.wav``
    spk        4-digit speaker id, e.g. "1001"
    sent       one of 12 sentence codes (IEO, TIE, IOM, IWW, TAI, MTI, IWL, ITH, DFA, ITS,
               TSI, WSI) — the fixed sentence read aloud
    EMO        3-letter emotion code: ANG=anger, DIS=disgust, FEA=fear, HAP=happy,
               NEU=neutral, SAD=sad  (6 classes)
    intensity  LO / MD / HI, or XX when intensity is unspecified/not applicable

Rows: {"wav": <path>, "gold": {"emo", "spk", "sent", "intensity"}, "meta": {...}}. ``gold``
is a dict (not a bare label) so a caller can also probe speaker-ID (same "spk", different
"emo"/"sent" = natural disentanglement contrast pair, per the survey's T-C typology) without
a second loader.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402

EMO_CODE = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
SENT_CODES = {"IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"}
INTENSITY_CODES = {"LO", "MD", "HI", "XX"}


def _parse_filename(stem: str) -> dict:
    """Decode ``{spk}_{sent}_{EMO}_{intensity}`` from a crema-d wav filename stem."""
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"crema-d filename does not match {{spk}}_{{sent}}_{{EMO}}_{{int}}: {stem}")
    spk, sent, emo_code, intensity = parts
    if emo_code not in EMO_CODE:
        raise ValueError(f"crema-d unknown emotion code {emo_code!r} in {stem}")
    return {
        "spk": spk,
        "sent": sent,
        "emo": EMO_CODE[emo_code],
        "intensity": intensity,
    }


def _read_split_paths(csv_path: Path) -> list[str]:
    """Read a crema-d split CSV; return the ``path`` column only (classname is untrusted)."""
    import csv

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["path"] for row in reader]


def load_crema_d(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load CREMA-D rows for ``split`` ("train" or "test"), decoding gold from the filename.

    Determinism: split membership comes from the dataset's own train.csv/test.csv (fixed,
    not resampled); ``n`` then draws a seeded, sorted-order slice from that split's rows via
    ``numpy.random.default_rng(seed)``, matching the README contract.
    """
    import numpy as np

    root = datasets_dir() / "crema-d"
    csv_name = "train.csv" if split == "train" else "test.csv"
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"crema-d split CSV not found: {csv_path}")

    paths = sorted(set(_read_split_paths(csv_path)))  # sorted() per README's determinism rule

    rng = np.random.default_rng(seed)
    if n is not None and n < len(paths):
        idx = rng.choice(len(paths), size=n, replace=False)
        idx.sort()
        paths = [paths[int(i)] for i in idx]

    rows = []
    sampled_ids = []
    for rel_path in paths:
        wav_path = root / rel_path
        if not wav_path.exists():
            raise FileNotFoundError(f"crema-d audio missing: {wav_path}")
        stem = wav_path.stem  # e.g. "1085_TSI_SAD_XX"
        gold = _parse_filename(stem)
        item_id = f"crema-d/{stem}"
        rows.append({
            "wav": str(wav_path),
            "gold": gold,
            "meta": {"dataset": "crema-d", "item_id": item_id, "split": split},
        })
        sampled_ids.append(item_id)

    freeze_ids("crema-d", sampled_ids, dataset="crema-d", seed=seed, extra={"split": split})
    return rows

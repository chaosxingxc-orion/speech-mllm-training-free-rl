"""esd — Emotional Speech Dataset (Mandarin subset) loader (task B2-paralinguistic).

Layout on disk (SPEECHRL_DATA_DIR/datasets/esd/"Emotion Speech Dataset"/):
    0001/.../0001.txt            -- tab-separated: utt_id \\t text \\t emotion (1750 lines)
    0001/{Angry,Happy,Neutral,Sad,Surprise}/0001_XXXXXX.wav
    ... speakers 0001-0020 total; 0001-0010 = Mandarin (zh), 0011-0020 = English (en)
        (confirmed empirically 2026-07-09: 0001.txt rows are Chinese text with Chinese
        emotion labels e.g. "中立"/"伤心"/"快乐"/"惊喜"/"生气"; 0011.txt rows are English)

Per B2 spec: this loader is scoped to the **zh speakers 0001-0010** only (5 emotions:
neutral/sad/happy/surprise/angry -- English folder names, used as the canonical ``emo``
value regardless of the txt file's language, so zh and en gold are comparable if a future
loader adds 0011-0020).

**Discrepancy from the recipe, documented here**: the recipe assumes ESD's official
train/evaluation/test per-emotion subdivision (the canonical HKUST release nests each
emotion folder into Train/Test/Evaluation). This HF mirror is FLATTENED -- confirmed
empirically 2026-07-09 (``find .../0001/Angry -maxdepth 1 -type d`` returns nothing; all 350
wavs sit directly under ``Angry/``) -- so the official split boundary is not recoverable from
this copy. Rather than fabricate a train/test boundary that doesn't match upstream, this
loader self-splits: it deterministically holds out the LAST 20% of each (speaker, emotion)
block (sorted by utt_id) as "test" and the rest as "train", using frozen ids (via
``freeze_ids``) so the boundary is reproducible and auditable even though it isn't the
original ESD boundary. This mirrors the "self-split with frozen ids" pattern already used
for csemotions (see ``csemotions.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402

ZH_SPEAKERS = [f"{i:04d}" for i in range(1, 11)]  # 0001..0010
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]


def _read_speaker_txt(txt_path: Path) -> dict:
    """Parse ``{spk}.txt``: utt_id \\t text \\t emotion(zh label). Returns {utt_id: (text, emo_zh)}."""
    out = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            utt_id, text, emo_zh = parts
            out[utt_id] = (text, emo_zh)
    return out


def _enumerate_speaker(root: Path, spk: str) -> list[dict]:
    """All (wav_path, utt_id, text, emo) rows for one speaker, sorted by utt_id."""
    spk_dir = root / spk
    txt_path = spk_dir / f"{spk}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"esd speaker txt missing: {txt_path}")
    text_map = _read_speaker_txt(txt_path)

    items = []
    for emo in EMOTIONS:
        emo_dir = spk_dir / emo
        if not emo_dir.exists():
            continue
        for wav_path in sorted(emo_dir.glob(f"{spk}_*.wav")):
            utt_id = wav_path.stem  # e.g. "0001_000351"
            text, emo_zh = text_map.get(utt_id, (None, None))
            items.append({
                "wav_path": wav_path,
                "utt_id": utt_id,
                "text": text,
                "emo": emo,  # canonical English folder name
                "emo_zh": emo_zh,
            })
    items.sort(key=lambda r: r["utt_id"])
    return items


def load_esd(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load ESD (zh, speakers 0001-0010) rows for a self-split "train"/"test".

    See module docstring for the self-split rationale (this mirror lacks the official
    Train/Test/Evaluation subdirectories). Split boundary: per (speaker, emotion) block
    sorted by utt_id, the last 20% -> "test", the rest -> "train" (deterministic, no RNG
    needed for the boundary itself). ``n``/``seed`` then draw a seeded slice from the
    resulting split pool across all 10 speakers.
    """
    import numpy as np

    root = datasets_dir() / "esd" / "Emotion Speech Dataset"
    if not root.exists():
        raise FileNotFoundError(f"esd root not found: {root}")

    pool = []
    for spk in ZH_SPEAKERS:
        items = _enumerate_speaker(root, spk)
        # group by emotion to hold out the tail of EACH emotion block, not the tail overall
        by_emo: dict[str, list] = {}
        for it in items:
            by_emo.setdefault(it["emo"], []).append(it)
        for emo, emo_items in by_emo.items():
            cut = max(1, int(len(emo_items) * 0.8))
            block = emo_items[cut:] if split == "test" else emo_items[:cut]
            pool.extend({**it, "spk": spk} for it in block)

    pool.sort(key=lambda r: (r["spk"], r["emo"], r["utt_id"]))  # stable order before sampling

    rng = np.random.default_rng(seed)
    if n is not None and n < len(pool):
        idx = rng.choice(len(pool), size=n, replace=False)
        idx.sort()
        pool = [pool[int(i)] for i in idx]

    rows = []
    sampled_ids = []
    for it in pool:
        item_id = f"esd/{it['utt_id']}"
        gold = {"emo": it["emo"], "spk": it["spk"], "text": it["text"]}
        meta = {
            "dataset": "esd",
            "item_id": item_id,
            "split": split,
            "emo_zh": it["emo_zh"],
            "lang": "zh",
        }
        rows.append({"wav": str(it["wav_path"]), "gold": gold, "meta": meta})
        sampled_ids.append(item_id)

    freeze_ids("esd", sampled_ids, dataset="esd", seed=seed, extra={"split": split, "self_split": True})
    return rows

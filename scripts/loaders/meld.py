"""meld — MELD (Multimodal EmotionLines Dataset) loader (task B2-paralinguistic).

Layout on disk (SPEECHRL_DATA_DIR/datasets/meld/MELD.Raw/):
    test_sent_emo.csv                         -- columns incl. Dialogue_ID, Utterance_ID,
                                                  Emotion, Sentiment, Speaker, Utterance
    output_repeated_splits_test/diaX_uttY.mp4  -- the test-split video/audio, keyed by
                                                  (Dialogue_ID, Utterance_ID) -> "dia{X}_utt{Y}.mp4"

Discrepancy from the recipe: the recipe names the test-split directory generically; the
actual on-disk name (matching the official MELD.Raw release) is
``output_repeated_splits_test/`` (not e.g. ``test_splits/``) — confirmed by directory listing
2026-07-09. train/dev use ``train_splits/`` / ``dev_splits_complete/`` respectively (not
wired up here; B2 only asks for the test split).

**Blocker — ffmpeg is required and absent in this WSL env** (CLAUDE.md: no sudo, do not
install system packages; verified 2026-07-09 via ``command -v ffmpeg`` -> not found). MELD
ships utterances as .mp4 (video+audio); this loader needs a decoded .wav to hand back per
the Row contract. Rather than silently failing, ``load_meld`` runs fully up to the point of
audio materialization: it resolves the CSV join and the exact .mp4 path per row, then calls
``_extract_wav`` which raises ``NeedsExtraction`` (a clear, typed error) naming the missing
tool and the exact ffmpeg command needed, e.g.:
    ffmpeg -y -i "<meld>/MELD.Raw/output_repeated_splits_test/dia0_utt0.mp4" \
        -ar 16000 -ac 1 -vn "<cache>/dia0_utt0.wav"
Once ffmpeg is installed, no code change is required — ``_extract_wav`` will materialize and
cache the wav on first call.

7 emotions (unbalanced) per the survey's T-C typology: anger, disgust, fear, joy, neutral,
sadness, surprise — a caller doing accuracy should stratify/macro-F1 rather than raw accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir  # noqa: E402


class NeedsExtraction(RuntimeError):
    """Raised when a MELD utterance's .mp4 needs ffmpeg extraction to .wav and ffmpeg is missing."""


def _read_test_csv(csv_path: Path) -> list[dict]:
    import csv

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _ffmpeg_cmd(mp4_path: Path, wav_path: Path) -> list[str]:
    return ["ffmpeg", "-y", "-i", str(mp4_path), "-ar", "16000", "-ac", "1", "-vn", str(wav_path)]


def _extract_wav(mp4_path: Path, wav_path: Path) -> str:
    """Decode ``mp4_path`` to a 16kHz mono wav at ``wav_path`` via ffmpeg; return the wav path.

    Raises NeedsExtraction (not a bare CalledProcessError) if the extraction fails. Caller is
    expected to have already checked ffmpeg is on PATH (see ``load_meld``) so a full batch of
    missing-tool errors can be reported at once instead of failing on the first row.
    """
    import subprocess

    if wav_path.exists():
        return str(wav_path)
    cmd = _ffmpeg_cmd(mp4_path, wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise NeedsExtraction(
            f"ffmpeg extraction failed for {mp4_path} -> {wav_path}: "
            f"{e.stderr.decode(errors='replace')[:500]}"
        ) from e
    return str(wav_path)


def load_meld(split: str = "test", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load MELD rows for ``split`` (only "test" is wired up; others raise NotImplementedError).

    Joins test_sent_emo.csv (Dialogue_ID, Utterance_ID) -> output_repeated_splits_test/
    dia{D}_utt{U}.mp4, then extracts each selected utterance to a cached wav via ffmpeg. If
    ffmpeg is unavailable, the row-building itself still succeeds (CSV join, path resolution,
    seeded sampling) and only the wav materialization raises ``NeedsExtraction`` -- so callers
    can inspect ``e.args`` / catch it per-row rather than the whole loader failing blind.
    """
    import shutil

    import numpy as np

    if split != "test":
        raise NotImplementedError(
            f"meld split={split!r} not wired up -- only 'test' (output_repeated_splits_test/) "
            "is implemented for B2-paralinguistic; train/dev CSVs+dirs exist on disk but are "
            "out of scope here."
        )

    root = datasets_dir() / "meld" / "MELD.Raw"
    csv_path = root / "test_sent_emo.csv"
    video_dir = root / "output_repeated_splits_test"
    if not csv_path.exists():
        raise FileNotFoundError(f"meld test_sent_emo.csv not found: {csv_path}")

    all_rows = _read_test_csv(csv_path)
    # sorted() by (Dialogue_ID, Utterance_ID) as ints for a stable, file-order-independent
    # enumeration before sampling (README determinism rule).
    all_rows.sort(key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"])))

    rng = np.random.default_rng(seed)
    if n is not None and n < len(all_rows):
        idx = rng.choice(len(all_rows), size=n, replace=False)
        idx.sort()
        all_rows = [all_rows[int(i)] for i in idx]

    cache_dir = wav_cache_dir("meld")
    prepared = []
    for r in all_rows:
        dia, utt = r["Dialogue_ID"], r["Utterance_ID"]
        mp4_name = f"dia{dia}_utt{utt}.mp4"
        mp4_path = video_dir / mp4_name
        wav_path = cache_dir / f"{mp4_name[:-4]}.wav"
        item_id = f"meld/test/{mp4_name[:-4]}"
        gold = {
            "emotion": r["Emotion"],
            "sentiment": r["Sentiment"],
            "speaker": r["Speaker"],
            "utterance": r["Utterance"],
        }
        meta = {
            "dataset": "meld",
            "item_id": item_id,
            "split": split,
            "dialogue_id": int(dia),
            "utterance_id": int(utt),
            "mp4_path": str(mp4_path),
        }
        if not mp4_path.exists():
            raise FileNotFoundError(f"meld video missing: {mp4_path}")
        prepared.append((mp4_path, wav_path, item_id, gold, meta))

    # Check ffmpeg ONCE up front so a missing tool reports the full batch of needed commands
    # (all rows this call would have extracted) rather than failing opaquely on row 1.
    if shutil.which("ffmpeg") is None and any(not w.exists() for _, w, *_ in prepared):
        missing_cmds = "\n".join(
            "  " + " ".join(_ffmpeg_cmd(m, w)) for m, w, *_ in prepared if not w.exists()
        )
        raise NeedsExtraction(
            "ffmpeg not found on PATH -- meld loader needs it to decode .mp4 -> .wav for "
            f"{sum(1 for _, w, *_ in prepared if not w.exists())} row(s). No sudo / package "
            "install in this env (CLAUDE.md). Row-building (CSV join + seeded sampling) "
            "succeeded; once ffmpeg is available, run the following (or just re-call "
            f"load_meld -- extraction+caching happens automatically per row):\n{missing_cmds}"
        )

    rows = []
    sampled_ids = []
    for mp4_path, wav_path, item_id, gold, meta in prepared:
        wav = _extract_wav(mp4_path, wav_path)
        rows.append({"wav": wav, "gold": gold, "meta": meta})
        sampled_ids.append(item_id)

    freeze_ids("meld", sampled_ids, dataset="meld", seed=seed, extra={"split": split})
    return rows

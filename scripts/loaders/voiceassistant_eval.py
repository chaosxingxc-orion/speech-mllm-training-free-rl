"""voiceassistant_eval — VoiceAssistant-Eval (MIT) re-examined include-subset loaders
(task B5-reexamined).

Recipe: ``wiki/survey/2026-07-09-datasets-lock-second14.md``'s "voiceassistant-eval" section --
wide schema (``user_audio_0..6`` / ``ref_answers`` / ``image_0..4``), re-examined to an
include-subset verdict: ``listening/{general,music,sound,speech}`` (single-audio-question +
short ``ref_answers``) + ``speaking/reasoning`` (has ``ref_answers``, single-turn). Excluded:
``viewing/multi_discipline`` (needs ``image_*`` -- structurally out of scope for an audio-only
pipeline) and ``speaking/{emotion,roleplay,safety,robustness,multi_round,assistant,
instruction_following}`` (open/judged or empty-``ref_answers`` or genuinely multi-turn).

On-disk layout: ``datasets/voiceassistant-eval/{listening,speaking}/test_<category1>_<category2>
.parquet`` (one file per subset, already split by category -- no merging needed, unlike
VoiceBench's per-domain/per-dialect files).

Row shape (shared across all 5 loaders in this module):
    wav  -> materialized ``user_audio_0`` (the single-turn spoken question/prompt)
    gold -> ``ref_answers[0]`` (per the recipe) -- ``meta["ref_answers"]`` keeps the FULL list
            in case a row has more than one acceptable phrasing (a caller doing exact/contains
            scoring against only ``[0]`` may want the rest)
    meta -> {"dataset", "item_id", "split", "category1", "category2", "raw_question",
             "raw_options", "user_audio_transcripts", "ref_answers",
             ["sound_audio_wav", "role_audio_wav"] -- see FIDELITY GAP note below,
             ["system_prompt"] if non-empty}

Single-turn filter (per recipe): every row with non-empty ``user_audio_1`` bytes is a multi-turn
item and is dropped BEFORE sampling (never counted toward ``n``) -- ``user_audio_1..6`` are all
``bytes`` columns that are ``b""`` (empty, not ``NULL``) when unused, confirmed empirically
2026-07-09 (``pyarrow`` reads the struct column as always-present zero-length bytes for unused
turns, not a null).

**FIDELITY GAP, documented per README's "adapt and document" rule (2026-07-09)**: several
``listening/*`` rows carry a THIRD audio channel, ``sound_audio`` (the actual sound/music clip
being asked about), separate from ``user_audio_0`` (the short spoken QUESTION about that clip)
-- e.g. a sampled ``listening/general`` row has ``user_audio_0`` = "Which rhythm has been added
to the music?" (~2s) while ``sound_audio`` (~20s) is the music clip itself; the gold answer
("Hand claps on every count") is NOT recoverable from ``user_audio_0`` alone. The recipe's
"audio = user_audio_0" is followed literally for the PRIMARY ``wav`` field (interface contract:
one wav path per Row), but this loader does NOT silently drop ``sound_audio`` / ``role_audio``
when present -- it materializes them too and exposes the paths as EXTRA meta fields
(``sound_audio_wav`` / ``role_audio_wav``, present only when the source bytes are non-empty) so
a caller building an actually-answerable multi-audio prompt has what it needs. A caller that
only reads ``wav`` will get an under-specified query for those rows -- this is a known
limitation of the single-``wav``-field Row contract for this dataset's wide-schema rows, not a
bug in this loader; report/flag before drawing conclusions from listening/{general,music,sound}
accuracy that ignores ``sound_audio_wav``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402

_ID_SUFFIX_RE = re.compile(r"^(.*_)(\d+)$")


def _id_sort_key(item_id: str):
    """Numeric-aware sort key for ids like "Listening_General_0"/"...12" (avoid lexicographic
    "10" < "2"), falling back to plain string sort if the id doesn't end in "_<digits>"."""
    m = _ID_SUFFIX_RE.match(item_id)
    return (m.group(1), int(m.group(2))) if m else (item_id, -1)


def _load_voiceassistant(
    rel_path: str,
    name: str,
    split: str = "test",
    n: int | None = None,
    seed: int = SLICE_SEED,
) -> list:
    """Shared loader body for every voiceassistant-eval subset in this module.

    ``rel_path`` is the parquet path relative to ``datasets/voiceassistant-eval/`` (e.g.
    ``"listening/test_listening_general.parquet"``); ``name`` is the registry key, reused as the
    ``freeze_ids`` experiment name and ``meta["dataset"]``.
    """
    import numpy as np
    import pyarrow.parquet as pq

    if split != "test":
        raise NotImplementedError(
            f"{name}: split={split!r} not wired up -- voiceassistant-eval ships a single "
            "test_*.parquet per subset, no train/dev split."
        )

    f = datasets_dir() / "voiceassistant-eval" / rel_path
    if not f.exists():
        raise FileNotFoundError(f"{name}: parquet not found at {f}")

    cols = ["id", "category1", "category2", "system_prompt", "role_audio", "sound_audio",
            "user_audio_0", "user_audio_1", "ref_answers", "extra"]
    all_rows = pq.read_table(f, columns=cols).to_pylist()
    # single-turn filter BEFORE sampling (never counts toward n) -- see module docstring
    rows = [r for r in all_rows if not r["user_audio_1"]]
    rows.sort(key=lambda r: _id_sort_key(r["id"]))  # stable enumeration order (contract step 2)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows)) if n is not None else np.arange(len(rows))
    idx = idx[:n] if n is not None else idx

    cache_dir = wav_cache_dir("voiceassistant-eval")
    out, ids = [], []
    for i in idx:
        r = rows[int(i)]
        item_id = r["id"]
        extra = r.get("extra") or {}
        ref_answers = list(r["ref_answers"]) if r["ref_answers"] else []
        meta = {
            "dataset": name, "item_id": item_id, "split": split,
            "category1": r["category1"], "category2": r["category2"],
            "raw_question": extra.get("raw_question"), "raw_options": list(extra.get("raw_options") or []),
            "user_audio_transcripts": list(extra.get("user_audio_transcripts") or []),
            "ref_answers": ref_answers,
        }
        if r["system_prompt"]:
            meta["system_prompt"] = r["system_prompt"]
        if r["sound_audio"]:
            meta["sound_audio_wav"] = wav_from_bytes(r["sound_audio"], f"{item_id}_sound", cache_dir)
        if r["role_audio"]:
            meta["role_audio_wav"] = wav_from_bytes(r["role_audio"], f"{item_id}_role", cache_dir)

        wav = wav_from_bytes(r["user_audio_0"], f"{item_id}_q", cache_dir)
        gold = ref_answers[0] if ref_answers else None
        out.append({"wav": wav, "gold": gold, "meta": meta})
        ids.append(item_id)

    freeze_ids(name, ids, dataset=f"voiceassistant-eval/{rel_path}", seed=seed,
               extra={"split": split, "n_requested": n})
    return out


def load_voiceassistant_listening_general(split: str = "test", n: int | None = None,
                                           seed: int = SLICE_SEED) -> list:
    return _load_voiceassistant("listening/test_listening_general.parquet",
                                 "voiceassistant-listening-general", split, n, seed)


def load_voiceassistant_listening_music(split: str = "test", n: int | None = None,
                                         seed: int = SLICE_SEED) -> list:
    return _load_voiceassistant("listening/test_listening_music.parquet",
                                 "voiceassistant-listening-music", split, n, seed)


def load_voiceassistant_listening_sound(split: str = "test", n: int | None = None,
                                         seed: int = SLICE_SEED) -> list:
    return _load_voiceassistant("listening/test_listening_sound.parquet",
                                 "voiceassistant-listening-sound", split, n, seed)


def load_voiceassistant_listening_speech(split: str = "test", n: int | None = None,
                                          seed: int = SLICE_SEED) -> list:
    """listening/speech: single-audio speaker-attribute probes (e.g. gender) -- ``ref_answers``
    is a short discrete label (e.g. "Man"/"Woman"), not free text; a caller may treat this as a
    classification target."""
    return _load_voiceassistant("listening/test_listening_speech.parquet",
                                 "voiceassistant-listening-speech", split, n, seed)


def load_voiceassistant_speaking_reasoning(split: str = "test", n: int | None = None,
                                            seed: int = SLICE_SEED) -> list:
    return _load_voiceassistant("speaking/test_speaking_reasoning.parquet",
                                 "voiceassistant-speaking-reasoning", split, n, seed)


_ALL = {
    "voiceassistant-listening-general": load_voiceassistant_listening_general,
    "voiceassistant-listening-music": load_voiceassistant_listening_music,
    "voiceassistant-listening-sound": load_voiceassistant_listening_sound,
    "voiceassistant-listening-speech": load_voiceassistant_listening_speech,
    "voiceassistant-speaking-reasoning": load_voiceassistant_speaking_reasoning,
}


if __name__ == "__main__":
    for name, fn in _ALL.items():
        try:
            rows = fn(n=3)
            keys = sorted(rows[0].keys()) if rows else []
            print(f"{name}: n={len(rows)} keys={keys} gold0={repr(rows[0]['gold'])[:80] if rows else None}")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")

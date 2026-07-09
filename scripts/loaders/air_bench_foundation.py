"""air_bench_foundation — AIR-Bench (24.09) Foundation-track per-task loaders (task B5-reexamined).

Recipe: ``wiki/survey/2026-07-09-datasets-lock-second14.md``'s "air-bench" section -- Foundation
track MCQ only (upstream Apache license; per-clip audio license varies by source corpus), Chat
track excluded (GPT-4-judged, no discrete reference). ``Speech_Grounding`` is called out as the
one speech-NATIVE Foundation task (everything else is a sound/music perception probe) and is
prioritized here, though see the BLOCKER below -- its audio is not actually on disk on this box.

On-disk layout: ``datasets/air-bench/Foundation/Foundation_meta.json`` is ONE flat list of
24683 rows across 25 (task_name, dataset_name) pairs (19 distinct ``task_name`` values, 4 of
which have 2 upstream ``dataset_name`` sources each -- e.g. Music_Genre_Recognition draws from
BOTH ``MTJ-Jamendo`` and ``fma``). Each row: ``{path, question, choice_a..choice_d (2-4 present,
not always 4), answer_gt, task_name, dataset_name, uniq_id, [task-specific extra key, e.g.
Audio_Grounding's answer_gt_word]}``. Audio for a given (task_name, dataset_name) pair lives
under ``Foundation/<task_name>_<dataset_name>/<path>`` (verified: this is exact string
concatenation for every present pair, no other separator) -- confirmed NOT parquet-embedded,
unlike VoiceBench/voiceassistant-eval; ``path`` usually resolves directly to a standalone file on
disk, so no ``wav_from_bytes`` materialization step is needed (mirrors ``cn_celeb1.py``'s
precedent of handing back non-``.wav`` paths as-is in the ``wav`` field).

**Metadata path is NOT always trustworthy verbatim (found + fixed 2026-07-09)**: two independent
issues, both handled by resolving each row's audio by FILENAME STEM against an on-disk directory
index rather than joining ``task_dir / row["path"]`` literally:
  1. **Extension mismatch**: ``Audio_Grounding/AudioGrounding``'s metadata records every ``path``
     with a ``.wav`` suffix, but every file actually on disk is ``.flac`` (same stem) -- a literal
     join resolves 0/896 rows; stem-matching resolves 896/896.
  2. **Genuine partial download + many-rows-per-clip**: ``Sound_AQA/clothoaqa`` has 1000 metadata
     rows (multiple QA pairs can share one audio clip, same as ``Music_AQA/music_avqa`` -- 814
     rows over 797 clips resolves fully) but only 351 of the underlying clips are on disk, so
     stem-matching still leaves a genuine 400/1000 gap. Unresolvable rows are dropped BEFORE
     sampling (never counted toward ``n``, and never silently returned as a dangling path) --
     verified empirically: naive literal-path smoke-testing before this fix returned
     ``wav_exists=False`` for a sampled Audio_Grounding row and would have for ~40% of
     Sound_AQA/clothoaqa draws.

**BLOCKER, documented 2026-07-09 -- PARTIAL download, 13/25 (task_name, dataset_name) pairs
present on disk**: ``Foundation_meta.json`` references audio for all 25 pairs, but only 13 of
the corresponding ``Foundation/<task>_<dataset>/`` directories actually exist under
``SPEECHRL_DATA_DIR`` (verified via directory listing, 2026-07-09) -- including
``Speech_Grounding_librispeech``, the task this loader is asked to prioritize. The missing 12
pairs (``Speaker_Age_Prediction/common_voice_13.0_en``, ``Speaker_Emotion_Recontion/{iemocap,
meld}``, ``Speaker_Gender_Recognition/{common_voice_13_en,meld}``,
``Speaker_Intent_Classification/slurp``, ``Speaker_Number_Verification/voxceleb1``,
``Speech_Entity_Reconition/slurp``, ``Speech_Grounding/librispeech``,
``Spoken_Language_Identification/covost2``, ``Synthesized_Voice_Detection/fake_or_real``,
``vocal_sound_classification/VocalSound``) all have a live ``._____temp/Foundation/`` sibling
tree missing the same directories too -- so this is not "still downloading", it looks like the
fetch never reached them. This loader implements the FULL shape for every pair (metadata read,
path resolution, sampling, freeze_ids) and raises ``NeedsAirBenchFoundationAudio`` -- naming the
exact missing directory and the ``hf-mirror``/ModelScope re-fetch this needs -- only at the
point where it would touch/return an actually-missing audio file, per the loader-contract
"implement the shape, raise a specific NeedsX error" instruction. ``load_air_bench_speech_grounding``
is registered FIRST (per the task) and WILL raise this error until the gap is re-fetched.

Row shape (uniform across every task):
    wav  -> ``Foundation/<task_name>_<dataset_name>/<path>`` (a .flac path, no conversion)
    gold -> {"answer_gt": str, "choices": {"A": ..., "B": ..., ...} (only the populated
             choice_a.. letters), "task_name": str} -- plus "extra" (dict) for any task-specific
             field beyond the standard columns (e.g. Audio_Grounding's answer_gt_word), only
             when present, so no information is silently dropped
    meta -> {"dataset", "item_id" (f"{task_name}:{dataset_name}:{uniq_id}"), "split",
             "task_name", "dataset_name", "question"}
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402

_CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]  # generous upper bound; only populated ones kept
_STANDARD_KEYS = {"path", "question", "answer_gt", "task_name", "dataset_name", "uniq_id"} | {
    f"choice_{c.lower()}" for c in _CHOICE_LETTERS
}


class NeedsAirBenchFoundationAudio(RuntimeError):
    """Raised when a (task_name, dataset_name) pair's audio directory is not on disk.

    See module-docstring BLOCKER: 12/25 pairs (incl. Speech_Grounding) are metadata-only on this
    box as of 2026-07-09 -- re-fetch ``air-bench`` (ModelScope/hf-mirror) to resolve.
    """


def _foundation_dir() -> Path:
    return datasets_dir() / "air-bench" / "Foundation"


def _on_disk_stem_index(task_dir: Path) -> dict[str, str]:
    """{filename stem -> actual filename} for every file directly under ``task_dir``.

    One directory listing (O(n_files)), reused to resolve every row's ``path`` by stem instead
    of joining it literally -- see module docstring's "Metadata path is NOT always trustworthy"
    note (handles both the Audio_Grounding extension-mismatch and generic partial-download gaps
    uniformly).
    """
    return {p.stem: p.name for p in task_dir.iterdir() if p.is_file()}


def _load_meta() -> list[dict]:
    import json

    p = _foundation_dir() / "Foundation_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"air-bench Foundation_meta.json not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def available_tasks() -> list[tuple[str, str]]:
    """(task_name, dataset_name) pairs whose audio directory actually exists on disk."""
    root = _foundation_dir()
    pairs = sorted({(r["task_name"], r["dataset_name"]) for r in _load_meta()})
    return [(t, d) for t, d in pairs if (root / f"{t}_{d}").is_dir()]


def missing_tasks() -> list[tuple[str, str]]:
    """(task_name, dataset_name) pairs referenced in metadata but with no on-disk audio dir."""
    root = _foundation_dir()
    pairs = sorted({(r["task_name"], r["dataset_name"]) for r in _load_meta()})
    return [(t, d) for t, d in pairs if not (root / f"{t}_{d}").is_dir()]


def _load_foundation_task(
    task_name: str,
    dataset_name: str,
    split: str = "test",
    n: int | None = None,
    seed: int = SLICE_SEED,
) -> list:
    """Shared loader body for every AIR-Bench Foundation (task_name, dataset_name) pair.

    Raises ``NeedsAirBenchFoundationAudio`` if the pair's audio directory is missing (see module
    docstring BLOCKER) -- metadata filtering/sampling still runs first so the error message can
    report exactly how many rows were affected.
    """
    import numpy as np

    if split != "test":
        raise NotImplementedError(
            f"air-bench-foundation {task_name}/{dataset_name}: split={split!r} not wired up -- "
            "Foundation is a single eval pool, no train/dev split."
        )

    rows = [r for r in _load_meta() if r["task_name"] == task_name and r["dataset_name"] == dataset_name]
    if not rows:
        raise ValueError(f"air-bench Foundation: no metadata rows for ({task_name!r}, {dataset_name!r})")
    rows.sort(key=lambda r: r["uniq_id"])  # stable enumeration order before sampling (contract step 2)

    task_dir = _foundation_dir() / f"{task_name}_{dataset_name}"
    registry_name = f"air-bench-foundation-{task_name}-{dataset_name}".lower().replace("_", "-")

    if not task_dir.is_dir():
        raise NeedsAirBenchFoundationAudio(
            f"{registry_name}: audio dir missing -- {task_dir} does not exist ({len(rows)} "
            "metadata row(s) for this pair, 0 resolvable). Re-fetch air-bench Foundation "
            "(ModelScope/hf-mirror, see docs/datasets.lock.json) to materialize "
            f"{task_name}_{dataset_name}/ before this loader can return rows."
        )

    # Resolve every row's audio by filename STEM against the on-disk directory index, not by
    # joining `row["path"]` literally -- see module docstring's "Metadata path is NOT always
    # trustworthy" note. Rows that don't resolve are dropped HERE, before sampling, so `n`
    # draws from the genuinely-available pool (never a dangling path).
    on_disk = _on_disk_stem_index(task_dir)
    resolved = [(r, task_dir / on_disk[Path(r["path"]).stem]) for r in rows
                if Path(r["path"]).stem in on_disk]
    dropped = len(rows) - len(resolved)
    if dropped:
        print(f"[air_bench_foundation] {registry_name}: {dropped}/{len(rows)} metadata row(s) have "
              "no resolvable on-disk audio file -- dropped before sampling (see module docstring).",
              flush=True)
    if not resolved:
        raise NeedsAirBenchFoundationAudio(
            f"{registry_name}: audio dir {task_dir} exists but 0/{len(rows)} metadata row(s) "
            "resolve to an on-disk file by stem -- re-fetch needed."
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(resolved)) if n is not None else np.arange(len(resolved))
    idx = idx[:n] if n is not None else idx

    out, ids = [], []
    for i in idx:
        r, wav_path = resolved[int(i)]
        choices = {c: r[f"choice_{c.lower()}"] for c in _CHOICE_LETTERS
                   if r.get(f"choice_{c.lower()}") is not None}
        extra = {k: v for k, v in r.items() if k not in _STANDARD_KEYS}
        gold = {"answer_gt": r["answer_gt"], "choices": choices, "task_name": task_name}
        if extra:
            gold["extra"] = extra
        item_id = f"{task_name}:{dataset_name}:{r['uniq_id']}"
        meta = {"dataset": registry_name, "item_id": item_id, "split": split,
                "task_name": task_name, "dataset_name": dataset_name, "question": r["question"]}
        out.append({"wav": str(wav_path), "gold": gold, "meta": meta})
        ids.append(item_id)

    freeze_ids(registry_name, ids, dataset=f"air-bench/Foundation/{task_name}_{dataset_name}",
               seed=seed, extra={"split": split, "n_requested": n})
    return out


def get_loader(task_name: str, dataset_name: str):
    """Factory: a zero-extra-arg ``load_<name>(split, n, seed)`` callable for any (task_name,
    dataset_name) pair present in ``Foundation_meta.json`` -- not just the 14 hand-registered
    below. Use this to add more per-task accessors without re-deriving the schema."""
    return partial(_load_foundation_task, task_name, dataset_name)


# ---- per-task accessors -- Speech_Grounding FIRST per the task, though it currently raises
# NeedsAirBenchFoundationAudio (see BLOCKER above); the other 13 are the on-disk-complete pairs.
load_air_bench_speech_grounding = get_loader("Speech_Grounding", "librispeech")

load_air_bench_acoustic_scene_cochlscene = get_loader("Acoustic_Scene_Classification", "CochlScene")
load_air_bench_acoustic_scene_tut2017 = get_loader("Acoustic_Scene_Classification", "TUT2017")
load_air_bench_audio_grounding = get_loader("Audio_Grounding", "AudioGrounding")
load_air_bench_music_aqa = get_loader("Music_AQA", "music_avqa")
load_air_bench_music_genre_mtj_jamendo = get_loader("Music_Genre_Recognition", "MTJ-Jamendo")
load_air_bench_music_genre_fma = get_loader("Music_Genre_Recognition", "fma")
load_air_bench_music_instruments_mtj_jamendo = get_loader("Music_Instruments_Classfication", "MTJ-Jamendo")
load_air_bench_music_instruments_nsynth = get_loader("Music_Instruments_Classfication", "nsynth")
load_air_bench_music_midi_pitch_nsynth = get_loader("Music_Midi_Pitch_Analysis", "nsynth")
load_air_bench_music_midi_velocity_nsynth = get_loader("Music_Midi_Velocity_Analysis", "nsynth")
load_air_bench_music_mood_mtj_jamendo = get_loader("Music_Mood_Recognition", "MTJ-Jamendo")
load_air_bench_sound_aqa_avqa = get_loader("Sound_AQA", "avqa")
load_air_bench_sound_aqa_clothoaqa = get_loader("Sound_AQA", "clothoaqa")


_ALL = {
    "air-bench-foundation-speech-grounding": load_air_bench_speech_grounding,  # first, per the task
    "air-bench-foundation-acoustic-scene-cochlscene": load_air_bench_acoustic_scene_cochlscene,
    "air-bench-foundation-acoustic-scene-tut2017": load_air_bench_acoustic_scene_tut2017,
    "air-bench-foundation-audio-grounding": load_air_bench_audio_grounding,
    "air-bench-foundation-music-aqa": load_air_bench_music_aqa,
    "air-bench-foundation-music-genre-mtj-jamendo": load_air_bench_music_genre_mtj_jamendo,
    "air-bench-foundation-music-genre-fma": load_air_bench_music_genre_fma,
    "air-bench-foundation-music-instruments-mtj-jamendo": load_air_bench_music_instruments_mtj_jamendo,
    "air-bench-foundation-music-instruments-nsynth": load_air_bench_music_instruments_nsynth,
    "air-bench-foundation-music-midi-pitch-nsynth": load_air_bench_music_midi_pitch_nsynth,
    "air-bench-foundation-music-midi-velocity-nsynth": load_air_bench_music_midi_velocity_nsynth,
    "air-bench-foundation-music-mood-mtj-jamendo": load_air_bench_music_mood_mtj_jamendo,
    "air-bench-foundation-sound-aqa-avqa": load_air_bench_sound_aqa_avqa,
    "air-bench-foundation-sound-aqa-clothoaqa": load_air_bench_sound_aqa_clothoaqa,
}


if __name__ == "__main__":
    print("missing (task_name, dataset_name) pairs:", missing_tasks())
    for name, fn in _ALL.items():
        try:
            rows = fn(n=3)
            keys = sorted(rows[0].keys()) if rows else []
            print(f"{name}: n={len(rows)} keys={keys} gold0={repr(rows[0]['gold'])[:100] if rows else None}")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")

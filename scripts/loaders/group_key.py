"""scripts/loaders/group_key.py — group_key_of(dataset_key, item) dispatch (ticket #26,
QUESTION-INDEPENDENT machinery only).

Design doc: wiki/2026-07-11-group-split-statistics-design.md §1.3 (per-dataset inventory) + §2.5
(dispatch sketch). This is the single place the design's G-FIELD/G-ID classifications become code.
Status (2026-07-11 update, ticket #26 implementation): ``_common.draw_disjoint_grouped`` and
``locked_split.py`` now call this for real (the group-disjoint LOCKED_HOLDOUT manifest
generation); ``run_baseline._run_item`` also persists ``group_id`` per_item. The owner-gated
GROUP-DISJOINT RERUN itself (actually regenerating scored cells) is still a separate step
(design doc §5) from manifest generation.

Contract: ``group_key_of(dataset_key, item) -> str | None``.
  - ``item`` is a loader ``Row`` (``{"wav", "gold", "meta"}``, see scripts/loaders/README.md) — a
    REAL loaded row, or a synthetic test double shaped the same way (only ``gold``/``meta`` are
    ever read; ``wav`` is never touched).
  - Returns a group id (str) for every G-FIELD / G-ID dataset in the design doc's inventory
    (~30 keys, §4.1: "covered with zero loader edits") PLUS the 4 formerly-G-SOURCE dataset
    families whose loader meta-field addition landed this session (speech-massive ``-attr``'s
    speaker_id, the 3 air-bench-foundation ``*_AQA`` tasks' clip_id, audio2tool's query_idx,
    mmau-mini's audio_id — see each branch below).
  - Returns ``None`` for the STILL-unresolved G-SOURCE datasets (``G_SOURCE_DATASETS`` below —
    the grouping field either doesn't exist in this corpus mirror, or is too coarse to use; see
    the legacy branch's comment for the per-key evidence).
  - Returns ``None`` for G-NONE datasets (no grouping unit exists at any grain finer than the
    whole dataset — design doc §1.4) and for anything not explicitly classified below.

Callers MUST treat ``None`` as "no group id for this item", never silently upgrade it to a
fabricated group — ``_common.draw_disjoint_grouped`` folds a ``None`` into a SINGLETON group keyed
by the item's own ``item_id`` (item-level fallback, honestly flagged by that function's
``fallback_item_level`` return key when EVERY item in a dataset resolves to ``None``).
"""
from __future__ import annotations

import hashlib


def _stable_hash(text: str, length: int = 16) -> str:
    """Deterministic (NOT ``PYTHONHASHSEED``-randomized, unlike Python's builtin ``hash()``) hash
    of ``text`` — used to derive a group id from a free-text field (e.g. heysquad's SQuAD
    ``context`` passage: design doc §1.3 "Hash `context` → group, or add explicit `passage_id`").
    sha1 truncated to ``length`` hex chars — collision risk is irrelevant here (two items' group
    ids only need to be equal IFF their source text is byte-identical, not globally unique across
    all possible strings).
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


# ---- G-NONE datasets (design doc §1.4): no grouping unit exists at any grain finer than the
# whole dataset. Listed here for documentation/test purposes ONLY — group_key_of returns None for
# these via the default fall-through at the bottom, no per-key branch needed. Includes the
# air-bench-foundation CLASSIFICATION subsets (clip == item, 1:1) where item-level bootstrap IS
# already clip-level (no caveat needed beyond noting group≡item, per that section).
G_NONE_DATASETS = frozenset({
    "seed-tts-eval-en", "seed-tts-eval-zh", "fleurs-r",
    "uro-bench-Repeat", "uro-bench-Repeat-zh",
    "uro-bench-UnderEmotion-en", "uro-bench-UnderEmotion-zh",
    "uro-bench-SQuAD-zh", "uro-bench-OpenbookQA-zh", "uro-bench-Gsm8kEval", "uro-bench-GaokaoEval",
    "uro-bench-HSK5-zh", "uro-bench-APE-zh", "uro-bench-MuChoEval-en", "uro-bench-MLC",
    "uro-bench-MLC-zh", "uro-bench-MLCpro-en", "uro-bench-MLCpro-zh", "uro-bench-TruthfulEval",
    "voicebench-openbookqa", "voicebench-advbench",
    "audiocaps-qa", "mmar", "vocalbench-emotion",
    "air-bench-foundation-acoustic-scene-cochlscene", "air-bench-foundation-acoustic-scene-tut2017",
    "air-bench-foundation-music-genre-mtj-jamendo", "air-bench-foundation-music-genre-fma",
    "air-bench-foundation-music-instruments-mtj-jamendo", "air-bench-foundation-music-instruments-nsynth",
    "air-bench-foundation-music-midi-pitch-nsynth", "air-bench-foundation-music-midi-velocity-nsynth",
    "air-bench-foundation-music-mood-mtj-jamendo", "air-bench-foundation-audio-grounding",
    "air-bench-foundation-speech-grounding",  # env-blocked, never produced a real cell
})

# ---- G-SOURCE datasets that remain UNRESOLVED (design doc §1.3/§4.1): the grouping field either
# does not exist in this corpus mirror, or exists but is too coarse to use under
# ``draw_disjoint_grouped``'s whole-group-never-split rule (design doc §2.2 step 5's
# "oversized_group" case) -- see group_key_of's legacy branch below and p2_baselines.py's LOADERS
# comment for the per-key evidence. 2026-07-11 (ticket #26): six of the original thirteen G-SOURCE
# keys moved OUT of this set into real dispatch below once their loader meta-field landed
# (speech-massive-*-attr's speaker_id, the 3 air-bench-foundation *_AQA tasks' clip_id,
# audio2tool's query_idx, mmau-mini's audio_id) -- this set is now exactly what's left unresolved.
G_SOURCE_DATASETS = frozenset({
    "SQuAD-zh", "spoken-squad", "OpenbookQA-zh", "vocalbench-zh",
    "big-bench-audio", "minds14-zh",
})


def group_key_of(dataset_key: str, item: dict) -> str | None:
    """dataset_key x item -> group id, or None. See module docstring for the full contract."""
    meta = item.get("meta") or {}
    gold = item.get("gold")
    iid = meta.get("item_id")

    # ---- G-ID: parse the existing item_id string (works even on ARCHIVED per_item cells, once
    # group_id starts being persisted going forward -- see run_baseline._run_item) ----
    if dataset_key == "aishell-1":
        return iid[6:11] if iid else None                          # 'BAC009S0769W0185' -> 'S0769'
    if dataset_key == "thchs-30":
        return iid.split("_")[0] if iid else None                   # 'D7_841' -> 'D7'
    if dataset_key == "librispeech":
        return iid.split("-")[0] if iid else None                   # '6930-75918-0000' -> '6930'
    if dataset_key == "voicebench-bbh":
        return iid.rsplit("_", 1)[0] if iid else None                # 'bbh_navigate_147' -> 'bbh_navigate'
    if dataset_key == "voicebench-sd-qa":
        return iid.split("#")[-1] if iid and "#" in iid else None    # 'sd-qa/usa#37' -> '37' (same Q, all dialects)
    if dataset_key == "squtr":
        return iid.split("|")[-1] if iid and "|" in iid else None    # 'fiqa|clean|q19' -> 'q19' (noise-augmented copies)

    # ---- G-FIELD: group unit already lives in meta/gold (zero loader edits) ----
    if dataset_key == "crema-d":
        return gold.get("spk") if isinstance(gold, dict) else None
    if dataset_key == "esd":
        return gold.get("spk") if isinstance(gold, dict) else None
    if dataset_key == "csemotions":
        return gold.get("speaker") if isinstance(gold, dict) else None
    if dataset_key == "meld":
        return str(meta["dialogue_id"]) if "dialogue_id" in meta else None
    if dataset_key == "voicebench-mmsu-spoken":
        return meta.get("src") or meta.get("domain")
    if dataset_key == "mmsu":
        return meta.get("task_name")
    if dataset_key.startswith("vocalbench-") and dataset_key != "vocalbench-emotion":
        # knowledge -> topic/source; reasoning -> category/source; multi-round -> category (no
        # source/topic column for that axis -- see vocalbench.py's per-axis meta_cols).
        return meta.get("source") or meta.get("topic") or meta.get("category")
    if dataset_key.startswith("voiceassistant-"):
        if "category1" in meta and "category2" in meta:
            return f'{meta["category1"]}/{meta["category2"]}'
        return None
    if dataset_key in ("slurp", "slurp-slot"):
        return gold.get("scenario") if isinstance(gold, dict) else None
    if dataset_key.endswith("-attr"):
        # K5 speaker-ATTRIBUTE probe: the label IS speaker_sex/speaker_age, so the split MUST be
        # speaker-disjoint, never scenario-disjoint (scenario grouping would still let the SAME
        # speaker's sex/age answer appear on both dev and test, via a different scenario/utterance
        # from the same speaker) -- design doc §1.3 K5 row. 2026-07-11 (ticket #26): resolved --
        # speech_massive.py's _COLUMNS now exposes speaker_id (design doc §4.1 table's loader
        # edit landed), so this is real speaker-disjoint grouping, not a guess.
        return meta.get("speaker_id")
    if dataset_key.startswith("speech-massive"):
        # K6/K7 (intent/slot) scenario grouping -- already resolvable without a loader edit (design
        # doc §4.1: "scenario grouping already works without this [speaker_id add]").
        return meta.get("scenario_str")
    if dataset_key == "heysquad":
        # SQuAD passage grouping via a stable hash of the (leakage-flagged, see heysquad.py) full
        # `context` field -- design doc §1.3: "Hash `context` → group ... resolvable without a
        # loader edit". This reads `context` ONLY to derive an opaque group id, never injects it
        # into a prompt (the T7 leakage concern is about generation-time KB use, unrelated here).
        context = meta.get("context")
        return _stable_hash(context) if context else None

    # ---- G-FIELD but MOOT (score is always None at Step-1 -- kept for completeness/future use) --
    if dataset_key == "voicebench-ifeval":
        ids = meta.get("instruction_id_list")
        return _stable_hash(",".join(sorted(ids))) if ids else None

    # ---- resolved G-SOURCE (2026-07-11, ticket #26): loader meta-field additions landed this
    # session -- real dispatch, no longer a TODO. ----
    if dataset_key in ("air-bench-foundation-sound-aqa-avqa", "air-bench-foundation-sound-aqa-clothoaqa",
                       "air-bench-foundation-music-aqa"):
        # air_bench_foundation.py now sets meta["clip_id"] = Path(row["path"]).stem (design doc
        # §4.1 table, landed) -- these 3 AQA tasks have multiple QA pairs sharing one audio clip.
        # The other air-bench-foundation-* CLASSIFICATION subsets need no such fix (clip == item
        # already, see G_NONE_DATASETS above); they also get clip_id populated but it's a no-op
        # there since it always equals a singleton.
        return meta.get("clip_id")
    if dataset_key == "audio2tool":
        # audio2tool.py now sets meta["query_idx"] (design doc §1.3 K10 row / §4.1 table, landed)
        # -- one query rendered by MANY speakers shares a query_idx. Stringified for a stable
        # group-id type (source jsonl's query_idx is an int).
        qi = meta.get("query_idx")
        return str(qi) if qi is not None else None
    if dataset_key == "mmau-mini":
        # scripts/p2_baselines.py's load_mmau now sets it["group_key"] = the upstream MMAU
        # audio_id (design doc §4.1 table, landed) -- carried through run_baseline._legacy_rows
        # into meta["group_key"]. On the CURRENT test_mini mirror this is empirically 1:1 with
        # items (no clip is shared by >1 question here -- verified this session), so this
        # currently degenerates to item-level in practice, same as the air-bench classification
        # "clip == item" case -- but it's a REAL field, not a guess, and future-proofs a fuller
        # MMAU pool where clips do repeat.
        return meta.get("group_key")
    if dataset_key in ("SQuAD-zh", "spoken-squad", "OpenbookQA-zh", "vocalbench-zh",
                       "big-bench-audio", "minds14-zh"):
        # These 6 remaining scripts/p2_baselines.py-native LEGACY_DATASETS (positional item_ids --
        # run_baseline._legacy_rows synthesizes "{dataset_key}#{i}") were checked against the
        # on-disk schema this session (ticket #26) and deliberately left G-NONE -- see
        # p2_baselines.py's LOADERS comment for the per-key evidence (either no source-family
        # field exists in this corpus mirror at all, e.g. SQuAD-zh/OpenbookQA-zh/spoken-squad/
        # minds14-zh, or the only available field -- big-bench-audio's "category", vocalbench-zh's
        # "Source" -- is too coarse [~4 values, up to hundreds of rows each] to use under
        # draw_disjoint_grouped's whole-group-never-split rule without blowing the 60/40 split-size
        # convention). Honest item-level fallback, not an oversight.
        return None

    # ---- G-NONE (design doc §1.4) + anything not explicitly classified above: no group exists at
    # any grain finer than the whole dataset -- honest item-level fallback; caller must flag it. ----
    return None

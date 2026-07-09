"""registry — the loader lookup table: dataset name -> load_<name> callable.

Populated as ``scripts/loaders/<name>.py`` modules are added (see README.md for the interface
every loader must implement: ``def load_<name>(split='test', n=None, seed=SLICE_SEED) ->
list[Row]``).

Lazy-import discipline (CLAUDE.md): this module itself only imports stdlib + ``_common`` (light);
individual loaders keep their own heavy deps (pyarrow/soundfile/librosa/...) inside their
``load_<name>`` functions, so importing ``registry`` never touches them until a loader actually
runs.

Resilient-import discipline (task B7): six agents each contributed one section (B1-B6) below.
Every section's imports + ``LOADERS[...]= ...`` assignments are wrapped inside a small
``_register_*`` function invoked through ``_try_register`` (see below) rather than sitting at
this module's top level. That way a broken loader MODULE (syntax error, an eagerly-imported dep
that isn't installed, a typo'd symbol name, ...) only drops *that section's* keys from
``LOADERS`` -- it can never raise all the way out of ``import registry`` and take every other
already-working loader down with it. This is a stronger, import-time version of the same
resilience ``smoke_all()`` already gives at call-time for a loader that imports fine but raises
when actually invoked (e.g. a missing data file).
"""
from __future__ import annotations

import os
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders

from _common import SLICE_SEED  # noqa: E402

# name -> load_<name> callable, signature: (split='test', n=None, seed=SLICE_SEED) -> list[Row].
LOADERS: dict[str, Callable] = {}

# section -> "ExceptionType: message" for any section whose import failed outright (module-level
# crash, not a runtime loader failure -- see smoke_all() for that). Empty when every section
# imported cleanly.
IMPORT_ERRORS: dict[str, str] = {}


def _try_register(section: str, register_fn: Callable[[], None]) -> None:
    """Run ``register_fn`` (imports one section's loader modules + populates ``LOADERS``).

    On any exception, record it in ``IMPORT_ERRORS`` and continue -- see module docstring's
    "Resilient-import discipline". Deliberately catches ``Exception`` broadly: an import-time
    crash in one loader module must never prevent every other section from registering.
    """
    try:
        register_fn()
    except Exception as e:  # noqa: BLE001 -- see docstring above
        IMPORT_ERRORS[section] = f"{type(e).__name__}: {e}"
        print(f"[registry] WARNING: section {section!r} failed to import -- "
              f"{IMPORT_ERRORS[section]}", flush=True)


# --- B1-asr-content ---
# ASR/content family (K1 en, K2 zh, K3 multilingual) -- see scripts/loaders/README.md and
# wiki/2026-07-09-coverage-dataset-taxonomy.md K1/K2/K3 for the recipe each loader implements.
def _register_b1_asr_content() -> None:
    from aishell_1 import load_aishell_1
    from fleurs_r import load_fleurs_r
    from librispeech import load_librispeech
    from seed_tts_eval import load_seed_tts_eval
    from thchs_30 import load_thchs_30

    LOADERS["aishell-1"] = load_aishell_1
    LOADERS["fleurs-r"] = load_fleurs_r
    LOADERS["librispeech"] = load_librispeech
    LOADERS["seed-tts-eval"] = load_seed_tts_eval
    LOADERS["thchs-30"] = load_thchs_30


_try_register("B1-asr-content", _register_b1_asr_content)
# --- end B1-asr-content ---

# --- B2-paralinguistic ---
# SER/SID family (K4/K5) -- see scripts/loaders/README.md and
# wiki/survey/2026-07-09-datasets-lock-first14.md ("crema-d" / "meld" rows) for the recipe each
# loader implements. NOTE: cn-celeb1's official trial pairs are a SEPARATE accessor
# (cn_celeb1.trial_pairs()), not registered here -- a trial is a (enroll, test, label) pair,
# not a single Row, so it doesn't fit the load_<name>() -> list[Row] contract. meld is blocked
# on a missing ffmpeg (see meld.py's NeedsExtraction / module docstring).
def _register_b2_paralinguistic() -> None:
    from cn_celeb1 import load_cn_celeb1
    from crema_d import load_crema_d
    from csemotions import load_csemotions
    from esd import load_esd
    from meld import load_meld
    from voxceleb1_test import load_voxceleb1_test

    LOADERS["cn-celeb1"] = load_cn_celeb1
    LOADERS["crema-d"] = load_crema_d
    LOADERS["csemotions"] = load_csemotions
    LOADERS["esd"] = load_esd
    LOADERS["meld"] = load_meld
    LOADERS["voxceleb1-test-split"] = load_voxceleb1_test


_try_register("B2-paralinguistic", _register_b2_paralinguistic)
# --- end B2-paralinguistic ---

# --- B3-slu ---
# SLU family (intent + slots) -- see scripts/loaders/README.md and
# wiki/survey/2026-07-09-datasets-lock-first14.md ("slurp" / "speech-massive" rows) for the recipe
# each loader implements.
def _register_b3_slu() -> None:
    from slurp import load_slurp
    from speech_massive import load_speech_massive_de_de, load_speech_massive_fr_fr

    LOADERS["slurp"] = load_slurp
    LOADERS["speech-massive-de-DE"] = load_speech_massive_de_de
    LOADERS["speech-massive-fr-FR"] = load_speech_massive_fr_fr


_try_register("B3-slu", _register_b3_slu)
# --- end B3-slu ---

# --- B4-qa-mcq ---
# QA/MCQ family -- MMAR (audio-reasoning MCQ) + URO-Bench bucket-A (~16 verifiable-reference
# subsets; bucket-B open-ended/judge-scored and bucket-C speech-output subsets are excluded, see
# uro_bench.py's module docstring). TruthfulEval's gold is a coarse list-containment proxy for
# the real (trained-judge) TruthfulQA standard -- see uro_bench.py's "qa-truthful-weak" recipe
# note; treat that one subset's numbers as directional only.
def _register_b4_qa_mcq() -> None:
    from mmar import load_mmar
    from uro_bench import _ALL as _URO_BENCH_LOADERS

    LOADERS["mmar"] = load_mmar
    LOADERS.update({f"uro-bench-{k}": v for k, v in _URO_BENCH_LOADERS.items()})  # 16 subset keys


_try_register("B4-qa-mcq", _register_b4_qa_mcq)
# --- end B4-qa-mcq ---

# --- B5-reexamined ---
# Re-examined include-subset family (voicebench / voiceassistant-eval / air-bench Foundation) --
# see scripts/loaders/README.md and wiki/survey/2026-07-09-datasets-lock-second14.md for the
# recipe each loader implements. voicebench-ifeval ships the raw instruction_id_list/kwargs but
# NO rule checker (no permissive-licensed reference implementation locatable offline as of
# 2026-07-09 -- see voicebench.py's load_voicebench_ifeval docstring). air-bench-foundation-
# speech-grounding is registered but currently raises NeedsAirBenchFoundationAudio -- 12/25
# Foundation (task_name, dataset_name) pairs are metadata-only on this box, incl. Speech_Grounding
# itself (see air_bench_foundation.py module docstring BLOCKER).
def _register_b5_reexamined() -> None:
    from air_bench_foundation import _ALL as _AIR_BENCH_FOUNDATION_LOADERS
    from voiceassistant_eval import (
        load_voiceassistant_listening_general, load_voiceassistant_listening_music,
        load_voiceassistant_listening_sound, load_voiceassistant_listening_speech,
        load_voiceassistant_speaking_reasoning,
    )
    from voicebench import (
        load_voicebench_advbench, load_voicebench_bbh, load_voicebench_ifeval,
        load_voicebench_mmsu_spoken, load_voicebench_openbookqa, load_voicebench_sdqa,
    )

    LOADERS["voicebench-advbench"] = load_voicebench_advbench
    LOADERS["voicebench-bbh"] = load_voicebench_bbh
    LOADERS["voicebench-ifeval"] = load_voicebench_ifeval
    LOADERS["voicebench-mmsu-spoken"] = load_voicebench_mmsu_spoken
    LOADERS["voicebench-openbookqa"] = load_voicebench_openbookqa
    LOADERS["voicebench-sd-qa"] = load_voicebench_sdqa
    LOADERS["voiceassistant-listening-general"] = load_voiceassistant_listening_general
    LOADERS["voiceassistant-listening-music"] = load_voiceassistant_listening_music
    LOADERS["voiceassistant-listening-sound"] = load_voiceassistant_listening_sound
    LOADERS["voiceassistant-listening-speech"] = load_voiceassistant_listening_speech
    LOADERS["voiceassistant-speaking-reasoning"] = load_voiceassistant_speaking_reasoning
    LOADERS.update(_AIR_BENCH_FOUNDATION_LOADERS)  # 14 air-bench-foundation-* keys, Speech_Grounding first


_try_register("B5-reexamined", _register_b5_reexamined)
# --- end B5-reexamined ---

# --- B6-retrieval-tools ---
# Retrieval/tool-calling family -- Audio2Tool (spoken tool-calling, cc-by-nc-4.0 EVAL-ONLY),
# AudioCaps-QA (open-ended audio-event QA), MMSU (audio-reasoning MCQ, gated on the A1 metadata
# refetch -- see mmsu.py's NeedsMetadata), SQuTR (spoken-query-to-text-retrieval; registered here
# is ONLY the standard Row-list entry point `load_squtr` -- `load_squtr_retrieval` returns a
# different BEIR-style shape and is not part of the load_<name>() -> list[Row] contract, see
# squtr.py's module docstring). `audio2tool.load_tools_registry` and `squtr.build_mini_corpus`
# are knowledge-source/helper functions, not per-item loaders -- also excluded here.
def _register_b6_retrieval_tools() -> None:
    from audio2tool import load_audio2tool
    from audiocaps_qa import load_audiocaps_qa
    from mmsu import load_mmsu
    from squtr import load_squtr

    LOADERS["audio2tool"] = load_audio2tool
    LOADERS["audiocaps-qa"] = load_audiocaps_qa
    LOADERS["mmsu"] = load_mmsu
    LOADERS["squtr"] = load_squtr


_try_register("B6-retrieval-tools", _register_b6_retrieval_tools)
# --- end B6-retrieval-tools ---


def smoke_all(n: int = 3) -> dict:
    """Load ``n`` items from every registered loader; report success/failure without stopping.

    Prints ``name / count / keys`` per loader (count = rows actually returned, which may be <
    ``n`` for a small dataset), and continues past a failing loader rather than raising, so one
    broken loader never blocks the smoke of the rest.

    Returns:
        {name: {"ok": True, "n": <rows returned>, "keys": [...]}
                | {"ok": False, "error": "<ExceptionType>: <message>"}}
    """
    report: dict = {}
    if not LOADERS:
        print(f"[smoke_all] LOADERS is empty (n={n}) -- nothing registered yet.", flush=True)
        return report
    for name, loader in LOADERS.items():
        try:
            rows = loader(n=n, seed=SLICE_SEED)
            keys = sorted(rows[0].keys()) if rows else []
            report[name] = {"ok": True, "n": len(rows), "keys": keys}
            print(f"[smoke_all] {name}: n={len(rows)} keys={keys}", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            report[name] = {"ok": False, "error": err}
            print(f"[smoke_all] {name}: FAILED {err}", flush=True)
    return report


if __name__ == "__main__":
    smoke_all()

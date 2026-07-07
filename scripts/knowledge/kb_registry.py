"""kb_registry — the FULL (28-dataset) knowledge-organization map, in the KEY/VALUE lens.

Owner's multimodal KB architecture: KEY = speech (audio embedding), VALUE = another modality. So every
local dataset is classified by (key_modality, value_type) plus a build status. This is the
organizational全覆盖 behind the WS-C coverage diagnostic, and it makes the empty cells explicit.

key_modality : audio (primary, speech-keyed) | text (legacy passage-keyed) | none
value_type   : transcript | translation | labels | intent | answer | response | text-fact | none
status       : built | buildable | deferred | n-a
  built     — a source has actually been constructed at least once
  buildable — implementable now from on-disk data (Stage-1 PoC target)
  deferred  — needs infra we lack offline (agent simulator / DB-env / rubric model)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeSourceSpec:
    dataset: str
    key_modality: str
    value_type: str
    status: str
    note: str = ""


# One row per local dataset (docs/datasets.lock.json). KEY = audio for every speech dataset.
REGISTRY: dict[str, KnowledgeSourceSpec] = {
    # ---- ASR / ST: key=audio, value=transcript/translation ----
    "librispeech": KnowledgeSourceSpec("librispeech", "audio", "transcript", "buildable", "ASR; value=transcript"),
    "covost2": KnowledgeSourceSpec("covost2", "audio", "translation", "buildable", "ST; value=translation"),
    "fleurs-r": KnowledgeSourceSpec("fleurs-r", "audio", "translation", "buildable", "ST/LID; value=translation/lang label"),
    "seed-tts-eval": KnowledgeSourceSpec("seed-tts-eval", "audio", "transcript", "buildable", "TTS eval; value=reference transcript"),
    # ---- SER / SID: key=audio, value=labels (THE audio-native cell — acoustic content -> label) ----
    "crema-d": KnowledgeSourceSpec("crema-d", "audio", "labels", "buildable", "acoustic key -> speaker+emotion labels"),
    "meld": KnowledgeSourceSpec("meld", "audio", "labels", "buildable", "acoustic key -> emotion labels"),
    # ---- SLU: key=audio, value=intent(+slots) ----
    "minds14": KnowledgeSourceSpec("minds14", "audio", "intent", "buildable", "acoustic key -> banking intent"),
    "slurp": KnowledgeSourceSpec("slurp", "audio", "intent", "buildable", "acoustic key -> intent+slots"),
    "speech-massive": KnowledgeSourceSpec("speech-massive", "audio", "intent", "buildable", "acoustic key -> intent+slots (12 langs)"),
    # ---- audio understanding / reasoning: key=audio, value=answer ----
    "mmau-mini": KnowledgeSourceSpec("mmau-mini", "audio", "answer", "buildable", "audio-understanding MCQ"),
    "mmar": KnowledgeSourceSpec("mmar", "audio", "answer", "buildable", "audio reasoning"),
    "air-bench": KnowledgeSourceSpec("air-bench", "audio", "answer", "buildable", "audio benchmark"),
    "mmsu": KnowledgeSourceSpec("mmsu", "audio", "answer", "buildable", "spoken-reasoning MCQ"),
    # ---- spoken-QA: key=audio, value=answer (legacy text-key passage pool also possible) ----
    "big-bench-audio": KnowledgeSourceSpec("big-bench-audio", "audio", "answer", "built", "spoken-reasoning QA (T9)"),
    "heysquad": KnowledgeSourceSpec("heysquad", "audio", "answer", "built", "extractive spoken-QA; value LEAKS gold -> scrub"),
    "spoken-squad": KnowledgeSourceSpec("spoken-squad", "audio", "answer", "buildable", "ASR-noise-robust spoken-QA"),
    "uro-bench": KnowledgeSourceSpec("uro-bench", "audio", "answer", "built", "only SQuAD-zh subset touched; 40+ subsets untouched"),
    "vocalbench": KnowledgeSourceSpec("vocalbench", "audio", "response", "buildable", "EN conversational"),
    "vocalbench-zh": KnowledgeSourceSpec("vocalbench-zh", "audio", "answer", "built", "ZH spoken-interaction QA (T9)"),
    # ---- agentic voice: key=audio, value=response via tools (mostly offline-infeasible) ----
    "voicebench": KnowledgeSourceSpec("voicebench", "audio", "response", "buildable", "spoken-QA + agentic suite"),
    "voiceassistant-eval": KnowledgeSourceSpec("voiceassistant-eval", "audio", "response", "buildable", "assistant tasks"),
    "audiomc": KnowledgeSourceSpec("audiomc", "audio", "response", "deferred", "multi-turn retention; offline-infeasible"),
    "eva-bench": KnowledgeSourceSpec("eva-bench", "audio", "response", "deferred", "voice-agent; simulator required"),
    "tau2-bench": KnowledgeSourceSpec("tau2-bench", "audio", "response", "deferred", "voice tool-use; DB-env required"),
    "soulx-duplug": KnowledgeSourceSpec("soulx-duplug", "audio", "none", "n-a", "full-duplex turn-taking; not a knowledge task"),
    # ---- text-keyed (no speech): pure text reasoning ----
    "aime24": KnowledgeSourceSpec("aime24", "text", "answer", "n-a", "text math reasoning; not speech-keyed"),
    "aime25": KnowledgeSourceSpec("aime25", "text", "answer", "n-a", "text math reasoning"),
    "aime26": KnowledgeSourceSpec("aime26", "text", "answer", "n-a", "text math reasoning"),
}


def by_key_modality(km: str) -> list[str]:
    return [k for k, v in REGISTRY.items() if v.key_modality == km]


def by_value_type(vt: str) -> list[str]:
    return [k for k, v in REGISTRY.items() if v.value_type == vt]


def by_status(status: str) -> list[str]:
    return [k for k, v in REGISTRY.items() if v.status == status]


def coverage_summary() -> dict:
    """Counts per (key_modality, value_type, status) — code-backed coverage matrix for WS-C."""
    km, vt, st = {}, {}, {}
    for v in REGISTRY.values():
        km[v.key_modality] = km.get(v.key_modality, 0) + 1
        vt[v.value_type] = vt.get(v.value_type, 0) + 1
        st[v.status] = st.get(v.status, 0) + 1
    return {"n_datasets": len(REGISTRY), "by_key_modality": km, "by_value_type": vt, "by_status": st}

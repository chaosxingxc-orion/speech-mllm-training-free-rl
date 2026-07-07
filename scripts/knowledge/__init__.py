"""speechrl knowledge-base ops — SPEECH-KEYED, value-typed, persistent, auditable.

Architecture (owner, 2026-07-07; mainstream vector-DB shape): KEY = speech (audio embedding) stored as
a matrix + a persisted vector index; VALUE = another modality (transcript / labels / answer / ...)
stored row-aligned as the payload. Retrieval = embed query audio -> ANN search -> return value.

Version-controlled SOURCE lives here (work repo ``scripts/knowledge/``); a deployed copy is mirrored
to the canonical operating layer ``E:\\speechrl-knowledge\\ops\\`` (co-located with the knowledge) via
``deploy_to_e.py``, stamped with the git sha in ``ops/VERSION``.

Modules (flat; heavy deps imported inside functions — CLAUDE.md lazy-import discipline):
    kb_schema    — KnowledgeValue / SourceManifest / paths (SPEECHRL_KB_DIR); KEY/VALUE model
    kb_embed     — KEY embedders: audio (omni-embed / logmel-stats PoC) + legacy text (minilm/tfidf)
    kb_index     — the KEY matrix as a persisted vector index (FAISS-flat-IP else numpy-flat)
    kb_build     — ingest dataset -> persisted source (keys.npy + index.faiss + values.jsonl + manifest)
    kb_retrieve  — load a persisted source + retrieve values by key similarity
    kb_audit     — Information-Boundary Guard as a machine guardrail (value-side answer-overlap + scrub)
    kb_inject    — standardized delivery FORMS (single-turn / two-turn tool; the clean lever)
    kb_snapshot  — freeze a reproducible eval slice (item ids + seed + revision + kb hash)
    kb_registry  — the FULL 28-dataset knowledge-organization map (key_modality × value_type × status)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from kb_schema import (  # noqa: E402
    KnowledgeValue,
    SourceManifest,
    kb_root,
    list_sources,
    load_manifest,
    load_values,
    snapshot_dir,
    source_dir,
)

__all__ = [
    "KnowledgeValue",
    "SourceManifest",
    "kb_root",
    "source_dir",
    "snapshot_dir",
    "list_sources",
    "load_manifest",
    "load_values",
]

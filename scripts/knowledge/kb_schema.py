"""KnowledgeBase schema — SPEECH-KEYED, value-typed, persistent (mainstream vector-DB shape).

Owner's architecture (2026-07-07): a MULTIMODAL knowledge base is keyed on SPEECH.
    KEY   = a dense AUDIO embedding — a row in ``keys.npy``, indexed by a persisted vector index.
    VALUE = another modality — transcript / labels / intent / answer / text-fact / other-modal ref —
            stored row-aligned in ``values.jsonl`` (the "payload").
Retrieval = embed query audio -> ANN search over the key index -> return the value. This is exactly
the mainstream vector-DB / RAG shape (embedding index + payload store; FAISS/Milvus/Qdrant/Pinecone),
applied with AUDIO as the query key — the genuinely multimodal design, replacing the ex-transient
TF-IDF text-passage pool (``t7_rag_gate_probe.py``) which was text-keyed, in-RAM, index-less.

Layout (under ``SPEECHRL_KB_DIR``, default ``E:\\speechrl-knowledge``):
    knowledge_base/<source>/keys.npy       KEY matrix [N×d] (audio embeddings, L2-normalized)
    knowledge_base/<source>/index.faiss    persisted ANN index over the keys (FAISS; optional)
    knowledge_base/<source>/values.jsonl   row-aligned VALUES (one KnowledgeValue per line)
    knowledge_base/<source>/retriever.pkl  legacy: fitted TF-IDF (only if key_modality == 'text')
    knowledge_base/<source>/manifest.json  SourceManifest (embedder, key_modality, dim, provenance, hash)
    snapshots/<experiment>/sample_manifest.json  frozen eval slice (item ids + seed + revision + hash)

Lazy-import discipline (CLAUDE.md): heavy deps live inside functions; ``import kb_schema`` stays light.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# KEY modality — what the retrieval is keyed on. Speech/omni boundary (no image, by design).
KEY_MODALITIES = ("audio", "text")  # audio = primary (multimodal); text = legacy passage-keyed
# VALUE type — what the stored payload IS (the "other modality / label / transcript").
VALUE_TYPES = ("transcript", "translation", "labels", "intent", "answer", "text-fact", "other-modal")

# Step-2 schema evolution (2026-07-10; wiki/2026-07-09-q1q2-embedder-granularity-decision-memo.md
# S2.5 + wiki/2026-07-10-step2-grid-draft.md S2/S5). KEY_GRANULARITIES tags what span of audio a
# key row was embedded from — 'utterance' (the Stage-1 default, unchanged), 'word' (CLAR-style
# word-boundary keys for ASR hard-sample memory, K1/K2), or 'segment' (sub-utterance span keys for
# SLU-slot two-level retrieval, K6/K7). VALUE_GRAINS tags what KIND of thing a value payload IS in
# the knowledge/skill/memory taxonomy (wiki 2026-07-06-capability-taxonomy...): 'knowledge' (a
# general fact/label — the pre-existing implicit default), 'memory' (a specific recalled instance,
# e.g. an ASR hard-sample correction), or 'exemplar' (a few-shot demonstration item, e.g. an
# intent/slot-tagged reference utterance). ``None`` (the default) preserves the pre-2026-07-10
# untagged behavior exactly — this is purely additive.
KEY_GRANULARITIES = ("utterance", "word", "segment")
VALUE_GRAINS = ("knowledge", "memory", "exemplar")


class KBLeakageError(RuntimeError):
    """Raised by the Information-Boundary Guard enforcement gate.

    Two call sites: ``kb_build.build_source`` refuses to PERSIST a source whose value-side leakage
    audit verdict is ``LEAKAGE`` (unless ``force_persist=True``), and ``kb_retrieve.load_source``
    refuses to LOAD a source whose ``leakage_audit.verdict`` is not ``CLEAN`` (unless
    ``allow_unclean=True``). Both escape hatches are PoC/debug-only and stamp/warn loudly so an unclean
    source can never be used for a knowledge-utilization claim silently.
    """


@dataclass(frozen=True)
class KnowledgeValue:
    """One row of the value store — the payload retrieved when its key matches.

    Step-2 additive fields (2026-07-10; all default to the pre-existing untagged behavior, so every
    values.jsonl written before this change loads unmodified — see ``from_json``):

    ``key_granularity`` — what span of audio this row's KEY embedding was built from. Default
    ``'utterance'`` (the only granularity ever used before this change). ``'word'``/``'segment'``
    keys are CHILD keys of a parent utterance-level row (see ``parent_ref``/``start_s``/``end_s``).

    ``parent_ref`` — for a child (word/segment) key, the ``kid`` of its parent utterance-level
    ``KnowledgeValue`` in the SAME source (traceability: which utterance this sub-span came from).
    ``None`` for utterance-level rows (the default — nothing points to a parent).

    ``start_s`` / ``end_s`` — the child key's span offset in seconds within the parent utterance's
    audio (nullable; only meaningful when ``key_granularity != 'utterance'``).

    ``grain`` — the VALUE's role in the knowledge/skill/memory taxonomy (wiki
    2026-07-06-capability-taxonomy-knowledge-skill-memory.md): ``'knowledge'`` (general fact/label),
    ``'memory'`` (a specific recalled instance, e.g. an ASR hard-sample correction), ``'exemplar'``
    (a few-shot demonstration item), or ``None`` (untagged — the default, preserves prior behavior;
    NOT the same as ``'knowledge'`` — callers that need a grain-aware split must treat ``None`` as
    "not yet classified", not silently treat it as knowledge-grain).
    """

    row: int  # aligns to keys.npy row and the ANN index id
    kid: str  # stable content-addressed id
    source: str
    key_modality: str  # "audio" (primary) | "text"
    value_type: str  # one of VALUE_TYPES
    value: str  # the stored knowledge (transcript / label / answer / fact / other-modal ref)
    key_audio_ref: str | None = None  # the audio whose embedding is the key (traceability)
    provenance: dict = field(default_factory=dict)  # {dataset, revision, build_seed, from_item_id, leakage_checked}
    key_granularity: str = "utterance"  # one of KEY_GRANULARITIES
    parent_ref: str | None = None  # parent utterance-level KnowledgeValue.kid, for child keys
    start_s: float | None = None  # child key span start (s) within the parent utterance's audio
    end_s: float | None = None  # child key span end (s) within the parent utterance's audio
    grain: str | None = None  # one of VALUE_GRAINS, or None (untagged)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "KnowledgeValue":
        """Backward-compatible: a pre-2026-07-10 values.jsonl line has none of the Step-2 fields
        in its JSON — the dataclass defaults (``key_granularity='utterance'``, ``parent_ref=None``,
        ``start_s=None``, ``end_s=None``, ``grain=None``) fill in automatically since
        ``KnowledgeValue(**d)`` only requires the keys actually present in ``d``.
        """
        return KnowledgeValue(**json.loads(line))


PROVENANCE_KEYS = ("dataset", "revision", "build_seed", "from_item_id", "leakage_checked")


@dataclass(frozen=True)
class SourceManifest:
    """Per-source reproducibility + audit header."""

    source: str
    dataset: str
    revision: str | None  # pinned revision from docs/datasets.lock.json
    key_modality: str  # audio | text
    value_type: str  # dominant value type of this source
    embedder: str  # "omni-embed:..." | "logmel-stats-64" | "minilm:..." | "tfidf-word12"
    dim: int  # key embedding dimension
    n_entries: int
    index_backend: str  # "faiss-flat-ip" | "numpy-flat"
    build_seed: int
    build_hash: str
    leakage_audit: dict = field(default_factory=dict)
    created_note: str = ""
    forced: bool = False  # True iff persisted despite a LEAKAGE verdict via force_persist=True

    def to_dict(self) -> dict:
        return asdict(self)


def build_hash(dataset, revision, key_modality, value_type, embedder, build_seed, n_entries) -> str:
    """Deterministic build fingerprint (no wall-clock / randomness)."""
    canonical = f"{dataset}|{revision}|{key_modality}|{value_type}|{embedder}|{build_seed}|{n_entries}"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def entry_id(source: str, key_ref: str, value: str) -> str:
    """Stable content-addressed id from (key ref, value)."""
    return f"{source}:{hashlib.sha1(f'{key_ref}|{value}'.encode('utf-8')).hexdigest()[:12]}"


def leakage_verdict(audit: dict) -> str | None:
    """The FINAL verdict of a ``leakage_audit`` dict (post-scrub if scrubbed, else the raw audit).

    Returns ``None`` if no audit was ever run (``audit`` is empty) — an unaudited source is NOT the
    same as a CLEAN one; callers that require CLEAN must treat ``None`` as "not admissible" too.
    """
    if not audit:
        return None
    return audit.get("post_scrub", audit).get("verdict")


# ---- path resolution (mirrors registry.data_root's SPEECHRL_DATA_DIR pattern) ----
def kb_root() -> Path:
    """Root of the persistent knowledge base store.

    Resolution order:
      1. ``SPEECHRL_KB_DIR`` env var, if set — always honored verbatim.
      2. Unset, on native Windows (``os.name == 'nt'``) — the Windows-literal default
         ``E:/speechrl-knowledge``.
      3. Unset, on POSIX (WSL/Linux) — ``/mnt/e/speechrl-knowledge`` IF ``/mnt/e``
         exists, else a loud ``RuntimeError``. On POSIX, ``Path("E:/speechrl-knowledge")``
         is not an absolute path — it silently creates a junk relative dir literally named
         ``E:`` under the caller's cwd. Never fall back to that; refuse instead.
    """
    env = os.environ.get("SPEECHRL_KB_DIR")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path("E:/speechrl-knowledge")
    posix_default = Path("/mnt/e/speechrl-knowledge")
    if posix_default.parent.exists():
        return posix_default
    raise RuntimeError(
        "SPEECHRL_KB_DIR is unset and the POSIX default parent '/mnt/e' does not exist. "
        "Set SPEECHRL_KB_DIR explicitly, e.g.: export SPEECHRL_KB_DIR=/mnt/e/speechrl-knowledge"
    )


def source_dir(source: str) -> Path:
    return kb_root() / "knowledge_base" / source


def snapshot_dir(experiment: str) -> Path:
    return kb_root() / "snapshots" / experiment


def load_manifest(source: str) -> SourceManifest:
    d = json.load(open(source_dir(source) / "manifest.json", encoding="utf-8"))
    return SourceManifest(**d)


def load_values(source: str) -> list[KnowledgeValue]:
    p = source_dir(source) / "values.jsonl"
    return [KnowledgeValue.from_json(l) for l in open(p, encoding="utf-8") if l.strip()]


def list_sources() -> list[str]:
    root = kb_root() / "knowledge_base"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []

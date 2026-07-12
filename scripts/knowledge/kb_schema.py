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


class KBCrossModalBlockedError(RuntimeError):
    """Raised by ``kb_retrieve``'s cross-modal AUDIO-query-into-TEXT-keyed-source bridge
    (2026-07-13, M1 engineering-base item 2: the "known follow-up" ``kb_retrieve.py``'s own
    docstring flagged on 2026-07-12 -- an audio query against a TEXT-keyed source built by
    ``kb_batch_build.build_squtr_corpus_source`` had no wired routing at all). Carries ``.status``,
    one of:

    ``"ARM-BLOCKED-cross-modal"`` — the source's embedder has no audio-query path into its
    text-key space at all (permanent, not a GPU-availability question — e.g. ``clap``,
    ``wavlm-*``, ``eres2netv2``, ``campplus``, ``emotion2vec-*``, ``dasheng``, ``clsp``, ``sense``,
    ``sensevoice-small``: none of these expose a text tower / cross-modal query encoder).

    ``"pending-GPU-window"`` — the embedder COULD support this (asymmetric query encoder that also
    accepts audio input) but needs a resident llama-server that is not up this session
    (``lco-3b``/``lco-7b``/``qwen3-omni-own``) — temporary, matching the SAME status vocabulary
    ``kb_batch_build.build_squtr_corpus_source`` already returns at BUILD time for the identical
    embedders.

    Mirrors ``KBLeakageError``/``KBSourceExistsError``'s role: a clean, specific, catchable signal
    instead of a bare ``ValueError``/``RuntimeError``, so a caller (or a test) can branch on
    ``.status`` rather than string-parse the message.
    """

    def __init__(self, message: str, status: str):
        super().__init__(message)
        self.status = status


class KBSourceExistsError(RuntimeError):
    """Raised by ``kb_build.build_source`` (2026-07-12, RI item 8) when ``source`` already has a
    persisted build (an existing ``manifest.json`` under its ``source_dir``) and the caller did not
    pass ``supersede=True``. Fixes the prior silent in-place-overwrite behaviour (forensic finding
    续15: "KB build_hash 不含内容+原位覆盖"). The caller's two ways out: build under a NEW,
    distinctly-versioned ``source`` name (e.g. ``f"{source}__v2"``), or pass ``supersede=True``
    to archive (never delete) the existing build and record it as this build's ``predecessor``.
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

    ``key_text_ref`` (2026-07-13, M1 engineering-base item 2/3) — the TEXT-key counterpart of
    ``key_audio_ref``: the raw text string whose embedding is this row's KEY, for a
    ``key_modality=='text'`` row (``None`` for an audio-keyed row). Before this field, a
    text-keyed source's ``values.jsonl`` recorded the EMBEDDING (via ``keys.npy``) but never the
    raw key text itself — no way to audit "what text produced this key" for any text-keyed source
    (legacy TF-IDF/MiniLM passage pools, ``kb_batch_build.build_squtr_corpus_source``'s corpus-doc
    keys, or the new pseudo-question keys, ``kb_batch_build.build_pseudo_question_source``) without
    re-deriving it from the value or provenance. ``None`` for every row persisted before this field
    existed (``from_json``'s dataclass-default backward-compatibility, same as every other Step-2
    additive field).
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
    key_text_ref: str | None = None  # the raw TEXT whose embedding is this row's key (text-keyed only)

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
    """Per-source reproducibility + audit header.

    2026-07-11 (ticket #25 P1c) additive fields — both default so every pre-existing
    ``manifest.json`` on disk still loads unmodified via ``SourceManifest(**d)``:

    ``embedder_token`` — the REGISTRY TOKEN (a ``kb_embed.EMBEDDERS`` key, or one of the legacy
    ``'clap'``/``'omni-embed'``/``'logmel-stats'`` names, or a ``'composite:a+b+c'`` multi-embedder
    key) that ``kb_retrieve._query_embedder`` must re-invoke to embed a QUERY into the exact same
    space as this source's persisted keys. This is the FIX for the old bug where ``_query_embedder``
    guessed the token back from ``embedder`` (the descriptive ``ename`` like ``"glap:GLAP"``) via
    fragile prefix matching that defaulted to ``'auto'`` (silently landing on CLAP) whenever the
    guess failed. ``None`` means "built before this field existed" — ``kb_retrieve`` falls back to
    ``kb_embed.infer_embedder_token(embedder)`` for those, and RAISES (never defaults to CLAP/'auto')
    if that inference is ambiguous. See ``kb_embed.resolve_embedder_token``.

    ``pool_split`` — the loader ``split`` (e.g. ``'dev'``/``'validation'``) this source's records were
    drawn from (ticket #25 P1a: this used to be baked into the SOURCE NAME itself
    — ``f"{dataset}__{embedder}__{pool_split}"`` — which collided with the runner's own 4-field
    naming convention, ``f"{dataset}__{embedder}__{key_org}__{value_org}"``
    (``run_mock.source_name_for``). Now the name carries ``key_org``/``value_org`` only, and
    ``pool_split`` moves here, into the manifest, where a builder/consumer can still recover it
    without it colliding with the runner's own naming axis. ``None`` for sources built before this
    field existed (their pool_split is only recoverable from ``created_note``, best-effort).
    """

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
    embedder_token: str | None = None  # registry token for query-time re-embedding (see docstring)
    pool_split: str | None = None  # loader split this source's records were drawn from
    # 2026-07-12 (RI mechanical remediation item 8; forensic finding 续15: "KB build_hash 不含内容
    # +原位覆盖" -- build_hash above is a pure METADATA fingerprint (dataset/embedder/seed/n_entries
    # as strings) and is IDENTICAL for two builds that differ only in actual byte content (e.g. a
    # re-embed after a code fix, or silent corruption) -- it cannot detect content drift, and
    # kb_build.build_source used to overwrite an existing source dir in place with no record of
    # what was there before. content_hash fixes the detection gap; predecessor + refuse-overwrite
    # (see kb_build.build_source) fix the in-place-overwrite gap.
    content_hash: str | None = None  # sha256(values.jsonl bytes + keys.npy bytes + sorted
    # from_item_ids + code_git_sha) -- see `content_hash_of` below. None only for a source built by
    # a pre-2026-07-12 kb_build (backward-compatible default).
    predecessor: dict | None = None  # {"source": ..., "content_hash": ..., "archived_path": ...,
    # "superseded_at": ...} if this build superseded (via --supersede, archived not deleted) an
    # earlier build under the SAME source name; None for a source that never superseded anything.

    # 2026-07-13 (ticket #37 item 6) -- provenance extension. All additive/defaulted so every
    # pre-existing manifest.json on disk still loads unmodified via SourceManifest(**d); NONE of
    # these are backfilled onto an existing manifest in place (append-only store -- see
    # scripts/knowledge/backfill_provenance.py's SIDECAR approach instead).
    code_git_sha: str | None = None  # git rev-parse HEAD of the W1 repo AT BUILD TIME -- this was
    # previously only folded INTO content_hash's input (never itself a visible, inspectable field
    # on the manifest); None for a source built before this field existed.
    code_git_dirty: bool | None = None  # True iff `git status --porcelain` was non-empty at build
    # time (uncommitted local changes were present when this source was built) -- None for a source
    # built before this field existed (git-dirty state was never captured pre-2026-07-13) OR when
    # the git query itself failed (see kb_schema.git_dirty_of's degrade-to-None contract).
    embedder_revision: str | None = None  # model name + checkpoint path/revision string, where
    # knowable (e.g. kb_embed._model_dir's resolved local snapshot directory, or a fixed HF repo id
    # for a Hub-fetched embedder) -- see kb_embed.embedder_revision_of. None when unresolvable
    # (e.g. every row arrived precomputed, so no embedder was actually invoked this build) or for a
    # source built before this field existed.
    normalization: str | None = None  # e.g. "l2" -- every embedder this repo uses L2-normalizes its
    # output vectors (see kb_embed's shared `_l2` helper) so this is "l2" for every build produced
    # by kb_build.build_source since this field's introduction; explicit rather than assumed, per
    # RI item 6's "make normalization inspectable" ask. None for a source built before this field
    # existed.
    index_backend_params: dict = field(default_factory=dict)  # the ACTUAL parameters
    # kb_index.VectorIndex.build used to construct `index_backend` (e.g. {"requested_index_type":
    # "auto", "hnsw_m": 32 | None, "metric": "inner-product (cosine, since keys are L2-normalized)"})
    # -- as opposed to `index_backend` above, which only records the RESULTING backend string
    # ("faiss-flat-ip" | "faiss-hnsw-ip" | "numpy-flat"), not what was actually requested/configured.
    # Empty dict (the dataclass default) for a source built before this field existed.
    content_hash_schema_version: int | None = None  # which CONTENT_HASH_SCHEMA_VERSION formula (see
    # `content_hash_of` below) produced THIS manifest's `content_hash` -- None for a source built
    # before content_hash existed at all (RI item 8, pre-2026-07-12); 1 for a source built between
    # item 8 and this item 6 change (the ORIGINAL content_hash_of formula: values.jsonl + keys.npy
    # bytes + sorted from_item_ids + code_git_sha ONLY); >=2 once the formula folds in more inputs
    # (embedder identity / normalization / index_backend params -- see CONTENT_HASH_SCHEMA_VERSION's
    # own comment). A v1 hash is NEVER silently reinterpreted as if it were computed under a later
    # schema version's (richer) input set.

    def to_dict(self) -> dict:
        return asdict(self)


def build_hash(dataset, revision, key_modality, value_type, embedder, build_seed, n_entries) -> str:
    """Deterministic build fingerprint (no wall-clock / randomness)."""
    canonical = f"{dataset}|{revision}|{key_modality}|{value_type}|{embedder}|{build_seed}|{n_entries}"
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


# 2026-07-13 (ticket #37 item 6): the ORIGINAL content_hash_of formula (values.jsonl + keys.npy
# bytes + sorted from_item_ids + code_git_sha) is schema version 1. Version 2 ADDITIONALLY folds
# in the embedder's identity (registry token + resolved revision/checkpoint string) and the
# normalization / index-backend parameters, so two builds that differ ONLY in e.g. a swapped
# embedder checkpoint revision or a different index-backend configuration (identical
# values.jsonl/keys.npy bytes otherwise -- a genuinely possible drift, e.g. a stale re-embed
# against an updated model snapshot that happens to produce numerically-close vectors) are no
# longer indistinguishable by content_hash alone. Bump this constant (never mutate the v1/v2
# formula bodies in place) whenever the hash's input set changes again -- a manifest's OWN
# `content_hash_schema_version` field records which formula produced ITS `content_hash`, so old
# hashes are never silently reinterpreted under a newer (different-input) formula.
CONTENT_HASH_SCHEMA_VERSION = 2


def content_hash_of(values_path, keys_path, from_item_ids, code_git_sha: str, *,
                     embedder_token: str | None = None, embedder_revision: str | None = None,
                     normalization: str | None = None, index_backend: str | None = None,
                     index_backend_params: dict | None = None,
                     schema_version: int = CONTENT_HASH_SCHEMA_VERSION) -> str:
    """Content-addressed fingerprint over the ACTUAL PERSISTED BYTES of a source (2026-07-12, RI
    item 8) -- unlike ``build_hash`` (metadata only: dataset/embedder/seed/n_entries as strings),
    this hashes ``values.jsonl``'s and ``keys.npy``'s real bytes plus the sorted ``from_item_ids``
    (order-independent -- two builds over the same rows in a different row order still hash equal)
    plus the code git sha that produced them, so it changes whenever the actual content would,
    even if every metadata field happens to coincide (e.g. a re-embed after a bugfix with the same
    build_seed/n_entries). ``values_path``/``keys_path`` accept ``str`` or ``pathlib.Path``.

    2026-07-13 (ticket #37 item 6): ``schema_version>=2`` (the default,
    ``CONTENT_HASH_SCHEMA_VERSION``) ALSO folds in ``embedder_token``/``embedder_revision``/
    ``normalization``/``index_backend``/``index_backend_params`` — pass ``schema_version=1`` to
    recompute the ORIGINAL (pre-item-6) formula exactly, e.g. for an apples-to-apples comparison
    against a manifest built before this extension (see
    ``scripts/knowledge/backfill_provenance.py``, which does exactly this).
    """
    from pathlib import Path as _Path

    h = hashlib.sha256()
    h.update(_Path(values_path).read_bytes())
    h.update(_Path(keys_path).read_bytes())
    for fid in sorted(str(x) for x in from_item_ids if x is not None):
        h.update(fid.encode("utf-8"))
    h.update((code_git_sha or "").encode("utf-8"))
    if schema_version >= 2:
        h.update(f"|schema_version={schema_version}".encode("utf-8"))
        h.update(f"|embedder_token={embedder_token or ''}".encode("utf-8"))
        h.update(f"|embedder_revision={embedder_revision or ''}".encode("utf-8"))
        h.update(f"|normalization={normalization or ''}".encode("utf-8"))
        h.update(f"|index_backend={index_backend or ''}".encode("utf-8"))
        h.update(json.dumps(index_backend_params or {}, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def git_sha_of(repo_root) -> str:
    """Best-effort ``git rev-parse HEAD`` of ``repo_root`` (str/Path) -- the "code git sha" input
    to ``content_hash_of``. Never raises: returns an ``"UNKNOWN (...)"`` placeholder string on any
    failure (e.g. no git installed, not a repo) so a content_hash can still be computed in a
    degraded environment rather than blocking a build.
    """
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"UNKNOWN ({exc})"


def git_dirty_of(repo_root) -> bool | None:
    """Best-effort ``git status --porcelain`` truthiness of ``repo_root`` (str/Path) -- 2026-07-13,
    ticket #37 item 6 (``manifest.code_git_dirty``). ``True`` iff there is ANY uncommitted change
    (staged or not) at build time, ``False`` iff the tree is clean, ``None`` on any failure (mirrors
    ``git_sha_of``'s degrade-never-raise contract, so a code_git_dirty query can never block a
    build)."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_root), text=True
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover - environment dependent
        return None


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

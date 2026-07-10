"""scripts/baselines/run_mock.py — Step-2 mock-agentic pipeline runner SKELETON (freeze-independent).

Extends the Step-1 ``run_baseline.py`` flow with a FIXED retrieval-augmentation stage, per the
Step-2 grid draft (``wiki/2026-07-10-step2-grid-draft.md`` §1-§3/§5 -- read that first). This module
does not depend on the freeze meeting having happened yet (§6 sign-off items are still open) -- it
is pure infrastructure so Phase-A cells are ready to run the moment the grid is frozen and the
per-(dataset, embedder, key_org, value_org) KB sources exist.

Pipeline (mirrors run_baseline.run_one, with ONE new stage inserted):

    loader rows -> [NEW] retrieve (kb_retrieve against a prebuilt source)
                -> [NEW] inject (per delivery mode)
                -> generate (reuse backbone adapters, run_baseline.BACKBONES)
                -> metric (metrics.score, unchanged)
                -> JSON (mock_config embedded + every frozen wave-1 field)

**STRICTLY NO ADAPTIVE LOGIC** (owner ruling, grid draft line 6: mock 口径 = 严格无 RL): a
``MockConfig`` instance is a FIXED, pre-registered pipeline description -- no reward-based
selection, no confidence gating, no per-item branching on the model's own output. Every dispatch
below (retrieval-kind post-processing, delivery rendering) is a deterministic function of the
config + the already-computed similarity scores, never of anything learned online.
``assert_no_adaptive_logic`` is a runtime invariant check (not just a docstring claim) -- it
inspects every ``MockConfig``/``RetrievalConfig`` FIELD NAME (not string values -- a retrieval kind
literal like ``"retrieve-then-select"`` legitimately contains "select" as a fixed token, not a
per-run adaptive selector) against a forbidden-substrings list.

Six grid dimensions (``MockConfig``, matching the grid draft's §2 enumeration verbatim where
possible -- open counting discrepancies are flagged in each enum's own comment, not silently
resolved):

  1. embedder             -- which of the 8 vector models keyed the prebuilt KB source
  2. key_org               -- how the KEY side is organized (single-utt / multi-granularity / ...)
  3. value_org              -- how the VALUE side is organized (knowledge-passage / memory / ...)
  4. retrieval (kind, k, threshold) -- the retrieval STRATEGY over the prebuilt source
  5. query_construction     -- how the query is built from the item (audio-direct / ASR-text / ...)
  6. delivery               -- how retrieved values are handed to the frozen backbone

**Source naming**: ``source_name_for`` matches ``kb_batch_build.build_one``'s own source-naming
convention EXACTLY (2026-07-11, ticket #25 P1a -- see that function's docstring for the migration
note re: sources built under the pre-ticket-#25 3-field name).

**Real vs. plan-only coverage** (2026-07-11, ticket #25 P2e): every one of the 35 arms now has a
REAL code path -- none raises ``NotImplementedError``. Several genuinely need GPU/a resident
backbone server at real invocation time (documented per-arm below); a test with no GPU exercises
these by monkeypatching the specific function that would talk to the server
(``test_phase_a_e2e.py``), not by the arm being a stub.

  - ``query_construction``: ``"audio-direct"`` embeds the query audio directly (unchanged).
    ``"asymmetric-encode-query"`` is a DOCUMENTED PROJECTION -- no embedder wired in ``kb_embed``
    exposes a distinct AUDIO-side query encoder (the one asymmetric API it has, omni-embed's
    ``encode_query``, is TEXT-only), so this reuses the same audio embedding path as
    ``audio-direct``; real, not a stub, this simply IS the fallback the grid draft anticipated.
    ``"asr-transcript-text-key"`` (``_asr_transcribe``) and ``"hyde-single-shot"``
    (``_hyde_generate``) both call the resident backbone server (GPU-NEEDED at real runtime) to
    get a transcript / a hypothetical-answer text, then retrieve via a cross-modal TEXT query
    bridge (``_kb_retrieve_text``) -- wired ONLY for an ``omni-embed``-family embedder_token (the
    one cross-modal bridge ``kb_embed`` exposes); any other embedder raises a clear, specific
    ``RuntimeError`` (never ``NotImplementedError``) at real-invocation time.
  - ``retrieval.kind``: ``dense-topk``/``fixed-threshold``/``retrieve-then-select``/
    ``long-context-stuffing`` unchanged. ``"two-stage"`` is now a REAL two-level retrieval: an
    on-the-fly k-means-clustered coarse index over the source's OWN key matrix (cached per
    source_obj), coarse stage over cluster centroids, fine stage over the winning clusters' member
    rows (``_two_stage_retrieve`` -- decoupled from ``value_org='raptor-lite'``'s structure on
    purpose, so it works over ANY source). ``"hybrid-bm25-dense-rrf"`` fuses the dense ANN ranking
    with a BM25 (``rank_bm25`` if installed, else a pure-python fallback) ranking over the SAME
    source's stored value texts via Reciprocal Rank Fusion (``_hybrid_bm25_dense_rrf``) -- BM25's
    text query is ``_text_query_proxy`` (the loader's own ``meta['text']``, never the gold).
    ``"ircot-2hop"`` is a real fixed 2-hop (``_ircot_2hop``): retrieve, append hop-1's top passage
    text to the text-query-proxy, retrieve again via the cross-modal text bridge, merge+dedupe.
  - ``delivery``: ``"llmlingua2-compress"`` tries the real ``llmlingua`` package; if unavailable, a
    documented extractive-compression FALLBACK (``_extractive_compress_lite``) scores each
    passage's sentences by lexical overlap with the instruction and keeps the top 50%, tagged
    ``[compressed-lite]`` so it's never mistaken for real LLMLingua-2 output.

Lazy-import discipline (CLAUDE.md): ``kb_retrieve``/``kb_schema``/``p2_baselines`` stay inside the
functions that need them; ``import run_mock`` alone (and every ``--dry-run`` path) never touches
the data root, the KB store, or a model/GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "loaders"))

import metrics                        # noqa: E402
import provenance                     # noqa: E402  (scripts/baselines/provenance.py, ticket #25 P4)
import run_baseline as rb             # noqa: E402  (reuses BACKBONES/_load_rows/generate machinery, not reimplemented)
import templates                      # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "_repro" / "step2_mock"   # separate from _repro/baselines/ -- never touches
                                                  # the wave-1 marathon's own output files/dir while
                                                  # it may be concurrently writing there.

# ---------------------------------------------------------------------------------------------
# grid-dimension enumerations -- matching wiki/2026-07-10-step2-grid-draft.md §2 as closely as the
# prose supports; every open counting question is flagged in-line rather than silently resolved.
# ---------------------------------------------------------------------------------------------

# (1) vector model / embedder (8, matches the doc's "向量模型（8）" exactly).
# 2026-07-11 (ticket #25 P1b): "qwen3-omni-hidden" -> "qwen3-omni-own" everywhere in this module --
# the registry (kb_embed.EMBEDDERS / kb_embed._LOADER_FNS / kb_batch_build.CONTENT_KEY_EMBEDDERS) is
# the TRUTH and has only ever used "qwen3-omni-own"; "qwen3-omni-hidden" was a stray token that
# never matched any real embedder key (a source_name_for/kb_retrieve lookup using it would 404).
EMBEDDERS = [
    "glap",                 # ref-config default
    "lco-3b", "lco-7b",
    "qwen3-omni-own",        # H-b: the 30B's own hidden state as embedding (precondition unblocked)
    "sense",
    "meralion-se2",
    "clap",                  # negative control (2a: CLAP's lexical key is dead, R@1 0.1 vs GLAP 93.8)
    "omni-embed-nemotron",    # NC-licensed control (nvidia/omni-embed-nemotron-3b)
]

# (2) key organization (4, matches "key 组织（4）" exactly).
KEY_ORGS = [
    "single-utt",              # ref-config default
    "multi-granularity",        # utt + word-level keys, M2R dual-scale
    "ha-multi-key",              # H-a: 2-3 specialized key spaces (content + speaker + emotion)
    "hb-single-space",            # H-b: one shared space, multiple readouts
]

# (3) value organization -- doc header says "value 组织（4+2 对照）" = 6. 2026-07-11 (ticket #25
# P2d/P2f): 'raptor-lite' is now a REAL, concretely-implemented arm (kb_batch_build._value_org_
# raptor_lite -- 2-level leaf+cluster-summary structure, k-means over value-text embeddings,
# centroid keys, no LLM calls) that resolves what USED to be the open "struct-lite = RAPTOR-lite OR
# HippoRAG-lite, 二选一" placeholder's ambiguity for the RAPTOR-lite half. 'struct-lite' is KEPT as
# a separately-enumerated token (reaching the doc's stated 6, see phase_a_cells.py's schedule) --
# it currently ALIASES 'raptor-lite's implementation (see kb_batch_build._apply_value_org) since
# the HippoRAG-lite alternative was never separately built; kept distinct so a future HippoRAG-lite
# build can occupy this slot without renumbering the grid, rather than deleting the placeholder.
VALUE_ORGS = [
    "knowledge-passage",   # ref-config default
    "memory-instance",
    "exemplar",             # grain-tagged; kb_schema.KnowledgeValue already carries value_type
    "struct-lite",           # currently ALIASES raptor-lite (see comment above); §6.2 open item
    "raptor-lite",            # 2-level leaf+cluster-summary, k-means over value embeddings (P2d)
    "audio-text-hybrid",
]

# (4) retrieval kind (7, matches "检索策略（7）"). "dense top-k∈{1,3,5}" in the doc is read here as
# ONE kind (dense-topk) whose k is a RetrievalConfig field, not 3 separate kinds -- the only reading
# under which every OTHER named setting (fixed-threshold / hybrid-BM25-RRF / retrieve-then-select /
# ircot-2hop / long-context-stuffing = 5) plus dense-topk (1) sums to 7 without also double-counting
# "两级检索（slot 默认）" as an 8th Phase-A arm (read instead as a Phase-C/K7-context default, not a
# Phase-A marginal-scan arm) -- see build_phase_a_arms's own note. Kept here as a valid `kind` value
# (some callers may still want it), just not one of the Phase-A 7.
RETRIEVAL_KINDS = [
    "dense-topk",                 # ref-config default (k=3)
    "fixed-threshold",
    "hybrid-bm25-dense-rrf",       # NOT wired for real execution (needs a sparse BM25 index)
    "retrieve-then-select",         # Hearing-More-style: fixed (non-adaptive) argmax over a pool
    "two-stage",                     # slot-default two-level retrieval; PLACEHOLDER (see apply_retrieval_kind)
    "ircot-2hop",                     # NOT wired for real execution (needs interleaved multi-hop generate+retrieve)
    "long-context-stuffing",          # no-retrieval control: stuff a large fixed pool, no ranking cut
]

# (5) query construction (4, matches "查询构造（4）").
QUERY_CONSTRUCTIONS = [
    "audio-direct",                 # ref-config default -- the ONLY one wired for real execution today
    "asr-transcript-text-key",        # own ASR transcript (not gold) used as a TEXT query
    "asymmetric-encode-query",          # embedder's dedicated query-side encoder (vs. reusing the key encoder)
    "hyde-single-shot",                  # HyDE: one greedy hypothetical-answer generation, then embed that
]

# (6) delivery (6, matches "递送（6）" exactly).
DELIVERY_MODES = [
    "flat",                       # ref-config default -- in-prompt, single turn
    "structured-ref",               # grain-marked structured reference block (NEW here, not in kb_inject)
    "two-turn-tool",                  # T10's validated lever -- reuses kb_inject.two_turn_messages verbatim
    "system-prompt",                    # retrieved values as a genuine system-role turn (NEW here)
    "relevance-edge-reorder",             # lost-in-the-middle mitigation: best/worst sim at the edges
    "llmlingua2-compress",                 # NOT wired for real execution (needs the llmlingua package + model)
]


# ---------------------------------------------------------------------------------------------
# MockConfig -- the object under comparison. Frozen (immutable): a run's whole behavior is fixed by
# one MockConfig value, never mutated mid-sweep.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    kind: str = "dense-topk"
    k: int = 3
    threshold: float | None = None


@dataclass(frozen=True)
class MockConfig:
    embedder: str
    key_org: str
    value_org: str
    retrieval: RetrievalConfig
    query_construction: str
    delivery: str
    backbone: str = "qwen3-omni-30b-gguf"


# §1 ref-config verbatim: GLAP / single-utt key / knowledge-passage value / dense top-3 cosine /
# audio-direct query / flat in-prompt delivery / Qwen3-Omni-30B backbone.
REF_CONFIG = MockConfig(
    embedder="glap", key_org="single-utt", value_org="knowledge-passage",
    retrieval=RetrievalConfig(kind="dense-topk", k=3, threshold=None),
    query_construction="audio-direct", delivery="flat", backbone="qwen3-omni-30b-gguf",
)

# Field names an adaptive/reward-driven/learned knob would plausibly be named -- checked against
# MockConfig/RetrievalConfig's own dataclass field NAMES (never their string VALUES; a retrieval
# kind literal like "retrieve-then-select" is a fixed enum token, not a per-run adaptive selector).
FORBIDDEN_KNOB_SUBSTRINGS = (
    "reward", "adaptive", "learned", "policy", "bandit", "gating_by", "selector_model",
    "online_update", "confidence_gate", "rl_",
)


def assert_no_adaptive_logic(cfg: MockConfig) -> None:
    """STRICT invariant (owner ruling: step-2 mock is 严格无 RL). Raises AssertionError if
    ``cfg`` (or its nested ``RetrievalConfig``) carries any field whose NAME matches a forbidden
    adaptive-logic substring. This is a real runtime check, not just a docstring claim -- if a
    future edit adds e.g. a ``reward_gate`` field to MockConfig, this catches it immediately."""
    for obj in (cfg, cfg.retrieval):
        for f in fields(obj):
            low = f.name.lower()
            for bad in FORBIDDEN_KNOB_SUBSTRINGS:
                assert bad not in low, (
                    f"MockConfig field {f.name!r} (on {type(obj).__name__}) contains the forbidden "
                    f"adaptive-logic substring {bad!r} -- the step-2 mock pipeline is STRICTLY no "
                    "adaptive logic (no reward selection, no gating); see "
                    "wiki/2026-07-10-step2-grid-draft.md (mock 口径 = 严格无 RL)."
                )


def validate_mock_config(cfg: MockConfig) -> None:
    """Raise ValueError listing every field whose value is outside its enumerated arm list."""
    errs = []
    if cfg.embedder not in EMBEDDERS:
        errs.append(f"embedder={cfg.embedder!r} not in {EMBEDDERS}")
    if cfg.key_org not in KEY_ORGS:
        errs.append(f"key_org={cfg.key_org!r} not in {KEY_ORGS}")
    if cfg.value_org not in VALUE_ORGS:
        errs.append(f"value_org={cfg.value_org!r} not in {VALUE_ORGS}")
    if cfg.retrieval.kind not in RETRIEVAL_KINDS:
        errs.append(f"retrieval.kind={cfg.retrieval.kind!r} not in {RETRIEVAL_KINDS}")
    if cfg.query_construction not in QUERY_CONSTRUCTIONS:
        errs.append(f"query_construction={cfg.query_construction!r} not in {QUERY_CONSTRUCTIONS}")
    if cfg.delivery not in DELIVERY_MODES:
        errs.append(f"delivery={cfg.delivery!r} not in {DELIVERY_MODES}")
    if cfg.backbone not in rb.BACKBONES:
        errs.append(f"backbone={cfg.backbone!r} not in {sorted(rb.BACKBONES)}")
    if errs:
        raise ValueError("validate_mock_config: " + "; ".join(errs))


# ---------------------------------------------------------------------------------------------
# source naming (PROVISIONAL -- see module docstring)
# ---------------------------------------------------------------------------------------------

def source_name_for(dataset_key: str, cfg: MockConfig) -> str:
    """Source-naming convention -- 2026-07-11 (ticket #25 P1a) UNIFIED with
    ``scripts/knowledge/kb_batch_build.build_one``, which now builds ``kb_build.build_source``
    sources under this EXACT name (see its docstring): one PHYSICAL source per
    (dataset, embedder, key_org, value_org) -- those four determine the actual bytes persisted in
    ``keys.npy``/``values.jsonl`` (different embedder = different vector space; different key_org
    changes which rows exist as keys at all, e.g. H-a's composite key vs. a single-utt key;
    different value_org changes what's stored as the payload). ``retrieval`` / ``query_construction``
    / ``delivery`` are RUNTIME choices over an already-built source, not separate builds, so
    deliberately NOT part of the name; ``pool_split`` lives in the manifest instead (NOT the name --
    ``kb_batch_build.build_one``'s own docstring has the migration note for any source built under
    the OLD 3-field ``<dataset>__<embedder>__<pool_split>`` convention, pre-ticket-#25). Mirrors this
    repo's existing double-underscore convention (``run_baseline.write_result``'s
    ``<dataset>__<backbone>__<split>.json``, ``gpu_session.sh``'s per-model-key pidfiles).
    """
    return f"{dataset_key}__{cfg.embedder}__{cfg.key_org}__{cfg.value_org}"


def config_hash(cfg: MockConfig) -> str:
    """Short deterministic fingerprint of a whole MockConfig (mirrors kb_schema.build_hash's
    style) -- used only for a stable, collision-resistant result filename, never for the KB source
    name itself (see source_name_for)."""
    import hashlib

    canonical = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------------------------
# query construction
# ---------------------------------------------------------------------------------------------

def _asr_transcribe(cfg: MockConfig, wav_path: str, seed: int) -> str:
    """query_construction='asr-transcript-text-key': the model's OWN ASR transcript of the query
    audio (never the gold), via the resident backbone server -- GPU-NEEDED at real runtime (calls
    the SAME llama-server generate path ``generate_mock`` uses for the real answer). Real code
    path, not a stub: a GPU-less test monkeypatches this function directly rather than this raising
    ``NotImplementedError`` (see ``test_phase_a_e2e.py``)."""
    instr = "请只输出这段音频的逐字转写文本(ASR transcript),不要输出任何其他内容或解释。"
    return generate_mock(cfg, wav_path, instr, seed=seed)


def _hyde_generate(cfg: MockConfig, wav_path: str, base_instruction: str, seed: int) -> str:
    """query_construction='hyde-single-shot': ONE greedy hypothetical-answer generation (HyDE) --
    that hypothetical answer becomes the retrieval query TEXT. GPU-NEEDED at real runtime (same
    resident backbone as the real answer generation); real code path, not a stub."""
    return generate_mock(cfg, wav_path, base_instruction, seed=seed)


def build_query(cfg: MockConfig, dataset_key: str, row: dict, seed: int = 0) -> tuple[str, str]:
    """Returns (query_kind, query_value). ``query_kind`` is ``"audio"`` (a wav path, retrieved via
    the source's own audio ``embed_query``) or ``"text"`` (a text query, retrieved via the
    cross-modal bridge ``_kb_retrieve_text`` -- see the module docstring's "Real vs. plan-only
    coverage" note for which embedders that bridge actually supports).
    """
    qc = cfg.query_construction
    if qc in ("audio-direct", "asymmetric-encode-query"):
        # 'asymmetric-encode-query' is a DOCUMENTED PROJECTION onto the same audio-direct path --
        # see module docstring; no embedder here exposes a distinct audio-side query encoder.
        return "audio", row["wav"]
    if qc == "asr-transcript-text-key":
        return "text", _asr_transcribe(cfg, row["wav"], seed=seed)
    if qc == "hyde-single-shot":
        base_instr = templates.build_instruction(dataset_key, row)
        return "text", _hyde_generate(cfg, row["wav"], base_instr, seed=seed)
    raise ValueError(f"build_query: unknown query_construction {qc!r} (expected one of {QUERY_CONSTRUCTIONS})")


# ---------------------------------------------------------------------------------------------
# retrieval-kind post-processing -- every branch is a FIXED, deterministic function of
# (kind, k, threshold) and the already-computed similarity scores (never the model's own
# output/confidence) -- see assert_no_adaptive_logic.
# ---------------------------------------------------------------------------------------------

def _candidate_pool_k(rcfg: RetrievalConfig) -> int:
    """How large a candidate pool to ask kb_retrieve.retrieve for, before apply_retrieval_kind's
    own (fixed) post-processing narrows it -- e.g. retrieve-then-select needs a bigger pool to
    pick its single best from; long-context-stuffing wants a large fixed dump."""
    if rcfg.kind == "long-context-stuffing":
        return 50
    if rcfg.kind in ("retrieve-then-select", "two-stage"):
        return max(rcfg.k * 4, 10)
    return rcfg.k


def _text_query_proxy(row: dict) -> str:
    """The text query proxy ``hybrid-bm25-dense-rrf``/``ircot-2hop`` use when the ACTUAL
    ``query_construction`` is audio (e.g. Phase-A's one-factor-at-a-time ref-config pin,
    ``audio-direct``): the loader's own ``meta['text']`` (the same field
    ``kb_batch_build._extract_value(value_spec='text')`` stores as a KB VALUE for every registry
    Row -- see that function's docstring) -- NEVER the gold answer. This is retrieval-INTERNAL
    bookkeeping for a FIXED, non-adaptive algorithm (``assert_no_adaptive_logic``); it is never
    itself delivered to the model (only the retrieved passages are, via ``render_delivery``), so
    using it here does not violate the Information-Boundary-Guard boundary note in
    ``templates.py``'s module docstring.
    """
    text = row.get("meta", {}).get("text")
    if not text:
        raise RuntimeError(
            "_text_query_proxy: this row carries no meta['text'] -- hybrid-bm25-dense-rrf/"
            "ircot-2hop need a text query proxy and this dataset's loader doesn't provide one."
        )
    return str(text)


def _bm25_okapi_pure(corpus_texts: list[str], query_text: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Pure-python BM25 Okapi scores (no dependency) -- fallback for ``_bm25_scores`` when
    ``rank_bm25`` isn't installed. Whitespace tokenization, lowercased; adequate for a Phase-A
    structural-coverage retrieval kind, not a production sparse-retrieval implementation."""
    import math

    docs = [t.lower().split() for t in corpus_texts]
    n_docs = len(docs)
    avgdl = (sum(len(d) for d in docs) / n_docs) if n_docs else 0.0
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((n_docs - c + 0.5) / (c + 0.5) + 1) for w, c in df.items()}
    q_terms = query_text.lower().split()
    scores = []
    for d in docs:
        dl = len(d) or 1
        tf: dict[str, int] = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in q_terms:
            f = tf.get(w)
            if not f:
                continue
            s += idf.get(w, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / (avgdl or 1)))
        scores.append(s)
    return scores


def _bm25_scores(corpus_texts: list[str], query_text: str) -> list[float]:
    """``rank_bm25`` if installed, else the pure-python fallback above (ticket #25 P2e: "rank_bm25
    or pure-python BM25")."""
    try:
        from rank_bm25 import BM25Okapi

        tokenized = [t.lower().split() for t in corpus_texts]
        return list(BM25Okapi(tokenized).get_scores(query_text.lower().split()))
    except ImportError:
        return _bm25_okapi_pure(corpus_texts, query_text)


def _hybrid_bm25_dense_rrf(source_obj: dict, raw_hits: list[dict], row: dict, rcfg: RetrievalConfig,
                            rrf_k: int = 60) -> list[dict]:
    """retrieval.kind='hybrid-bm25-dense-rrf': fuse the dense ANN ranking (``raw_hits``, already
    sim-sorted) with a BM25 ranking over ALL of this source's stored value texts, via Reciprocal
    Rank Fusion, over the union of the two candidate sets."""
    query_text = _text_query_proxy(row)
    values = source_obj["values"]
    bm25_scores = _bm25_scores([v.value for v in values], query_text)
    bm25_topk = sorted(range(len(values)), key=lambda i: bm25_scores[i], reverse=True)[: max(rcfg.k * 4, 10)]
    bm25_rank = {row_i: rank for rank, row_i in enumerate(bm25_topk)}

    dense_rank, hit_by_row = {}, {}
    for rank, h in enumerate(raw_hits):
        row_i = h["value"].row
        dense_rank[row_i] = rank
        hit_by_row[row_i] = h

    fused = []
    for row_i in set(dense_rank) | set(bm25_rank):
        rrf_score = 0.0
        if row_i in dense_rank:
            rrf_score += 1.0 / (rrf_k + dense_rank[row_i] + 1)
        if row_i in bm25_rank:
            rrf_score += 1.0 / (rrf_k + bm25_rank[row_i] + 1)
        h = hit_by_row.get(row_i)
        val = h["value"] if h else values[row_i]
        sim = h["sim"] if h else float(bm25_scores[row_i])
        fused.append((rrf_score, {"value": val, "sim": sim}))
    fused.sort(key=lambda t: t[0], reverse=True)
    return [h for _s, h in fused[: rcfg.k]]


def _two_stage_retrieve(source_obj: dict, query_val, rcfg: RetrievalConfig, coarse_m: int = 2,
                         n_clusters: int | None = None) -> list[dict]:
    """retrieval.kind='two-stage': a REAL two-level retrieval via an on-the-fly k-means-clustered
    coarse index over this SOURCE's own key matrix -- deliberately decoupled from
    ``value_org='raptor-lite'``'s cluster/summary structure (see grid draft §2's "coarse
    stage over cluster/summary index (from raptor-lite structure OR a downsampled index)" --
    'downsampled index' is the branch taken here) so 'two-stage' works over ANY source, not only
    ones built with value_org='raptor-lite'. Coarse stage: cluster the full key matrix (cached on
    ``source_obj`` after the first call this process) into ``n_clusters`` groups, rank cluster
    CENTROIDS against the query, keep the nearest ``coarse_m``. Fine stage: real cosine similarity
    of the query against only the member rows of those clusters, sim-sorted, truncated to
    ``rcfg.k``.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    index = source_obj["index"]
    keys = index.keys
    n = keys.shape[0]
    coarse = source_obj.get("_two_stage_coarse")
    if coarse is None:
        k = max(1, min(n_clusters or max(2, min(8, n // 3 or 1)), n))
        if k <= 1:
            labels = np.zeros(n, dtype=int)
            centroids = keys.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(keys)
            labels, centroids = km.labels_, km.cluster_centers_
        centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9, None)
        coarse = {"labels": labels, "centroids": centroids}
        source_obj["_two_stage_coarse"] = coarse

    q_vec = np.asarray(source_obj["embed_query"]([query_val]), dtype="float32")
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)
    csims = (q_vec @ coarse["centroids"].T)[0]
    top_clusters = set(np.argsort(-csims)[: min(coarse_m, len(csims))].tolist())
    member_idx = [i for i in range(n) if coarse["labels"][i] in top_clusters] or list(range(n))
    sims = (q_vec @ keys[member_idx].T)[0]
    order = np.argsort(-sims)[: rcfg.k]
    values = source_obj["values"]
    return [{"value": values[member_idx[o]], "sim": float(sims[o])} for o in order]


def _cross_modal_text_query_ok(source_obj: dict) -> bool:
    token = (source_obj["manifest"].embedder_token or "")
    return token.startswith("omni-embed")


def _kb_retrieve_text(source_obj: dict, text_query: str, topk: int) -> list[dict]:
    """Cross-modal TEXT query into an audio-keyed source -- the ONE bridge ``kb_embed`` exposes is
    ``embed_text(embedder='omni-embed')`` (omni-embed-nemotron's official asymmetric
    ``encode_query``), so this only succeeds when the source's ``embedder_token`` is an
    omni-embed-family token. Raises a specific, informative ``RuntimeError`` (never
    ``NotImplementedError``) for any other embedder -- a genuine infra gap for those embedders
    (no text-side query encoder wired), not a stub. Backs ``asr-transcript-text-key``/
    ``hyde-single-shot`` query_construction and ``ircot-2hop``'s 2nd hop.
    """
    if not _cross_modal_text_query_ok(source_obj):
        token = source_obj["manifest"].embedder_token
        raise RuntimeError(
            f"_kb_retrieve_text: source embedder_token={token!r} has no cross-modal text-query "
            "bridge wired in kb_embed (only an 'omni-embed'-family token exposes encode_query for "
            "text -- see kb_embed.py's module docstring). Build/query this arm against an "
            "omni-embed-nemotron KB source instead of the ref-config's pinned embedder."
        )
    import numpy as np

    import kb_embed  # lazy; scripts/knowledge

    _name, q_vec, _fitted = kb_embed.embed_text([text_query], embedder="omni-embed")
    q_vec = np.asarray(q_vec, dtype="float32")
    index = source_obj["index"]
    if q_vec.shape[1] != index.dim:
        raise ValueError(f"_kb_retrieve_text: query dim {q_vec.shape[1]} != index dim {index.dim}")
    idx, sims = index.search(q_vec, topk)
    values = source_obj["values"]
    return [{"value": values[int(j)], "sim": float(sims[0][c])} for c, j in enumerate(idx[0])]


def _ircot_2hop(source_obj: dict, row: dict, rcfg: RetrievalConfig, candidate_k: int) -> list[dict]:
    """retrieval.kind='ircot-2hop': a real fixed 2-hop -- retrieve (hop 1, the actual query audio)
    -> append hop-1's top-1 passage text to the text-query-proxy -> retrieve again (hop 2, via the
    cross-modal text bridge) -> merge (dedupe by ``kid``, sim-descending, truncate to ``rcfg.k``).
    """
    hop1 = _kb_retrieve(source_obj, row["wav"], topk=candidate_k)
    if not hop1:
        return []
    query_text = _text_query_proxy(row) + " " + hop1[0]["value"].value
    hop2 = _kb_retrieve_text(source_obj, query_text, topk=candidate_k)
    merged, seen = [], set()
    for h in hop1 + hop2:
        kid = h["value"].kid
        if kid in seen:
            continue
        seen.add(kid)
        merged.append(h)
    merged.sort(key=lambda h: h["sim"], reverse=True)
    return merged[: rcfg.k]


def apply_retrieval_kind(hits: list[dict], rcfg: RetrievalConfig, *, source_obj: dict | None = None,
                          query_val=None, row: dict | None = None) -> list[dict]:
    """``hits``: kb_retrieve.retrieve()[0]-shaped list of {"value": KnowledgeValue, "sim": float},
    already ANN-sorted by descending similarity. Returns the FINAL passage list for this cell.

    ``source_obj``/``query_val``/``row`` (2026-07-11, ticket #25 P2e) are needed ONLY by the 3
    kinds that must look beyond the pre-fetched ``hits`` pool (``two-stage``'s own coarse index,
    ``hybrid-bm25-dense-rrf``'s full-corpus BM25 scan, ``ircot-2hop``'s 2nd retrieve call) -- every
    other kind ignores them, unchanged from before this task.
    """
    kind = rcfg.kind
    if kind == "dense-topk":
        return hits[: rcfg.k]
    if kind == "fixed-threshold":
        if rcfg.threshold is None:
            raise ValueError("apply_retrieval_kind: kind='fixed-threshold' requires retrieval.threshold to be set")
        return [h for h in hits if h["sim"] >= rcfg.threshold]
    if kind == "retrieve-then-select":
        # Hearing-More-style FIXED retrieve-then-select: deterministic argmax over a larger
        # candidate pool (hits is already sim-sorted, so this is just the head) -- not a
        # learned/reward-driven selector.
        return hits[:1] if hits else []
    if kind == "two-stage":
        if source_obj is None or query_val is None:
            raise RuntimeError(
                "apply_retrieval_kind: kind='two-stage' needs source_obj+query_val (a real KB "
                "source + the query) -- not available from a --dry-run preview with no KB access."
            )
        return _two_stage_retrieve(source_obj, query_val, rcfg)
    if kind == "long-context-stuffing":
        return hits  # no-retrieval control: stuff the whole (large) candidate pool, no cut
    if kind == "hybrid-bm25-dense-rrf":
        if source_obj is None or row is None:
            raise RuntimeError(
                "apply_retrieval_kind: kind='hybrid-bm25-dense-rrf' needs source_obj+row (the full "
                "value corpus + a text query proxy) -- not available from a --dry-run preview."
            )
        return _hybrid_bm25_dense_rrf(source_obj, hits, row, rcfg)
    if kind == "ircot-2hop":
        if source_obj is None or row is None:
            raise RuntimeError(
                "apply_retrieval_kind: kind='ircot-2hop' needs source_obj+row for its 2nd hop -- "
                "not available from a --dry-run preview."
            )
        return _ircot_2hop(source_obj, row, rcfg, max(len(hits), rcfg.k))
    raise ValueError(f"apply_retrieval_kind: unknown retrieval.kind {kind!r} (expected one of {RETRIEVAL_KINDS})")


# ---------------------------------------------------------------------------------------------
# delivery rendering -- flat/structured-ref/relevance-edge-reorder are plain strings;
# two-turn-tool/system-prompt are messages lists (see generate_mock for how each shape reaches the
# backbone).
# ---------------------------------------------------------------------------------------------

_GRAIN_MARKERS = {
    "transcript": "[content]", "translation": "[content]", "labels": "[label]",
    "intent": "[intent]", "answer": "[answer]", "text-fact": "[fact]", "other-modal": "[ref]",
}


def _grain_marker(value_type: str | None) -> str:
    return _GRAIN_MARKERS.get(value_type or "", "[ref]")


def _apply_relevance_edge_reorder(hits: list[dict]) -> list[dict]:
    """Deterministic, FIXED reordering rule (lost-in-the-middle mitigation): interleave
    sim-descending hits so the two strongest sit at the delivered list's two EDGES and the
    weakest sit in the middle -- a fixed positional function of the already-computed similarity
    ranks, never a reward/gating decision (see assert_no_adaptive_logic)."""
    ordered = sorted(hits, key=lambda h: h["sim"], reverse=True)
    out = [None] * len(ordered)
    lo, hi = 0, len(ordered) - 1
    for i, h in enumerate(ordered):
        if i % 2 == 0:
            out[lo] = h
            lo += 1
        else:
            out[hi] = h
            hi -= 1
    return out


def _flat_prompt(passages: list[str], base_instruction: str) -> str:
    if not passages:
        return base_instruction
    block = "\n---\n".join(passages)
    return f"Reference material (retrieved, may or may not be relevant):\n{block}\n\n{base_instruction}"


def _structured_ref_prompt(passages: list[str], grains: list[str], base_instruction: str) -> str:
    if not passages:
        return base_instruction
    lines = [f"{_grain_marker(g)} {p}" for g, p in zip(grains, passages)]
    block = "\n".join(lines)
    return (
        "Reference material (retrieved, structured by grain; may or may not be relevant):\n"
        f"{block}\n\n{base_instruction}"
    )


def _two_turn_messages(passages: list[str], base_instruction: str) -> list[dict]:
    """Reuses kb_inject.two_turn_messages VERBATIM (the T10 validated lever -- ~doubled adoption
    on SQuAD-zh; kb_inject.py's own docstring says its templates are "kept verbatim so the
    persisted KB stays comparable to prior results", so this delivery mode must not reinvent it)."""
    knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")
    if knowledge_dir not in sys.path:
        sys.path.insert(0, knowledge_dir)
    import kb_inject  # lazy; scripts/knowledge (owned by a concurrent agent this session -- read-only use)

    return kb_inject.two_turn_messages(passages, instruction=base_instruction)


def _system_prompt_messages(passages: list[str], base_instruction: str) -> list[dict]:
    """NEW here (not in kb_inject.DELIVERY_FORMS): retrieved values as a genuine system-role turn,
    rather than inlined into the user's instruction text."""
    if passages:
        sys_text = (
            "You have access to the following retrieved reference material (may or may not be "
            "relevant to the upcoming question):\n" + "\n---\n".join(passages)
        )
    else:
        sys_text = "No reference material was retrieved for this item."
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": base_instruction}]


def _sentence_split(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?。!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _extractive_compress_lite(passage: str, query_text: str) -> str:
    """Extractive-compression FALLBACK for delivery='llmlingua2-compress' when the real
    ``llmlingua`` package isn't installed (2026-07-11, ticket #25 P2e): split the passage into
    sentences, score each by lexical (word-set Jaccard) overlap with ``query_text``, keep the top
    50% (at least 1 sentence), rejoin in ORIGINAL order (not re-sorted by score, so the compressed
    passage still reads coherently). Tagged ``[compressed-lite]`` in the caller so it is never
    mistaken for real LLMLingua-2 output.
    """
    sentences = _sentence_split(passage)
    if len(sentences) <= 1:
        return passage
    q_words = set(query_text.lower().split())

    def _score(s: str) -> float:
        w = set(s.lower().split())
        return len(w & q_words) / len(w | q_words) if (w or q_words) else 0.0

    scored = list(enumerate(sentences))
    keep_n = max(1, len(sentences) // 2)
    kept_idx = {i for i, _s in sorted(scored, key=lambda t: _score(t[1]), reverse=True)[:keep_n]}
    return " ".join(s for i, s in enumerate(sentences) if i in kept_idx)


def _llmlingua2_compress(passages: list[str], base_instruction: str) -> list[str]:
    """delivery='llmlingua2-compress': real ``llmlingua`` (LLMLingua-2) if installed, else the
    documented extractive-compression fallback above -- see its docstring. Real code path either
    way, never ``NotImplementedError``."""
    try:
        from llmlingua import PromptCompressor

        compressor = PromptCompressor(use_llmlingua2=True)
        out = []
        for p in passages:
            res = compressor.compress_prompt(p, rate=0.5)
            out.append(res.get("compressed_prompt", p) if isinstance(res, dict) else str(res))
        return out
    except ImportError:
        return [f"[compressed-lite] {_extractive_compress_lite(p, base_instruction)}" for p in passages]


def render_delivery(cfg: MockConfig, hits: list[dict], base_instruction: str):
    """Dispatch on cfg.delivery. Returns str (single-turn shapes) or list[dict] messages
    (two-turn-tool / system-prompt) -- see generate_mock for how each reaches the backbone."""
    delivery = cfg.delivery
    ordered = _apply_relevance_edge_reorder(hits) if delivery == "relevance-edge-reorder" else hits
    passages = [h["value"].value for h in ordered]
    grains = [h["value"].value_type for h in ordered]

    if delivery in ("flat", "relevance-edge-reorder"):
        return _flat_prompt(passages, base_instruction)
    if delivery == "structured-ref":
        return _structured_ref_prompt(passages, grains, base_instruction)
    if delivery == "two-turn-tool":
        return _two_turn_messages(passages, base_instruction)
    if delivery == "system-prompt":
        return _system_prompt_messages(passages, base_instruction)
    if delivery == "llmlingua2-compress":
        compressed = _llmlingua2_compress(passages, base_instruction)
        return _flat_prompt(compressed, base_instruction)
    raise ValueError(f"render_delivery: unknown delivery mode {delivery!r} (expected one of {DELIVERY_MODES})")


# ---------------------------------------------------------------------------------------------
# generation -- generalizes run_baseline.gen_llamacpp's request shape to an arbitrary MESSAGES
# list (needed for two-turn-tool/system-prompt; run_baseline's own generate() only ever sends ONE
# user turn). BACKBONES/sampling_params_for/strip_prefix are all REUSED from run_baseline, not
# reimplemented.
# ---------------------------------------------------------------------------------------------

def _gen_llamacpp_messages(base_url: str, wav_path: str, messages: list[dict], seed: int,
                            temp: float = 0.0, maxtok: int = 200) -> str:
    """Same request/response shape as run_baseline.gen_llamacpp, generalized to a caller-supplied
    MESSAGES list. The audio is attached to the LAST message (the final ask) -- exactly where
    run_baseline.gen_llamacpp attaches it for its single user turn -- so a delivery mode can
    prepend as many extra user/assistant/system turns as it needs."""
    import base64
    import urllib.request

    import p2_baselines as p2  # lazy: reads SPEECHRL_DATA_DIR at import time

    b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    msgs = [dict(m) for m in messages]  # shallow copy -- never mutate the caller's list
    last = msgs[-1]
    text_part = last["content"] if isinstance(last["content"], str) else ""
    last["content"] = [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": text_part},
    ]
    payload = {"messages": msgs, "max_tokens": maxtok, "seed": seed, "temperature": temp,
               **p2.sampling_defaults()}
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def generate_mock(cfg: MockConfig, wav_path: str, delivery_payload, seed: int) -> str:
    """Dispatches on delivery_payload's shape (str -> run_baseline.gen_llamacpp reused verbatim;
    list[dict] messages -> _gen_llamacpp_messages above). Reuses run_baseline.BACKBONES for the
    endpoint/model_tag/strip_prefix -- never reimplements the backbone roster."""
    cfg_bb = rb.BACKBONES[cfg.backbone]
    if cfg_bb["kind"] != "llamacpp":
        raise ValueError(
            f"generate_mock: backbone kind {cfg_bb['kind']!r} not supported -- the mock grid is "
            "llama.cpp-only per the two-backbone roster (run_baseline.BACKBONES)."
        )
    base_url = os.environ.get(cfg_bb["server_env"], cfg_bb["default_url"])
    if isinstance(delivery_payload, str):
        text = rb.gen_llamacpp(base_url, wav_path, delivery_payload, seed=seed)
    else:
        text = _gen_llamacpp_messages(base_url, wav_path, delivery_payload, seed=seed)
    prefix = cfg_bb.get("strip_prefix")
    if prefix and text.startswith(prefix):
        text = text[len(prefix):].lstrip()
    return text


# ---------------------------------------------------------------------------------------------
# KB source loading (lazy -- adds scripts/knowledge to sys.path only when actually needed)
# ---------------------------------------------------------------------------------------------

def _kb_path_ready() -> None:
    knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")
    if knowledge_dir not in sys.path:
        sys.path.insert(0, knowledge_dir)


def _load_kb_source(source: str):
    _kb_path_ready()
    import kb_retrieve  # lazy; scripts/knowledge (owned by a concurrent agent this session -- read-only use)

    return kb_retrieve.load_source(source)


def _kb_retrieve(source_obj: dict, query_val: str, topk: int) -> list[dict]:
    _kb_path_ready()
    import kb_retrieve  # lazy

    return kb_retrieve.retrieve(source_obj, [query_val], topk=topk)[0]


def _retrieve_by_qkind(source_obj: dict, qkind: str, query_val: str, topk: int) -> list[dict]:
    """Dispatch on ``qkind`` (from ``build_query``): ``"audio"`` -> the source's own audio
    ``embed_query`` path (``_kb_retrieve``); ``"text"`` -> the cross-modal bridge
    (``_kb_retrieve_text``, only wired for an omni-embed-family ``embedder_token`` -- see its
    docstring)."""
    if qkind == "audio":
        return _kb_retrieve(source_obj, query_val, topk=topk)
    if qkind == "text":
        return _kb_retrieve_text(source_obj, query_val, topk=topk)
    raise ValueError(f"_retrieve_by_qkind: unknown query kind {qkind!r} (expected 'audio' or 'text')")


def _exclude_own_item(hits: list[dict], item_id) -> list[dict]:
    """Priority-3g (ticket #25): drop any retrieved hit whose ``provenance.from_item_id`` equals
    the CURRENT item's own id -- a KB source built from a pool that (by construction or by
    accident) includes the very item being evaluated must never hand that item's own entry back to
    itself as "retrieved knowledge". Applied to the RAW (pre-``apply_retrieval_kind``) candidate
    pool so a self-hit never silently eats one of the final ``k`` slots."""
    if item_id is None:
        return hits
    return [h for h in hits if h["value"].provenance.get("from_item_id") != item_id]


def _record_retrieved(hits: list[dict], source: str) -> list[dict]:
    """Priority-3g (ticket #25): the exact retrieved-passage record persisted per item in the
    result JSON -- ``source``/``from_item_id``/``sim``/``text``, so a result is independently
    auditable (e.g. re-running the Information-Boundary Guard over what was ACTUALLY retrieved for
    each item, not just the aggregate score)."""
    return [
        {
            "source": source,
            "from_item_id": h["value"].provenance.get("from_item_id"),
            "sim": h["sim"],
            "text": h["value"].value,
        }
        for h in hits
    ]


# ---------------------------------------------------------------------------------------------
# result paths (separate namespace from run_baseline's _repro/baselines/ -- never collides with
# the concurrently-running wave-1 marathon's own output files).
# ---------------------------------------------------------------------------------------------

def result_path_mock(dataset: str, backbone: str, split: str, cfg: MockConfig) -> Path:
    return OUT_DIR / f"{dataset}__{backbone}__{split}__{config_hash(cfg)}.json"


def write_result_mock(result: dict, cfg: MockConfig) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = result_path_mock(result["dataset"], result["backbone"], result["split"], cfg)
    json.dump(result, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return p


# ---------------------------------------------------------------------------------------------
# per-cell runner (real execution -- requires a prebuilt KB source + the GPU session held + the
# target backbone's resident llama-server up; NOT exercised by this task, see the module docstring)
# ---------------------------------------------------------------------------------------------

def run_one_mock(dataset_key: str, cfg: MockConfig, split: str, n: int | None) -> dict:
    assert_no_adaptive_logic(cfg)
    validate_mock_config(cfg)

    seed = rb.DEV_SEED if split == "dev" else rb.TEST_SEED
    n = n if n is not None else (rb.DEV_N if split == "dev" else rb.TEST_N)

    rb.assert_gpu_session_held(rb.BACKBONES[cfg.backbone].get("gpu_session_name", cfg.backbone))

    # rb._load_rows is "private" (leading underscore) but IS the loader-dispatch entry point this
    # needs (legacy/registry/synthetic-subkey routing) -- reused rather than duplicated, exactly
    # like wave1_cells.py reuses rb.run_one/rb.write_result for the analogous reason.
    rows = rb._load_rows(dataset_key, split, n, seed)
    kt = templates.k_type_of(dataset_key) if dataset_key not in templates.LEGACY_DATASETS else "K8/K6 (legacy)"

    source = source_name_for(dataset_key, cfg)
    source_obj = _load_kb_source(source)
    candidate_k = _candidate_pool_k(cfg.retrieval)

    per_item, scores = [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        item_id = row.get("meta", {}).get("item_id")
        try:
            qkind, query_val = build_query(cfg, dataset_key, row, seed=seed + i)
            raw_hits = _retrieve_by_qkind(source_obj, qkind, query_val, topk=candidate_k)
            # P3g: own-item exclusion BEFORE the retrieval-kind's own top-k/threshold narrowing, so
            # a self-hit never silently eats one of the final k slots.
            raw_hits = _exclude_own_item(raw_hits, item_id)
            hits = apply_retrieval_kind(raw_hits, cfg.retrieval, source_obj=source_obj,
                                         query_val=query_val, row=row)
            # P3g (belt-and-suspenders): two-stage/hybrid-bm25-dense-rrf/ircot-2hop can pull hits
            # from BEYOND the pre-filtered raw_hits candidate pool (full-corpus BM25 scan, a 2nd
            # hop, cluster members) -- re-exclude the current item's own id from the FINAL list too.
            hits = _exclude_own_item(hits, item_id)
            base_instr = templates.build_instruction(dataset_key, row)
            payload = render_delivery(cfg, hits, base_instr)
            text = generate_mock(cfg, row["wav"], payload, seed=seed + i)
            result = metrics.score(dataset_key, row, text)
        except Exception as e:  # noqa: BLE001 -- one bad item must not sink the whole cell
            per_item.append({"item_id": item_id,
                              "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        per_item.append({
            "item_id": row["meta"]["item_id"], "instr_base": base_instr,
            "n_retrieved": len(hits),
            # P3g: full retrieved-passage record (source/from_item_id/sim/text) per item.
            "retrieved": _record_retrieved(hits, source),
            "reply": text,
            "score": result["score"], "detail": result["detail"],
        })
        scores.append(result["score"])
        if (i + 1) % 20 == 0:
            print(f"  [{dataset_key}/mock/{split} {i+1}/{len(rows)}] ({time.time()-t0:.0f}s)", flush=True)

    numeric = [s for s in scores if isinstance(s, (int, float))]
    mean = round(sum(numeric) / len(numeric), 4) if numeric else None
    ci = rb.paired_bootstrap(numeric, seed=seed) if numeric else None

    return {
        "stage": "2-mock-pregrid-directional",
        "dataset": dataset_key, "k_type": kt, "backbone": cfg.backbone, "split": split,
        "n_requested": n, "seed": seed,
        "mock_config": asdict(cfg),
        "source": source,
        "model": rb.BACKBONES[cfg.backbone]["model_tag"],
        "sampling_params": {"greedy_seed_base": seed, **rb.sampling_params_for(cfg.backbone)},
        "boundary": (
            "audio-only input + fixed task-definition text + retrieved KB values (never the "
            "current item's own gold) delivered via a FIXED, pre-registered pipeline -- embedder / "
            "key_org / value_org / retrieval / query_construction / delivery are all pinned per "
            "mock_config for the WHOLE cell, no per-item adaptive branching. See "
            "templates.py's Information-Boundary-Guard note and assert_no_adaptive_logic above."
        ),
        # ticket #25 P4: shared provenance block -- dataset_revision/manifest_hash resolved from
        # the KB SOURCE this cell actually queried (source_obj["manifest"]), since that IS "the
        # dataset" a mock cell is evaluated against (unlike a plain Stage-1 baseline cell).
        "provenance": provenance.collect_provenance(
            rb.REPO_ROOT,
            dataset_revision=source_obj["manifest"].revision,
            manifest_hash=source_obj["manifest"].build_hash,
        ),
        "per_item": per_item,
        "aggregate": {"mean": mean, "ci95": ci, "n_scored": len(numeric), "n_total": len(rows)},
        "reproduce": (
            f"python scripts/baselines/run_mock.py --dataset {dataset_key} --split {split} "
            f"--backbone {cfg.backbone} --embedder {cfg.embedder} --key-org {cfg.key_org} "
            f"--value-org {cfg.value_org} --retrieval-kind {cfg.retrieval.kind} "
            f"--retrieval-k {cfg.retrieval.k} --query-construction {cfg.query_construction} "
            f"--delivery {cfg.delivery}"
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------------------------
# --dry-run: render the pipeline plan for one dataset, ZERO model/KB calls.
# ---------------------------------------------------------------------------------------------

class _SyntheticValue:
    """Duck-typed stand-in for kb_schema.KnowledgeValue (``.value``/``.value_type`` only) -- used
    ONLY by --dry-run so it never imports scripts/knowledge just to render a plan preview."""

    __slots__ = ("value", "value_type")

    def __init__(self, value: str, value_type: str):
        self.value = value
        self.value_type = value_type


def _synthetic_hits(k: int) -> list[dict]:
    n = max(k, 3)
    return [
        {"value": _SyntheticValue(f"(placeholder retrieved passage #{i})",
                                   "transcript" if i % 2 == 0 else "labels"),
         "sim": round(0.9 - 0.05 * i, 3)}
        for i in range(n)
    ]


# query_construction arms whose REAL implementation calls the resident backbone server (GPU-
# needed) -- --dry-run must NEVER actually invoke these (module docstring: "ZERO model/KB/GPU
# calls"), so it previews them with a synthetic query note instead of calling build_query for real.
_GPU_NEEDED_QUERY_CONSTRUCTIONS = ("asr-transcript-text-key", "hyde-single-shot")


def dry_run_mock(dataset_key: str, cfg: MockConfig, split: str = "dev") -> None:
    """Render the whole 6-stage plan for one dataset + MockConfig with ZERO model/KB/GPU calls.

    2026-07-11 (ticket #25 P2e): every arm now has a REAL implementation (see module docstring),
    but ``--dry-run``'s own contract (zero model/KB/GPU calls) still holds -- the 3 retrieval kinds
    that need a real KB source (``two-stage``/``hybrid-bm25-dense-rrf``/``ircot-2hop``) raise a
    documented ``RuntimeError`` when ``source_obj`` is ``None`` (see ``apply_retrieval_kind``),
    caught here and printed as a "needs a real KB source" preview note rather than propagated; the
    2 query_construction arms that need a resident GPU server (``asr-transcript-text-key``/
    ``hyde-single-shot``) are previewed WITHOUT calling ``build_query`` for real (which would try an
    actual HTTP call to the backbone server otherwise).
    """
    assert_no_adaptive_logic(cfg)
    validate_mock_config(cfg)

    kt = "legacy" if dataset_key in templates.LEGACY_DATASETS else templates.k_type_of(dataset_key)
    source = source_name_for(dataset_key, cfg)
    row = rb._synthetic_row(dataset_key)
    base_instr = templates.build_instruction(dataset_key, row)

    synthetic_hits = _synthetic_hits(cfg.retrieval.k)
    try:
        posted = apply_retrieval_kind(synthetic_hits, cfg.retrieval)
        retrieval_note = f"{len(posted)} passage(s) after kind={cfg.retrieval.kind!r}"
    except (NotImplementedError, RuntimeError) as e:
        posted = synthetic_hits[: cfg.retrieval.k]
        retrieval_note = f"<PLAN PREVIEW ONLY (needs a real KB source, see below) -- {e}>"

    if cfg.query_construction in _GPU_NEEDED_QUERY_CONSTRUCTIONS:
        qkind = "text"
        query_note = (f"query_construction={cfg.query_construction!r} -> kind=text "
                      "<PLAN PREVIEW ONLY -- real impl needs the resident GPU backbone server, "
                      "not called from --dry-run; see build_query/_asr_transcribe/_hyde_generate>")
    else:
        try:
            qkind, _qval = build_query(cfg, dataset_key, row, seed=0)
            query_note = f"query_construction={cfg.query_construction!r} -> kind={qkind}"
        except (NotImplementedError, RuntimeError) as e:
            query_note = f"<PLAN PREVIEW ONLY -- {e}>"

    try:
        payload = render_delivery(cfg, posted, base_instr)
        if isinstance(payload, str):
            preview = payload.replace("\n", " \\n ")[:200]
        else:
            preview = f"<messages x{len(payload)}: roles={[m['role'] for m in payload]}>"
    except (NotImplementedError, RuntimeError) as e:
        preview = f"<PLAN PREVIEW ONLY -- {e}>"

    print(f"=== mock pipeline plan: {dataset_key} [{kt}] split={split} backbone={cfg.backbone} ===")
    print(f"  source: {source}")
    print(f"  1. loader rows   : rb._load_rows({dataset_key!r}, {split!r}, n, seed)")
    print(f"  2. query constr. : {query_note}")
    print(f"  3. retrieve      : kb_retrieve.retrieve(source_obj, [query], "
          f"topk={_candidate_pool_k(cfg.retrieval)}) -> apply_retrieval_kind("
          f"kind={cfg.retrieval.kind!r}, k={cfg.retrieval.k}, threshold={cfg.retrieval.threshold}) "
          f"-> {retrieval_note}")
    print(f"  4. inject        : delivery={cfg.delivery!r} -> {preview}...")
    print(f"  5. generate      : generate_mock(backbone={cfg.backbone!r}, ...) (reuses "
          "run_baseline.BACKBONES/gen_llamacpp)")
    print(f"  6. metric        : metrics.score({dataset_key!r}, row, text)")
    print(f"  mock_config: {asdict(cfg)}")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _cfg_from_args(args: argparse.Namespace) -> MockConfig:
    return MockConfig(
        embedder=args.embedder, key_org=args.key_org, value_org=args.value_org,
        retrieval=RetrievalConfig(kind=args.retrieval_kind, k=args.retrieval_k,
                                   threshold=args.retrieval_threshold),
        query_construction=args.query_construction, delivery=args.delivery, backbone=args.backbone,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="squtr", help="dataset key (see templates.DATASET_KTYPE)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=None, help="override the frozen DEV_N/TEST_N for this run")
    ap.add_argument("--backbone", default=REF_CONFIG.backbone, choices=list(rb.BACKBONES.keys()))
    ap.add_argument("--embedder", default=REF_CONFIG.embedder, choices=EMBEDDERS)
    ap.add_argument("--key-org", default=REF_CONFIG.key_org, choices=KEY_ORGS)
    ap.add_argument("--value-org", default=REF_CONFIG.value_org, choices=VALUE_ORGS)
    ap.add_argument("--retrieval-kind", default=REF_CONFIG.retrieval.kind, choices=RETRIEVAL_KINDS)
    ap.add_argument("--retrieval-k", type=int, default=REF_CONFIG.retrieval.k)
    ap.add_argument("--retrieval-threshold", type=float, default=REF_CONFIG.retrieval.threshold)
    ap.add_argument("--query-construction", default=REF_CONFIG.query_construction, choices=QUERY_CONSTRUCTIONS)
    ap.add_argument("--delivery", default=REF_CONFIG.delivery, choices=DELIVERY_MODES)
    ap.add_argument("--dry-run", action="store_true", help="render the pipeline plan; no model/KB/GPU calls")
    args = ap.parse_args()

    cfg = _cfg_from_args(args)
    assert_no_adaptive_logic(cfg)   # STRICT invariant, always checked (dry-run and real alike)
    validate_mock_config(cfg)

    if args.dry_run:
        dry_run_mock(args.dataset, cfg, split=args.split)
        return

    result = run_one_mock(args.dataset, cfg, args.split, args.n)
    p = write_result_mock(result, cfg)
    print(f"wrote {p} (mean={result['aggregate']['mean']} ci={result['aggregate']['ci95']})")


if __name__ == "__main__":
    main()

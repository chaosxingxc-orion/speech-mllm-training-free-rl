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

**Source naming**: ``source_name_for`` is a PROVISIONAL convention -- ``kb_batch_build.py`` (the
step-2 engineering prefix item ② in the grid draft's §5) had not landed anywhere in this repo as of
this task (a concurrent agent owns ``scripts/knowledge/`` + ``kb_embed.py`` in this session, and
neither ``kb_batch_build`` nor any doc describing its planned source-naming scheme exists yet --
searched, zero hits). See that function's own docstring for the adopted convention and the ONE
place to change it once the real plan lands.

**Real vs. plan-only coverage** (be honest about what this skeleton can actually EXECUTE today vs.
what it can only RENDER a plan for in ``--dry-run``): only ``query_construction="audio-direct"``
and ``retrieval.kind`` in {dense-topk, fixed-threshold, retrieve-then-select, two-stage,
long-context-stuffing} are wired for a REAL ``kb_retrieve`` call. The remaining arms
(asr-transcript-text-key / asymmetric-encode-query / hyde-single-shot query construction;
hybrid-bm25-dense-rrf / ircot-2hop retrieval; llmlingua2-compress delivery) need infra this box
does not have yet (a text-query path into an audio-keyed KB source, a BM25 sparse index, an
interleaved multi-hop generate+retrieve loop, an LLMLingua-2 compression model respectively) --
each raises a documented ``NotImplementedError`` if actually invoked, but renders correctly in
``--dry-run`` (see ``dry_run_mock``), matching this repo's existing "wired but not executed" stub
convention (``run_baseline._load_hf_backbone``, ``metrics.score_k11_ifeval_stub``).

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
EMBEDDERS = [
    "glap",                 # ref-config default
    "lco-3b", "lco-7b",
    "qwen3-omni-hidden",     # H-b: the 30B's own hidden state as embedding (precondition unblocked)
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

# (3) value organization -- doc header says "value 组织（4+2 对照）" = 6, but §2's prose only NAMES
# 5 distinct tokens (knowledge-passage / memory-instance / exemplar / ONE chosen struct-lite arm
# [RAPTOR-lite OR HippoRAG-lite, a 二选一 explicitly deferred to freeze sign-off §6.2] /
# audio-text-hybrid). That is an OPEN counting discrepancy in the source doc itself (a draft, not
# yet frozen) -- NOT silently resolved to hit "6" here. phase_a_cells.py's schedule prints the
# resulting total-arm mismatch against the doc's stated 35 rather than papering over it.
VALUE_ORGS = [
    "knowledge-passage",   # ref-config default
    "memory-instance",
    "exemplar",             # grain-tagged; kb_schema.KnowledgeValue already carries value_type
    "struct-lite",           # RAPTOR-lite OR HippoRAG-lite -- WHICH one is an open §6.2 sign-off item
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
    """PROVISIONAL source-naming convention -- ``kb_batch_build.py`` (grid draft §5 item ②) does
    not exist anywhere in this repo as of this task (checked: no hits for "kb_batch_build" under
    the umbrella repo). Adopted convention: one PHYSICAL source per (dataset, embedder, key_org,
    value_org) -- those four determine the actual bytes persisted in ``keys.npy``/``values.jsonl``
    (different embedder = different vector space; different key_org changes which rows exist as
    keys at all, e.g. H-a's 2-3 key spaces vs. a single-utt key; different value_org changes what's
    stored as the payload). ``retrieval`` / ``query_construction`` / ``delivery`` are RUNTIME
    choices over an already-built source, not separate builds, so deliberately NOT part of the
    name. Mirrors this repo's existing double-underscore convention
    (``run_baseline.write_result``'s ``<dataset>__<backbone>__<split>.json``,
    ``gpu_session.sh``'s per-model-key pidfiles). RECONCILE against kb_batch_build.py's real
    convention once it lands -- this is the ONE function to change if it differs.
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

def build_query(cfg: MockConfig, dataset_key: str, row: dict) -> tuple[str, str]:
    """Returns (query_kind, query_value). Only ``"audio-direct"`` is wired for a REAL retrieve()
    call in this skeleton -- the other three query_construction arms need a TEXT-query path into
    an audio-keyed KB source that ``kb_retrieve.load_source``'s ``_query_embedder`` does not expose
    today (for ``manifest.key_modality == "audio"`` it ALWAYS returns an audio-embedding lambda,
    never a text one) -- see the module docstring's "Real vs. plan-only coverage" note. They
    render correctly in ``--dry-run`` (``dry_run_mock`` catches the NotImplementedError and shows
    the plan) but RAISE here rather than silently degrading to audio-direct or crashing inside
    ``kb_embed`` on a text string mis-typed as a wav path.
    """
    qc = cfg.query_construction
    if qc == "audio-direct":
        return "audio", row["wav"]
    if qc in ("asr-transcript-text-key", "asymmetric-encode-query", "hyde-single-shot"):
        raise NotImplementedError(
            f"build_query: query_construction={qc!r} needs a text-query-capable path into an "
            "audio-keyed KB source that kb_retrieve.load_source does not expose yet -- wired for "
            "--dry-run planning only, not for a real retrieve() call. See run_mock.py's module "
            "docstring 'Real vs. plan-only coverage'."
        )
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


def apply_retrieval_kind(hits: list[dict], rcfg: RetrievalConfig) -> list[dict]:
    """``hits``: kb_retrieve.retrieve()[0]-shaped list of {"value": KnowledgeValue, "sim": float},
    already ANN-sorted by descending similarity. Returns the FINAL passage list for this cell."""
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
        # Documented PLACEHOLDER for a real two-level (coarse index -> fine index) retrieval: this
        # skeleton has only ONE index to search, so "stage 2" re-ranks by the SAME similarity
        # score and truncates to k. A real two-level pass (a distinct coarse/fine index pair) is
        # future work, not this runner -- flagged rather than faked as something more elaborate.
        return hits[: rcfg.k]
    if kind == "long-context-stuffing":
        return hits  # no-retrieval control: stuff the whole (large) candidate pool, no cut
    if kind in ("hybrid-bm25-dense-rrf", "ircot-2hop"):
        raise NotImplementedError(
            f"apply_retrieval_kind: kind={kind!r} needs infra beyond kb_retrieve (a BM25 sparse "
            "index for RRF fusion / interleaved multi-hop generate+retrieve calls for IRCoT "
            "respectively) -- wired for --dry-run planning only. See "
            "wiki/2026-07-10-step2-grid-draft.md §2."
        )
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
        raise NotImplementedError(
            "render_delivery: llmlingua2-compress needs the llmlingua package + a compression "
            "model -- not wired in this skeleton (grid draft's 'M 裁决臂', judged manually later). "
            "See wiki/2026-07-10-step2-grid-draft.md §2."
        )
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
        try:
            _qkind, query_val = build_query(cfg, dataset_key, row)
            raw_hits = _kb_retrieve(source_obj, query_val, topk=candidate_k)
            hits = apply_retrieval_kind(raw_hits, cfg.retrieval)
            base_instr = templates.build_instruction(dataset_key, row)
            payload = render_delivery(cfg, hits, base_instr)
            text = generate_mock(cfg, row["wav"], payload, seed=seed + i)
            result = metrics.score(dataset_key, row, text)
        except Exception as e:  # noqa: BLE001 -- one bad item must not sink the whole cell
            per_item.append({"item_id": row.get("meta", {}).get("item_id"),
                              "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        per_item.append({
            "item_id": row["meta"]["item_id"], "instr_base": base_instr,
            "n_retrieved": len(hits), "reply": text,
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


def dry_run_mock(dataset_key: str, cfg: MockConfig, split: str = "dev") -> None:
    """Render the whole 6-stage plan for one dataset + MockConfig with ZERO model/KB/GPU calls --
    every stage that would need one is caught (NotImplementedError from the plan-only arms) and
    printed as a labeled plan note instead of raised."""
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
    except NotImplementedError as e:
        posted = synthetic_hits[: cfg.retrieval.k]
        retrieval_note = f"<PLAN ONLY -- {e}>"

    try:
        qkind, _qval = build_query(cfg, dataset_key, row)
        query_note = f"query_construction={cfg.query_construction!r} -> kind={qkind}"
    except NotImplementedError as e:
        query_note = f"<PLAN ONLY -- {e}>"

    try:
        payload = render_delivery(cfg, posted, base_instr)
        if isinstance(payload, str):
            preview = payload.replace("\n", " \\n ")[:200]
        else:
            preview = f"<messages x{len(payload)}: roles={[m['role'] for m in payload]}>"
    except NotImplementedError as e:
        preview = f"<PLAN ONLY -- {e}>"

    print(f"=== mock pipeline plan: {dataset_key} [{kt}] split={split} backbone={cfg.backbone} ===")
    print(f"  source (PROVISIONAL naming, pending kb_batch_build.py): {source}")
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

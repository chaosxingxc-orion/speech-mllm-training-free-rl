"""kb_batch_build — Step-2 batch build pipeline: (dataset-loader-key, embedder, key_org, value_org)
-> ONE persisted ``kb_build.build_source`` call, gated by the SAME CLEAN leakage-verdict
enforcement every other kb_build caller goes through (``kb_schema.KBLeakageError`` -- see
kb_build.py's module docstring). Reuses the existing infrastructure end to end rather than
duplicating it: ``scripts/loaders/registry.py``'s ``LOADERS`` table for the dataset side,
``kb_embed.EMBEDDERS`` for the embedder side, ``kb_build.build_source`` for the actual persist.

2026-07-11 (ticket #25): the source NAME is now UNIFIED with the runner's own convention
(``run_mock.source_name_for``, P1a) -- ``f"{dataset}__{embedder}__{key_org}__{value_org}"``;
``pool_split`` moved into the manifest (see ``kb_build.build_source``). ``key_org``/``value_org``
each drive a REAL (not plan-only) construction -- see ``_apply_key_org``/``_apply_value_org`` and
each helper's own docstring for the exact, documented approximation (multi-granularity's 2-half
sub-segment keys / ha-multi-key's composite embedder / hb-single-space's shared-key readout rows /
raptor-lite's k-means-over-value-embeddings 2-level structure). ``eval_manifest`` (P3g, optional)
machine-verifies this build's own item ids never overlap an eval slice.

``--plan`` mode prints the step-2 build MATRIX (the "8 content-key embedders" axis --
wiki/2026-07-10-step2-grid-draft.md S2 -- crossed with a small default "main-field" dataset pool
set) WITHOUT building anything. The Phase-A dataset selection itself is an owner sign-off item
(same doc, S6) -- NOT decided here; ``MAIN_FIELD_POOLS`` below is a defensible illustrative
default (one pool per major content sub-family: ASR / ST / reasoning-QA / native-retrieval),
override with ``--dataset <any registry.LOADERS key>`` for anything else.

    python scripts/knowledge/kb_batch_build.py --plan                    # matrix, builds nothing
    python scripts/knowledge/kb_batch_build.py --plan --squtr-preview    # + squtr mini-corpus preview
    python scripts/knowledge/kb_batch_build.py --embedder glap --dataset librispeech --split dev \
        --value-spec text --key-org single-utt --value-org knowledge-passage --n 40
                                                                          # actually builds ONE source

Lazy-import discipline (CLAUDE.md): heavy deps (torch/transformers/funasr/...) live inside
``kb_embed``'s per-embedder loaders and inside each ``scripts/loaders`` module's ``load_<name>``,
never at this module's top level, so ``import kb_batch_build`` and ``--plan`` stay cheap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                        # scripts/knowledge -- bare `import kb_build` etc.
sys.path.insert(0, str(HERE.parent / "loaders"))      # scripts/loaders -- bare `import registry`/`squtr`

# The step-2 "8 content-key embedders" axis (wiki/2026-07-10-step2-grid-draft.md S2) -- a NAMED
# subset of kb_embed.EMBEDDERS, since that registry also carries speaker/emotion/style embedders
# (eres2netv2, campplus, emotion2vec-*, wavlm-*, dasheng, sensevoice-small) that are out of scope
# for the content-key H-a/H-b decision this matrix is for.
CONTENT_KEY_EMBEDDERS = [
    "glap", "lco-3b", "lco-7b", "qwen3-omni-own", "sense", "meralion-se2", "clap", "omni-embed-nemotron",
]

# Candidate "main-field" (content-key) dataset pools for the --plan matrix -- ONE pool per major
# content sub-family (ASR / multilingual ST / reasoning-QA MCQ / native spoken-query retrieval),
# each a real key in scripts/loaders/registry.py's LOADERS table. This is a DEFAULT/illustrative
# set, sized to match the grid-draft's "~4 datasets" Phase-A scale -- NOT the frozen Phase-A
# selection (that is an owner sign-off item, step2-grid-draft S6). Pass --dataset to build against
# any other registry.LOADERS key instead.
MAIN_FIELD_POOLS = ["librispeech", "fleurs-r", "mmar", "squtr"]


def _registry():
    import registry  # scripts/loaders/registry.py

    return registry


def build_matrix_plan(embedders: list[str] | None = None, pools: list[str] | None = None) -> list[dict]:
    """The (embedder x main-field pool) candidate matrix -- metadata only, builds nothing."""
    import kb_embed

    registry = _registry()
    embedders = embedders or CONTENT_KEY_EMBEDDERS
    pools = pools or MAIN_FIELD_POOLS
    rows = []
    for emb in embedders:
        meta = kb_embed.EMBEDDERS.get(emb, {})
        for ds in pools:
            rows.append({
                "embedder": emb,
                "embedder_dim": meta.get("dim"),
                "embedder_license": meta.get("license"),
                "embedder_needs_server": meta.get("needs_server", False),
                "embedder_note": meta.get("note"),
                "dataset": ds,
                "dataset_registered": ds in registry.LOADERS,
            })
    return rows


def print_plan(embedders: list[str] | None = None, pools: list[str] | None = None) -> None:
    rows = build_matrix_plan(embedders, pools)
    embedders = embedders or CONTENT_KEY_EMBEDDERS
    pools = pools or MAIN_FIELD_POOLS
    print("=== step-2 build matrix (CANDIDATE plan -- Phase-A dataset selection not frozen; "
          "see module docstring) ===")
    print(f"{len(embedders)} content-key embedders x {len(pools)} main-field dataset pools "
          f"= {len(rows)} cells\n")
    header = f"{'embedder':22s} {'dim':>6s} {'server?':>8s}  {'dataset':16s} {'registered?':>11s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        dim = "?" if r["embedder_dim"] is None else str(r["embedder_dim"])
        print(f"{r['embedder']:22s} {dim:>6s} {str(r['embedder_needs_server']):>8s}  "
              f"{r['dataset']:16s} {str(r['dataset_registered']):>11s}")
    notes = {r["embedder"]: r["embedder_note"] for r in rows if r["embedder_note"]}
    if notes:
        print("\n-- embedder notes --")
        for emb, note in notes.items():
            print(f"  {emb}: {note}")


def plan_squtr_mini_corpus(subset: str = "fiqa", noise_level: str = "clean", n: int | None = 40,
                            n_distractors: int = 200, seed: int = 20260705) -> dict:
    """PLAN-MODE ONLY preview of the squtr mini-corpus value-spec path.

    wiki/2026-07-10-step2-grid-draft.md S2's hybrid BM25+dense RRF retrieval arm ("squtr 文本语料
    侧") needs a TEXT-keyed corpus pool alongside squtr's audio-keyed query side. This wires
    ``loaders/squtr.build_mini_corpus`` (via ``load_squtr_retrieval``, which already calls it) to
    preview the shape a real build would take -- gold docs + sampled distractors -- WITHOUT ever
    calling ``kb_build.build_source``: nothing is persisted, no leakage audit runs. A real build
    would key on TEXT (``key_modality='text'``) since this is squtr's TEXT corpus side, not its
    audio query side (which is what ``build_one``/registry's ``"squtr"`` loader embeds).
    """
    import squtr

    retrieval = squtr.load_squtr_retrieval(subset=subset, noise_level=noise_level, n=n, seed=seed,
                                            n_distractors=n_distractors)
    return {
        "would_build_source": f"squtr-{subset}-{noise_level}__mini-corpus",
        "key_modality": "text", "value_type": "text-fact",
        "n_queries": len(retrieval["queries"]),
        "n_corpus_sample": len(retrieval["corpus_sample"]),
        "n_qrels": len(retrieval["qrels"]),
        "note": ("PLAN ONLY -- kb_build.build_source NOT called; no leakage audit ran; nothing "
                 "persisted. Previews the mini-corpus (gold docs + sampled distractors) "
                 "loaders/squtr.build_mini_corpus would hand to a text-keyed source build for the "
                 "hybrid BM25+dense RRF arm."),
    }


# subset -> GLAP NLLB-style source_lang code (squtr.SUBSETS' own "en"/"zh" tags, mapped to GLAP's
# tokenizer's language codes -- see modeling_glap.py's LANG table; "zho_Hans" = simplified Chinese,
# matching every zh squtr subset's actual script).
_SQUTR_SUBSET_GLAP_LANG = {"en": "eng_Latn", "zh": "zho_Hans"}

# Embedders this function can actually BUILD with, CPU-only, right now (2026-07-12, RI item 9):
# glap (native encode_text) and omni-embed-nemotron (encode_document over {"text": ...}) -- see
# kb_embed.py's _glap_embed_text / _omni_embed_document_text. Every other candidate in
# CONTENT_KEY_EMBEDDERS is either GPU-server-gated (lco-3b/lco-7b/qwen3-omni-own -- recorded
# "pending-GPU-window") or has no text-embedding path wired at all in this repo currently
# (sense/meralion-se2/clap -- recorded "ARM-BLOCKED-cross-modal"); see build_squtr_corpus_source's
# docstring for the full per-embedder gating logic.
SQUTR_CORPUS_TEXT_CPU_EMBEDDERS = {"glap", "omni-embed-nemotron"}


# ---------------------------------------------------------------------------------------------
# corpus_mode (2026-07-13, ticket #38 item 1 -- F'-3 remediation, Decision-Log 续26): 'full' (the
# DEFAULT) indexes EVERY document in the official corpus.jsonl (squtr.load_full_corpus -- 57,638
# docs for fiqa) WITHOUT ever importing/reading qrels (scoring reads qrels elsewhere, at eval time
# -- see squtr.load_full_corpus's own docstring for why this is a hard invariant, not an
# optimization). 'qrels-mini' is the LEGACY behavior this function used to be the only mode of
# (gold-linked docs + sampled distractors, CONDITIONED on the qrels/eval slice) -- kept as an
# explicit opt-in for DEV smoke only, never the default, and always manifest-marked as such (see
# QRELS_MINI_CREATED_NOTE_PREFIX below).
# ---------------------------------------------------------------------------------------------
CORPUS_MODES = ("full", "qrels-mini")

QRELS_MINI_CREATED_NOTE_PREFIX = "qrels-conditioned DEV smoke only — NOT evaluation-independent"


# ---------------------------------------------------------------------------------------------
# corpus-source audit axes (2026-07-13, ticket #38 item 2 -- F'-3 remediation; item 2 of the
# 2026-07-13 v4.2 doctoral review's remediation list ADDED a 6th axis, no_injected_gold_context --
# the tuple/function/dict-key names below deliberately keep their original "five" naming for
# call-site stability rather than a renaming churn; see corpus_five_axis_audit's own docstring for
# what each of the now-SIX axes checks). Replaces the single CLEAN/SUSPECT/LEAKAGE verdict for a
# CORPUS-TYPE KB source build, since a single verdict conflated several genuinely distinct
# questions.
# ---------------------------------------------------------------------------------------------
CORPUS_AUDIT_AXES = (
    "object_correct", "query_independent_corpus", "label_independent_build",
    "answer_presence_expected", "no_injected_gold_context", "provenance_complete",
)

# The manifest fields ticket #25 P1c (embedder_token/pool_split) and ticket #37 item 6
# (code_git_sha/build_hash) added -- "provenance_complete" checks these are all actually populated,
# never assumed present just because the code path CAN set them.
_PROVENANCE_REQUIRED_FIELDS = ("content_hash", "code_git_sha", "embedder_token", "pool_split", "build_hash")


def _provenance_complete(manifest: dict) -> bool:
    return all(manifest.get(f) not in (None, "") for f in _PROVENANCE_REQUIRED_FIELDS)


def _overall_corpus_audit_verdict(axes: dict) -> str:
    """Combine the five per-axis verdicts into one ``overall`` string. NEVER 'CLEAN' if any axis is
    'FAIL' or 'NOT_EVALUATED' (ticket #38 item 2's hard rule) -- 'CLEAN' requires all five PASS."""
    failing = sorted(k for k in CORPUS_AUDIT_AXES if axes.get(k) == "FAIL")
    not_evaluated = sorted(k for k in CORPUS_AUDIT_AXES if axes.get(k) == "NOT_EVALUATED")
    if failing:
        return f"FAIL({','.join(failing)})"
    if not_evaluated:
        return f"PASS-with-NOT_EVALUATED({','.join(not_evaluated)})"
    return "CLEAN"


def corpus_five_axis_audit(*, object_correct: bool, corpus_mode: str, label_independent_build: bool,
                            n_eval_golds_checked: int, answer_overlap_rate: float | None,
                            injected_gold_context_detected: bool,
                            corpus_lock_verification: dict | None,
                            manifest: dict) -> dict:
    """The (now SIX -- see ``CORPUS_AUDIT_AXES``'s own comment on the naming-stability decision)
    axis audit dict for a corpus-type KB source build. Returns one dict entry per
    ``CORPUS_AUDIT_AXES`` name plus ``overall`` (the combined verdict, see
    ``_overall_corpus_audit_verdict``); every axis except ``answer_presence_expected`` is one of
    ``"PASS"``/``"FAIL"``/``"NOT_EVALUATED"``.

      - ``object_correct`` -- is the persisted VALUE actually the corpus document's own text (never
        the query's, the P0-1 bug class this function's module docstring documents)? Passed in by
        the caller, which knows exactly what it assembled the record from.
      - ``query_independent_corpus`` -- (2026-07-13, ticket #38 item 1, F-6 remediation) does corpus
        CONSTRUCTION (which documents get indexed at all) depend on which queries/qrels exist? NO
        LONGER derived from the ``corpus_mode`` STRING alone (the v4.2 doctoral review's exact
        finding: "``query_independent_corpus = PASS iff corpus_mode == 'full'``" self-certifies from
        a mode string, not evidence -- an incomplete/altered/wrong corpus.jsonl could still pass).
        PASS now requires BOTH ``corpus_mode == 'full'`` AND
        ``corpus_lock_verification['verified'] is True`` -- the ACTUAL corpus this build embedded
        was checked byte-for-byte / doc-count / doc-id-order / normalized-content against the pinned
        ``docs/corpus.lock.json`` (see ``corpus_lock.verify_corpus_lock``, fail-closed). FAIL if the
        lock verification itself failed (missing lock file, doc-count mismatch, hash mismatch, ...)
        -- the ``corpus_lock_verification`` dict's own ``mismatches`` list carries the reason.
        ``corpus_mode == 'qrels-mini'`` is NEVER query-independent by construction (unchanged --
        gold-linked docs + distractors are CONDITIONED on the qrels/eval slice) -- FAIL regardless
        of ``corpus_lock_verification``.
      - ``label_independent_build`` -- was any gold ANSWER label used to DECIDE which documents are
        included (as opposed to only auditing VALUES, which never changes which docs are in the
        corpus)? Passed in by the caller.
      - ``answer_presence_expected`` -- (2026-07-13, ticket #38 item 2, F-7 remediation) NO LONGER a
        pass/fail derived from a post-SCRUB verdict -- open-corpus builds no longer scrub legal
        answer spans AT ALL (see ``build_squtr_corpus_source``'s docstring: a real evidence corpus
        naturally containing the literal answer to a real question about it is EXPECTED and
        CORRECT, not a leak). Purely DESCRIPTIVE now, one of three values (never gates ``overall``
        except via the unchanged ``NOT_EVALUATED`` rule): ``"NOT_EVALUATED"`` when
        ``n_eval_golds_checked == 0`` (unchanged: 0 golds checked is NOT the same as "checked and
        found clean"); else ``"PRESENT"`` (``answer_overlap_rate > 0`` -- some provided gold's
        literal text occurs somewhere in this corpus) or ``"ABSENT"`` (``answer_overlap_rate == 0``)
        -- both describe the corpus, neither is ever a defect.
      - ``no_injected_gold_context`` -- (2026-07-13, ticket #38 item 2, F-7 remediation) the ONE
        axis that CAN still hard-FAIL on gold-answer presence: PASS unless the caller detected a
        PER-ITEM, dataset-AUTHORED "gold-context" document (constructed specifically FOR one eval
        item, as opposed to an organically shared official corpus document indexed independent of
        any single query) that contains that SAME item's own gold answer -- a real
        information-boundary leak, structurally distinct from ``answer_presence_expected``'s
        expected/benign natural overlap. ``build_squtr_corpus_source`` always passes
        ``injected_gold_context_detected=False`` (structurally not applicable to it -- see that
        function's docstring); this axis exists as general, independently-testable infrastructure
        for any future builder that DOES construct per-item context (see
        ``test_kb_gate.py``'s positive/negative golden tests for the mechanism itself).
      - ``provenance_complete`` -- see ``_provenance_complete``.
    """
    if corpus_mode == "full":
        clv = corpus_lock_verification or {}
        query_independent_corpus = "PASS" if clv.get("verified") else "FAIL"
    else:
        query_independent_corpus = "FAIL"

    if n_eval_golds_checked == 0:
        answer_presence_expected = "NOT_EVALUATED"
    else:
        answer_presence_expected = "PRESENT" if (answer_overlap_rate or 0) > 0 else "ABSENT"

    axes = {
        "object_correct": "PASS" if object_correct else "FAIL",
        "query_independent_corpus": query_independent_corpus,
        "label_independent_build": "PASS" if label_independent_build else "FAIL",
        "answer_presence_expected": answer_presence_expected,
        "no_injected_gold_context": "FAIL" if injected_gold_context_detected else "PASS",
        "provenance_complete": "PASS" if _provenance_complete(manifest) else "FAIL",
    }
    axes["overall"] = _overall_corpus_audit_verdict(axes)
    return axes


def _corpus_lock_verification(subset: str, corpus_mode: str, corpus_sample: list[dict]) -> dict:
    """F-6 (2026-07-13): the EVIDENCE behind ``query_independent_corpus`` -- never derived from the
    ``corpus_mode`` string alone. 'full' mode: verify ``corpus_sample`` (the docs this build
    actually just fetched/embedded) against the pinned umbrella ``docs/corpus.lock.json`` via
    ``corpus_lock.verify_corpus_lock`` (non-raising here — a lock-verification failure becomes a
    reported ``FAIL`` axis, not an unrecoverable exception; the UNCONDITIONAL refusal-to-persist for
    a wrong doc_count is the separate hard assert in ``build_squtr_corpus_source``, checked before
    this function is even called). 'qrels-mini' is NEVER query-independent by construction (see
    ``CORPUS_MODES``' module comment) -- lock verification is not even attempted for it.
    """
    if corpus_mode != "full":
        return {"attempted": False, "verified": False, "reason": "corpus_mode != 'full'"}

    import corpus_lock

    if subset not in corpus_lock.UPSTREAM_SOURCE:
        return {
            "attempted": False, "verified": False,
            "reason": f"no corpus_lock.UPSTREAM_SOURCE entry for subset={subset!r} -- lock "
                      "verification not possible for this subset yet.",
        }
    lock_path = corpus_lock.default_lock_path()
    result = corpus_lock.verify_corpus_lock(
        str(lock_path), subset, corpus_docs=corpus_sample, raise_on_mismatch=False,
    )
    result["attempted"] = True
    result["lock_path"] = str(lock_path)
    return result


def build_squtr_corpus_source(embedder: str, subset: str = "fiqa", noise_level: str = "clean",
                                n: int | None = 40, n_distractors: int = 200, seed: int = 20260705,
                                eval_golds: list[str] | None = None, note: str = "",
                                supersede: bool = False, corpus_mode: str = "full",
                                precomputed_keys: dict[str, object] | None = None) -> dict:
    """REAL (not plan-only) corpus-side KB build for squtr's evidence documents (2026-07-12, RI
    mechanical remediation item 9; corpus_mode/five-axis-audit added 2026-07-13, ticket #38 items
    1/2 -- F'-3 remediation, Decision-Log 续26).

    ``corpus_mode`` (default ``'full'``, ticket #38 item 1): see ``CORPUS_MODES``'s module comment.
    ``'full'`` calls ``squtr.load_full_corpus(subset)`` -- EVERY document in the official
    ``corpus.jsonl``, qrels/queries NEVER read on this code path. ``'qrels-mini'`` is the LEGACY
    behavior (gold-linked docs + ``n_distractors`` sampled distractors via
    ``squtr.load_squtr_retrieval``/``build_mini_corpus``) -- an explicit opt-in for DEV smoke only;
    its manifest's ``created_note`` is ALWAYS prefixed with
    ``QRELS_MINI_CREATED_NOTE_PREFIX`` ("qrels-conditioned DEV smoke only — NOT
    evaluation-independent"), so nobody downstream can mistake it for an evaluation-independent
    corpus build. ``noise_level``/``n``/``n_distractors`` are ONLY consulted in ``'qrels-mini'``
    mode (a 'full' corpus has no noise-level concept -- corpus documents are query-independent by
    construction; those args are silently ignored in 'full' mode, not an oversight).

    ``precomputed_keys`` (ticket #38 item 1): an optional ``{doc_id: vector}`` mapping -- lets the
    resumable batch-checkpointed launcher (``scripts/knowledge/build_full_corpus.sh`` /
    ``build_full_corpus.embed_full_corpus_checkpointed``) hand already-embedded vectors back in
    here so this function never re-embeds work the launcher already did (the SAME
    ``precomputed_key`` per-record mechanism ``kb_build.build_source`` already supports -- see its
    docstring). Raises ``ValueError`` if any corpus doc id is missing from the mapping.

    Fixes forensic finding 续15 P0-1: the earlier squtr "knowledge-passage" KB build's VALUE field
    was actually the FiQA QUERY's own text (an OBJECT-MISMATCH / wrong-field bug — see
    ``docs/claim_ledger.yaml`` entry ``C-PHASEA``), not the retrieved corpus document; the same bug
    class hit vocalbench-knowledge (its value was likewise the question's own text; "gold" building
    was globally scrubbed empty). This function fetches the ACTUAL qrels-linked corpus documents
    (``loaders/squtr.load_squtr_retrieval`` -> ``build_mini_corpus``: gold docs referenced by qrels
    + seeded sampled distractors) and uses THEIR OWN text as both the KEY and the VALUE — no query
    text anywhere in the persisted source.

    ## Key-modality decision (documented, per task requirement)

    squtr's corpus documents ship as TEXT ONLY (``corpus.jsonl``: ``{_id, title, text}``) — there
    is NO audio for a corpus document, only for QUERIES. So this source is **TEXT-KEYED**
    (``key_modality='text'``), unlike every other squtr/kb_batch_build source in this repo (all of
    which are audio-keyed, built via ``build_one``). This has a real retrieval-time consequence,
    stated here rather than left implicit:

      - For an OMNI-FAMILY asymmetric bi-encoder (``encode_query``/``encode_document`` — e.g.
        ``omni-embed-nemotron``, and the GPU-server ``lco-3b``/``lco-7b``/``qwen3-omni-own`` once
        buildable): the QUERY side stays AUDIO (squtr queries are audio-only), embedded via
        ``encode_query``, landing in the SAME cosine space as this source's TEXT-keyed
        ``encode_document`` keys — retrieval against this source is genuinely **CROSS-MODAL**
        (audio query vs text key). (Query-side wiring in ``kb_retrieve._query_embedder`` for this
        specific cross-modal case is NOT yet implemented — a known follow-up, out of this ticket's
        scope; see that module's docstring.)
      - For an ASR-CASCADE arm (query audio -> ASR transcript -> text), the query becomes TEXT (the
        ASR hypothesis), so retrieval against this text-keyed source is **TEXT-TEXT**, not
        cross-modal — a fair same-modality comparison against the omni-family cross-modal arms.
      - An embedder with NO text-embedding path at all (``kb_embed.can_embed_text`` is ``False``)
        CANNOT key this source; such an ``embedder`` value returns ``{"status":
        "ARM-BLOCKED-cross-modal", ...}`` below rather than raising or silently skipping — the
        blocked arm is a recorded outcome, not an omission. An embedder that COULD embed text but
        needs an unavailable GPU server (``kb_embed.EMBEDDERS[embedder]["needs_server"]``) instead
        returns ``{"status": "pending-GPU-window", ...}`` — a distinct, non-permanent status.

    ## Answer-presence audit (Information-Boundary Guard) -- open-corpus values are NEVER scrubbed

    2026-07-13 (ticket #38 item 2, F-7 remediation -- v4.2 doctoral review §3 F-7: the prior
    version of this section instructed scrubbing legal answer spans out of an OPEN corpus, which is
    exactly backwards for a query-independent evidence corpus). squtr's OWN qrels carry no literal
    "answer" text field (they are IR relevance judgments — ``(query-id, corpus-id, score)`` — not a
    QA answer span), so the P0-1 bug this function replaces was an OBJECT-MISMATCH bug (wrong field
    used as the value), not an eval-gold leak in the T7/M3 sense. A REAL open evidence corpus
    (``corpus_mode='full'``: literally every document in the official ``corpus.jsonl``) naturally,
    CORRECTLY contains the literal text of the answer to real questions about it — that is exactly
    what an evidence corpus is FOR, not a leak. Leakage in the T7/M3 sense is when TEST
    qrels/answers DECIDE what goes into the corpus/index/prompt/candidates (see
    ``corpus_five_axis_audit``'s ``label_independent_build``/``query_independent_corpus`` axes,
    which DO gate on that) — NOT when a legitimately-included document happens to contain an answer
    span.

    Consequently this function calls ``kb_build.build_source`` with ``scrub=False`` (values are
    persisted EXACTLY as fetched from the corpus, never mutated) and ``enforce_leakage_gate=False``
    (a ``LEAKAGE``/``SUSPECT`` verdict — expected and common for a real evidence corpus against real
    question golds — never raises and never requires ``force_persist``). The audit still ALWAYS
    RUNS when ``eval_golds`` is non-empty (``audit_golds is not None`` in
    ``kb_build.build_source``'s own contract) and is stamped on the manifest for inspection; pass
    ``eval_golds`` — the actual gold ANSWER strings of whatever downstream QA eval task will consume
    this corpus KB — to get a real, non-empty overlap report; leave it ``None``/empty (the Stage-1
    default for pure-IR retrieval use) for a trivial 0-golds-checked report. Either way the result is
    PURELY DESCRIPTIVE — see ``corpus_five_axis_audit``'s ``answer_presence_expected`` axis and the
    returned dict's ``answer_overlap_stats`` below, ALWAYS recorded even when 0 golds were checked.

    A genuinely PER-ITEM, dataset-AUTHORED "gold-context" document (constructed specifically FOR one
    eval item, unlike squtr's shared/official corpus docs) is a DIFFERENT, still hard-FAILing check
    — see ``corpus_five_axis_audit``'s ``no_injected_gold_context`` axis (this function always
    passes ``injected_gold_context_detected=False``: structurally not applicable here, since squtr
    corpus documents are addressed by their own official ``_id`` and shared across many
    queries/items via qrels, never authored per query).

    Returns one of:
      ``{"status": "built", "manifest": <SourceManifest dict>, "corpus_mode": ..., "n_gold_docs":
      ..., "n_corpus_docs": ..., "answer_overlap_stats": {...}, "five_axis_audit": {...},
      "corpus_lock_verification": {...}}``
      ``{"status": "ARM-BLOCKED-cross-modal", "embedder": ..., "reason": ...}``
      ``{"status": "pending-GPU-window", "embedder": ..., "reason": ...}``
    """
    import kb_embed

    if corpus_mode not in CORPUS_MODES:
        raise ValueError(
            f"kb_batch_build.build_squtr_corpus_source: corpus_mode={corpus_mode!r} not in "
            f"{CORPUS_MODES}"
        )

    meta = kb_embed.EMBEDDERS.get(embedder, {})
    if meta.get("needs_server"):
        return {
            "status": "pending-GPU-window",
            "embedder": embedder, "subset": subset, "noise_level": noise_level,
            "corpus_mode": corpus_mode,
            "reason": (
                f"{embedder!r} needs a resident llama-server ({meta.get('server_env')}) -- not "
                "attempted this session (GPU held by other work per CLAUDE.md). Build once a "
                "GPU/server window is available; kb_embed.embed_text does not yet have a text-only "
                "code path for the llama-server embedders either (only audio input is wired in "
                "_llama_server_embed) -- that would need its own follow-up even with a server up."
            ),
        }
    if not kb_embed.can_embed_text(embedder):
        return {
            "status": "ARM-BLOCKED-cross-modal",
            "embedder": embedder, "subset": subset, "noise_level": noise_level,
            "corpus_mode": corpus_mode,
            "reason": (
                f"{embedder!r} has no text-embedding path wired in kb_embed.embed_text -- cannot "
                "key a TEXT corpus with it (squtr corpus documents ship as text only, no audio -- "
                "see this function's key-modality docstring section). Recorded, not silently "
                "skipped."
            ),
        }

    import kb_build
    import squtr

    n_gold_docs: int | None
    n_distractor_docs: int | None
    if corpus_mode == "full":
        # 2026-07-13 (ticket #38 item 1): squtr.load_full_corpus NEVER touches
        # queries*.jsonl/qrels/test.jsonl -- corpus construction is unconditional on the eval slice.
        corpus_sample = squtr.load_full_corpus(subset)
        gold_docids = None       # deliberately never computed -- qrels untouched on this path
        n_gold_docs = None
        n_distractor_docs = None

        # --- F-6 (2026-07-13, v4.2 doctoral review item 1): HARD, pre-persist invariant -- an
        # incomplete or miscounted corpus must NEVER be persisted as a 'full' (query-independent)
        # source, regardless of what any caller's precomputed_keys mapping happens to cover. This
        # is a DEFENSE-IN-DEPTH check independent of build_full_corpus.py's own completeness
        # assertions (that launcher's `missing` check already refuses an incomplete CHECKPOINT --
        # this checks the CORPUS ITSELF, in case corpus_sample ever came back short/long for any
        # other reason: a loader bug, a corrupted/truncated local zip, a wrong subset, ...). ---
        import corpus_lock

        expected_doc_count = corpus_lock.EXPECTED_DOC_COUNT.get(subset)
        if expected_doc_count is not None and len(corpus_sample) != expected_doc_count:
            raise ValueError(
                f"kb_batch_build.build_squtr_corpus_source: corpus_mode='full' but "
                f"len(corpus_sample)={len(corpus_sample)} != "
                f"corpus_lock.EXPECTED_DOC_COUNT[{subset!r}]={expected_doc_count} -- refusing to "
                "persist an incomplete/miscounted corpus as a full (query-independent) source "
                "(F-6 remediation: 'incomplete checkpoint must never persist as a full source')."
            )
        if precomputed_keys is not None and expected_doc_count is not None:
            n_embedded = len({d["_id"] for d in corpus_sample} & set(precomputed_keys))
            if n_embedded != expected_doc_count:
                raise ValueError(
                    f"kb_batch_build.build_squtr_corpus_source: corpus_mode='full' but the "
                    f"embedded doc count (precomputed_keys covering corpus_sample)={n_embedded} != "
                    f"corpus_lock.EXPECTED_DOC_COUNT[{subset!r}]={expected_doc_count} -- refusing "
                    "to persist an incomplete embedding checkpoint as a full source."
                )
    else:  # 'qrels-mini' (legacy, explicit opt-in)
        retrieval = squtr.load_squtr_retrieval(subset=subset, noise_level=noise_level, n=n, seed=seed,
                                                n_distractors=n_distractors)
        corpus_sample = retrieval["corpus_sample"]
        qrels = retrieval["qrels"]
        gold_docids = {docid for _, docid, _ in qrels}
        n_gold_docs = len(gold_docids)
        n_distractor_docs = len(corpus_sample) - len(gold_docids & {d["_id"] for d in corpus_sample})

    doc_texts = [f"{d.get('title', '')} {d.get('text', '')}".strip() for d in corpus_sample]
    records = []
    for doc, text in zip(corpus_sample, doc_texts):
        records.append({
            "key_text": text,
            "value": text,
            "from_item_id": f"squtr-{subset}-{noise_level}|corpus|{doc['_id']}",
        })

    # precomputed_keys (2026-07-13, ticket #38 item 1): the checkpointed full-corpus launcher
    # (build_full_corpus.py) embeds OUTSIDE this function, in resumable batches, and hands the
    # finished vectors back in keyed by doc _id -- skip embedding here entirely when supplied.
    if precomputed_keys is not None:
        missing = [d["_id"] for d in corpus_sample if d["_id"] not in precomputed_keys]
        if missing:
            raise ValueError(
                f"kb_batch_build.build_squtr_corpus_source: precomputed_keys is missing "
                f"{len(missing)} corpus doc id(s) (first few: {missing[:5]}) out of "
                f"{len(corpus_sample)} total -- every corpus doc must have a precomputed vector."
            )
        for r, doc in zip(records, corpus_sample):
            r["precomputed_key"] = precomputed_keys[doc["_id"]]
    elif embedder == "glap":
        # GLAP needs an explicit source_lang (it does not auto-detect); nemotron's encode_document
        # does not take one. Precompute GLAP's keys here (so the correct source_lang is used) and
        # pass them through as `precomputed_key` -- kb_build.build_source then skips its own
        # embedding step entirely and uses `embedder` as the manifest's ename/token verbatim.
        lang = _SQUTR_SUBSET_GLAP_LANG.get(squtr.SUBSETS.get(subset, "en"), "eng_Latn")
        _, precomp, _fitted = kb_embed.embed_text(doc_texts, embedder="glap", source_lang=lang)
        for r, vec in zip(records, precomp):
            r["precomputed_key"] = vec

    audit_golds = list(eval_golds) if eval_golds else []
    n_corpus_docs = len(corpus_sample)
    if corpus_mode == "full":
        source = f"squtr-{subset}-full__corpus__{embedder}__text-keyed"
        dataset_label = f"squtr-{subset}-full-corpus"
        note_body = note or (
            f"kb_batch_build.build_squtr_corpus_source (ticket #38 item 1, 2026-07-13): squtr "
            f"{subset} FULL TEXT-keyed corpus -- every document in the official corpus.jsonl, "
            "qrels/queries NEVER read on this code path (see squtr.load_full_corpus). "
            f"corpus_mode=full n_corpus_docs={n_corpus_docs}."
        )
        final_note = note_body
    else:
        source = f"squtr-{subset}-{noise_level}__corpus__{embedder}__text-keyed"
        dataset_label = f"squtr-{subset}-{noise_level}-corpus"
        note_body = note or (
            f"kb_batch_build.build_squtr_corpus_source (RI item 9, 2026-07-12; corpus_mode="
            f"qrels-mini re-flagged 2026-07-13 ticket #38 item 1): squtr {subset}/{noise_level} "
            "TEXT-keyed corpus (fixes 续15 P0-1 object-mismatch -- values are the qrels-linked "
            "corpus documents themselves, NOT query text). "
            f"corpus_mode=qrels-mini n_gold_docs={n_gold_docs} n_corpus_docs={n_corpus_docs}."
        )
        final_note = f"{QRELS_MINI_CREATED_NOTE_PREFIX}. {note_body}"

    # F-7 (2026-07-13, ticket #38 item 2): scrub=False -- open-corpus values are NEVER scrubbed
    # (see this function's "Answer-presence audit" docstring section). enforce_leakage_gate=False
    # -- a LEAKAGE/SUSPECT verdict (expected/common for a real evidence corpus against real
    # question golds) never raises and never requires force_persist; the audit still runs and is
    # still stamped on the manifest (leakage_gate_enforced=False records that this was a
    # descriptive-only audit, not a hard gate -- see kb_build.build_source's docstring).
    manifest = kb_build.build_source(
        source, dataset_label, revision=None, records=records,
        key_modality="text", value_type="text-fact", embedder=embedder,
        audit_golds=audit_golds, scrub=False, enforce_leakage_gate=False,
        pool_split="test", supersede=supersede,
        note=final_note,
    )

    audit = manifest.get("leakage_audit", {}) or {}

    # F-6 (2026-07-13, item 1): the EVIDENCE query_independent_corpus derives from -- computed from
    # corpus_sample (the docs THIS build actually just fetched), never from the corpus_mode string.
    corpus_lock_verification = _corpus_lock_verification(subset, corpus_mode, corpus_sample)

    five_axis = corpus_five_axis_audit(
        # object_correct: records are built directly from corpus_sample's own title/text fields
        # above -- structurally never the query text (the P0-1 bug class this function replaces).
        object_correct=True,
        corpus_mode=corpus_mode,
        # label_independent_build: eval_golds is used ONLY for the DESCRIPTIVE overlap audit
        # (never scrubbed, never gates persistence) -- it never decides which documents are
        # included, in EITHER corpus_mode.
        label_independent_build=True,
        n_eval_golds_checked=len(audit_golds),
        answer_overlap_rate=audit.get("answer_overlap_rate"),
        # squtr corpus documents are shared/official (addressed by their own corpus _id, indexed
        # independent of any one query via qrels) -- structurally never a per-item authored
        # "gold-context" doc; see this function's docstring.
        injected_gold_context_detected=False,
        corpus_lock_verification=corpus_lock_verification,
        manifest=manifest,
    )

    return {
        "status": "built",
        "manifest": manifest,
        "corpus_mode": corpus_mode,
        "n_gold_docs": n_gold_docs,
        "n_corpus_docs": n_corpus_docs,
        "n_distractor_docs": n_distractor_docs,
        "answer_overlap_stats": {
            "n_eval_golds_checked": len(audit_golds),
            "verdict": audit.get("verdict"),
            "answer_overlap_rate": audit.get("answer_overlap_rate"),
            "n_golds_present": audit.get("n_golds_leaked"),
            "note": ("DESCRIPTIVE ONLY (F-7 remediation, 2026-07-13) -- open-corpus values are "
                     "NEVER scrubbed; natural answer-text presence in a real evidence corpus is "
                     "expected, not a leak. See five_axis_audit['answer_presence_expected']."),
        },
        "corpus_lock_verification": corpus_lock_verification,
        "five_axis_audit": five_axis,
    }


# ---- real build path (not exercised by --plan; used once a config is actually selected) --------

def _gold_string(gold) -> str:
    """Best-effort flatten of a loader Row's ``gold`` field to one string, whatever shape a given
    loader uses for it (bare string / dict / list-of-dict, e.g. squtr's qrels list)."""
    if gold is None:
        return ""
    if isinstance(gold, str):
        return gold
    if isinstance(gold, (list, tuple)):
        return "; ".join(_gold_string(g) for g in gold)
    if isinstance(gold, dict):
        return json.dumps(gold, ensure_ascii=False, sort_keys=True)
    return str(gold)


def _extract_value(row: dict, value_spec: str) -> str:
    """Pull the VALUE payload out of a loader Row for a given ``value_spec``.

    ``'text'`` -- ``row['meta']['text']``, the reference transcript/query text every loader in this
    repo's Row contract carries (per squtr.py's convention: ``meta={..., "text": q["text"]}``).
    ``'gold'`` -- a best-effort string form of ``row['gold']`` via ``_gold_string`` (NOTE: this is
    exactly the field the Information-Boundary Guard's leakage audit exists to check -- see
    ``build_one``, which always runs the audit and scrubs regardless of ``value_spec``).
    Anything else is treated as a literal key to pull out of ``row['meta']``.
    """
    if value_spec == "text":
        return str(row.get("meta", {}).get("text", ""))
    if value_spec == "gold":
        return _gold_string(row.get("gold"))
    return str(row.get("meta", {}).get(value_spec, ""))


# 2026-07-11 (ticket #25 P1a/P2d): key_org/value_org enumerations, matching
# ``scripts/baselines/run_mock.py``'s ``KEY_ORGS``/``VALUE_ORGS`` token spellings EXACTLY (source
# naming unification, P1a) but defined HERE (not imported from run_mock) -- scripts/knowledge must
# never depend on scripts/baselines (run_mock already depends on scripts/knowledge, the other
# direction; importing it back here would be a layering violation / risk a circular import).
KEY_ORGS = ("single-utt", "multi-granularity", "ha-multi-key", "hb-single-space")
VALUE_ORGS = ("knowledge-passage", "memory-instance", "exemplar", "struct-lite", "raptor-lite",
              "audio-text-hybrid")

# 'ha-multi-key' composite key components (content embedder = the caller's own `embedder` arg;
# speaker/emotion are fixed per kb_schema's H-a framing -- content/speaker/emotion, 2026-07-10
# q1q2 decision memo). See _key_org_ha_multi_key.
HA_MULTI_KEY_SPEAKER_EMBEDDER = "eres2netv2"
HA_MULTI_KEY_EMOTION_EMBEDDER = "emotion2vec-plus-large"


def _segments_scratch_dir(dataset_loader_key: str) -> str:
    """Persistent (NOT tmp) scratch dir for derived sub-segment wavs (multi-granularity key_org) --
    persistent because a built source's ``key_audio_ref`` must keep pointing at a real file for as
    long as the source itself is kept, not a tmp path that may vanish on reboot."""
    from kb_schema import kb_root

    d = kb_root() / "_derived_audio" / dataset_loader_key / "segments"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _split_into_halves(wav_path: str, out_dir: str, tag: str, n_segments: int = 2) -> list[tuple]:
    """Slice ``wav_path`` into ``n_segments`` equal-duration chunks, written to ``out_dir``.

    2026-07-11 (ticket #25 P2d, ``multi-granularity`` key_org): a real word-level key needs
    forced-alignment infra this repo does not have -- PRAGMATIC APPROXIMATION per the task brief:
    sliding sub-segment (here: 2 halves) child keys instead. Returns
    ``[(start_s, end_s, seg_wav_path), ...]`` in order.
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    n = len(y)
    seg_len = max(n // n_segments, 1)
    out = []
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_segments):
        start = i * seg_len
        end = n if i == n_segments - 1 else min((i + 1) * seg_len, n)
        seg_path = os.path.join(out_dir, f"{tag}_seg{i}.wav")
        sf.write(seg_path, y[start:end], sr)
        out.append((round(start / sr, 3), round(end / sr, 3), seg_path))
    return out


def _key_org_multi_granularity(embedder: str, dataset_loader_key: str, source: str,
                                rows: list[dict], records: list[dict]) -> list[dict]:
    """utt key (unchanged) + 2 half-span child ("segment") keys per row -- see
    ``_split_into_halves``'s docstring for the documented word-level approximation. Child records
    reuse the PARENT's own value (a sub-segment has no distinct transcript without alignment --
    also documented, not silently pretended to be a finer-grained label) and set
    ``key_granularity='segment'`` + ``parent_ref``/``start_s``/``end_s`` per ``kb_schema``.

    KNOWN LIMITATION (not fixed this task, flagged rather than silently shipped): ``parent_ref`` is
    computed here from the PARENT's PRE-SCRUB value (``kb_build.build_source`` only scrubs gold
    leakage AFTER this function returns). If a parent's value happens to contain its own gold
    answer as a substring (rare -- ``value_spec='text'`` extracts the loader's QUERY/transcript
    text, not the answer, so this coincidence needs the query text to itself contain the answer
    string), ``scrub_golds`` would change that value at persist time and the ACTUAL persisted
    parent kid (computed from the post-scrub value) would then differ from the ``parent_ref``
    stamped on its children here -- a traceability mismatch, not a retrieval-correctness one (the
    child's OWN key/value are unaffected either way). Fixing this properly needs
    ``audit_golds``/``scrub`` threaded into this function so it can pre-scrub before computing
    ``parent_ref`` -- not done here; see ticket #25 P2d follow-up.
    """
    import kb_embed
    from kb_schema import entry_id

    out_dir = _segments_scratch_dir(dataset_loader_key)
    utt_wavs, seg_specs = [], []  # seg_specs: (row_idx, seg_idx, start_s, end_s, seg_path)
    for i, row in enumerate(rows):
        utt_wavs.append(row["wav"])
        for seg_i, (s, e, seg_path) in enumerate(_split_into_halves(row["wav"], out_dir, tag=f"r{i}")):
            seg_specs.append((i, seg_i, s, e, seg_path))

    all_wavs = utt_wavs + [spec[4] for spec in seg_specs]
    _ename, keys = kb_embed.embed_audio(all_wavs, embedder=embedder)

    out = []
    for i, rec in enumerate(records):
        r2 = dict(rec)
        r2["precomputed_key"] = keys[i]
        out.append(r2)
    for j, (row_i, seg_i, s, e, seg_path) in enumerate(seg_specs):
        parent = records[row_i]
        parent_kid = entry_id(source, str(parent["key_audio_ref"]), parent["value"])
        out.append({
            "key_audio_ref": seg_path,
            "value": parent["value"],  # documented approximation, see docstring
            "from_item_id": parent["from_item_id"],
            "precomputed_key": keys[len(utt_wavs) + j],
            "key_granularity": "segment",
            "parent_ref": parent_kid,
            "start_s": s, "end_s": e,
        })
    return out


def _key_org_ha_multi_key(embedder: str, rows: list[dict], records: list[dict]) -> tuple[list[dict], str]:
    """H-a approximated as ONE composite key = concat(content, speaker, emotion) embeddings --
    see ``kb_embed.embed_audio_composite``'s docstring for why this is a documented simplification
    of "2-3 independently-searchable key spaces" (a true H-a needs per-space indices + a
    fusion/routing layer at retrieval time; out of scope for this Phase-A structural-coverage
    pass). Returns ``(records_with_precomputed_key, composite_embedder_token)``.
    """
    import kb_embed

    wav_list = [r["wav"] for r in rows]
    name, keys = kb_embed.embed_audio_composite(
        wav_list, [embedder, HA_MULTI_KEY_SPEAKER_EMBEDDER, HA_MULTI_KEY_EMOTION_EMBEDDER]
    )
    out = []
    for rec, vec in zip(records, keys):
        r2 = dict(rec)
        r2["precomputed_key"] = vec
        out.append(r2)
    return out, name


def _key_org_hb_single_space(embedder: str, rows: list[dict], records: list[dict],
                              n_readouts: int = 2) -> list[dict]:
    """H-b: ONE shared (omni) embedding space, ``n_readouts`` tagged readout rows per utt sharing
    the IDENTICAL key vector (genuinely "one space, multiple readouts", not multiple key spaces --
    contrast with ha-multi-key above). Readout 0 is the real extracted value; readout>=1 has no
    distinct label source in this pipeline, so it is a documented, visibly-tagged PLACEHOLDER
    (prefixed ``"[readoutN]"``) rather than a silently duplicated copy mistaken for a real second
    signal -- a genuine second readout (e.g. a distinct speaker/emotion tag) is future work.
    """
    import kb_embed

    wav_list = [r["wav"] for r in rows]
    _ename, keys = kb_embed.embed_audio(wav_list, embedder=embedder)
    out = []
    for rec, vec in zip(records, keys):
        for ro in range(n_readouts):
            r2 = dict(rec)
            r2["precomputed_key"] = vec
            r2["value"] = rec["value"] if ro == 0 else f"[readout{ro}] {rec['value']}"
            out.append(r2)
    return out


def _apply_key_org(key_org: str, embedder: str, dataset_loader_key: str, source: str,
                    rows: list[dict], records: list[dict]) -> tuple[list[dict], str]:
    """Dispatch on ``key_org`` -> ``(final_records, embedder_token_for_build_source)``. The
    returned token is what actually gets stamped as ``manifest.embedder_token`` (P1c) -- identical
    to the caller's own ``embedder`` arg except for ``'ha-multi-key'``'s composite token.
    """
    if key_org == "single-utt":
        return records, embedder
    if key_org == "multi-granularity":
        return _key_org_multi_granularity(embedder, dataset_loader_key, source, rows, records), embedder
    if key_org == "ha-multi-key":
        return _key_org_ha_multi_key(embedder, rows, records)
    if key_org == "hb-single-space":
        return _key_org_hb_single_space(embedder, rows, records), embedder
    raise ValueError(f"kb_batch_build: unknown key_org {key_org!r} (expected one of {KEY_ORGS})")


def _value_org_raptor_lite(embedder: str, source: str, records: list[dict], seed: int,
                            n_clusters: int | None = None) -> list[dict]:
    """2-level RAPTOR-lite: leaf rows (unchanged) + synthetic cluster-summary rows.

    Cluster MEMBERSHIP is decided by simple k-means over VALUE-TEXT embeddings (semantic grouping
    of the payload -- per the task brief), using ``kb_embed.embed_text(embedder='auto')`` (MiniLM ->
    TF-IDF, CPU-only, no GPU/LLM). Each summary row's retrieval KEY is the CENTROID of its cluster
    members' own AUDIO key vectors (re-using ``precomputed_key`` if the key_org stage already
    computed one, else embedding fresh here) -- so it lands in the SAME audio-key cosine space as
    every leaf row, searchable by the same index. The summary VALUE is a concatenated-truncated
    join of (up to 5) member texts -- no LLM call, hence "-lite". Summary rows are tagged via a
    ``"[raptor-summary ...]"`` value-text prefix (no schema field for this; see comment below) and
    ``from_item_id=None`` (they don't correspond to one eval item).
    """
    import numpy as np
    from sklearn.cluster import KMeans

    import kb_embed

    key_refs = [r.get("key_audio_ref") for r in records]
    values = [r.get("value", "") for r in records]
    audio_keys = [r.get("precomputed_key") for r in records]
    need_embed_idx = [i for i, v in enumerate(audio_keys) if v is None]
    if need_embed_idx:
        _n, embedded = kb_embed.embed_audio([key_refs[i] for i in need_embed_idx], embedder=embedder)
        for j, i in enumerate(need_embed_idx):
            audio_keys[i] = np.asarray(embedded[j], dtype="float32")
    audio_keys = np.stack([np.asarray(v, dtype="float32") for v in audio_keys])

    _tname, value_vecs, _fitted = kb_embed.embed_text(values, embedder="auto")
    k = n_clusters or max(2, min(8, len(records) // 3 or 1))
    k = max(1, min(k, len(records)))
    if k == 1:
        labels = np.zeros(len(records), dtype=int)
    else:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(value_vecs).labels_

    out = [dict(r, precomputed_key=audio_keys[i]) for i, r in enumerate(records)]
    for cid in sorted(set(labels.tolist())):
        member_idx = [i for i, lab in enumerate(labels) if lab == cid]
        if not member_idx:
            continue
        texts = [values[i][:80] for i in member_idx[:5]]
        summary_text = " | ".join(texts)[:300]
        centroid = audio_keys[member_idx].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        out.append({
            "key_audio_ref": f"raptor-cluster:{source}:{cid}",  # synthetic ref, no playable file
            "value": f"[raptor-summary cluster={cid} n={len(member_idx)}] {summary_text}",
            "from_item_id": None,
            "precomputed_key": centroid.astype("float32"),
            "grain": records[member_idx[0]].get("grain"),
        })
    return out


def _apply_value_org(value_org: str, embedder: str, source: str, records: list[dict],
                      seed: int) -> list[dict]:
    """Dispatch on ``value_org`` -> final ``records`` (grain-tagged / RAPTOR-augmented)."""
    if value_org == "knowledge-passage":
        return records
    if value_org == "memory-instance":
        return [dict(r, grain="memory") for r in records]
    if value_org == "exemplar":
        return [dict(r, grain="exemplar") for r in records]
    if value_org == "audio-text-hybrid":
        # storage-identical to knowledge-passage: key_audio_ref already IS the source audio ref
        # (this is an audio-keyed KB by construction), so "value carries both text and the source
        # audio ref" is already true of every row here. The hybrid-vs-text-only DELIVERY decision
        # (does the backbone additionally receive the retrieved passage's own audio?) is a
        # run_mock/render_delivery-time concern, not a storage concern -- see run_mock.py.
        return [dict(r) for r in records]
    if value_org in ("raptor-lite", "struct-lite"):
        # 'struct-lite' currently ALIASES 'raptor-lite' -- the grid draft's §6.2 "RAPTOR-lite OR
        # HippoRAG-lite, 二选一" sign-off item is still open; 'struct-lite' is kept as a separately
        # enumerated token (matching the doc's "(4+2 对照)" = 6 value_org arms, see
        # run_mock.VALUE_ORGS's comment) so a future HippoRAG-lite build can occupy this slot
        # distinctly without renumbering the grid, rather than silently deleting the placeholder.
        return _value_org_raptor_lite(embedder, source, records, seed)
    raise ValueError(f"kb_batch_build: unknown value_org {value_org!r} (expected one of {VALUE_ORGS})")


def build_one(embedder: str, dataset_loader_key: str, pool_split: str, value_spec: str = "text",
              key_org: str = "single-utt", value_org: str = "knowledge-passage",
              n: int | None = None, seed: int = 20260705, note: str = "",
              eval_manifest: list[str] | None = None) -> dict:
    """Build ONE persisted source: (dataset_loader_key, embedder, key_org, value_org) ->
    ``kb_build.build_source(...)``. The CLEAN leakage-verdict enforcement gate applies exactly as
    it does for every other kb_build caller (``kb_schema.KBLeakageError`` on a confirmed LEAKAGE
    verdict unless ``force_persist`` -- not exposed here on purpose, this is a batch pipeline, not a
    debug escape hatch). The audit ALWAYS runs (``audit_golds`` is always populated from each row's
    ``gold`` field) and ``scrub=True`` always -- a batch build must never silently persist a leak.

    2026-07-11 (ticket #25):
    P1a -- the source NAME is now ``f"{dataset_loader_key}__{embedder}__{key_org}__{value_org}"``,
    the SAME 4-field convention ``run_mock.source_name_for`` already used (they were mismatched
    before this change: this used to be ``f"{dataset}__{embedder}__{pool_split}"``, a 3-field name
    with ``pool_split`` baked in where the runner expected ``key_org``/``value_org`` -- so a runner
    query for e.g. ``"squtr__glap__single-utt__knowledge-passage"`` could never find a source this
    function built as ``"squtr__glap__dev"``). ``pool_split`` moves into the manifest instead (see
    ``kb_build.build_source``/``kb_schema.SourceManifest``). MIGRATION: any source built by a
    PRE-ticket-#25 ``kb_batch_build`` is named under the OLD 3-field convention and will not be
    found by the new 4-field lookups -- there is no in-place rename shim (a source is fully
    reproducible from its build args), so REBUILD any such source with this version to pick up the
    new name (and the new ``embedder_token``/``pool_split`` manifest fields, P1c).

    P2d -- ``key_org``/``value_org`` drive real (not PLAN-ONLY) construction; see
    ``_apply_key_org``/``_apply_value_org`` and each helper's own docstring for the specific,
    documented approximation each arm makes.

    P3g -- ``eval_manifest`` (optional list of eval item ids): machine-verifies this build's own
    item ids (``from_item_id``, i.e. the ids of the rows the KB is BUILT from) are disjoint from
    ``eval_manifest`` -- raises ``ValueError`` if any id appears in both, rather than silently
    building a source that could hand a retrieval-time model its own eval item back as "knowledge".

    ``dataset_loader_key`` must be a key in ``scripts/loaders/registry.py``'s ``LOADERS`` table.
    ``embedder`` must be a key in ``kb_embed.EMBEDDERS``. ``pool_split``/``n``/``seed`` forward to
    the loader's standard ``load_<name>(split, n, seed) -> list[Row]`` contract.
    """
    import kb_build
    import kb_embed

    registry = _registry()
    if dataset_loader_key not in registry.LOADERS:
        raise KeyError(
            f"kb_batch_build.build_one: {dataset_loader_key!r} not in registry.LOADERS "
            f"({len(registry.LOADERS)} registered). Import failures: {registry.IMPORT_ERRORS or 'none'}"
        )
    if embedder not in kb_embed.EMBEDDERS:
        raise KeyError(f"kb_batch_build.build_one: {embedder!r} not in kb_embed.EMBEDDERS "
                        f"(choose one of {sorted(kb_embed.EMBEDDERS)})")
    if key_org not in KEY_ORGS:
        raise ValueError(f"kb_batch_build.build_one: key_org={key_org!r} not in {KEY_ORGS}")
    if value_org not in VALUE_ORGS:
        raise ValueError(f"kb_batch_build.build_one: value_org={value_org!r} not in {VALUE_ORGS}")

    rows = registry.LOADERS[dataset_loader_key](split=pool_split, n=n, seed=seed)
    records, audit_golds = [], []
    for r in rows:
        records.append({
            "key_audio_ref": r["wav"],
            "value": _extract_value(r, value_spec),
            "from_item_id": r.get("meta", {}).get("item_id"),
        })
        audit_golds.append(_gold_string(r.get("gold")))

    # --- P3g: machine-verified source/eval disjointness ---
    if eval_manifest is not None:
        source_ids = {rec["from_item_id"] for rec in records if rec["from_item_id"] is not None}
        overlap = source_ids & set(eval_manifest)
        if overlap:
            raise ValueError(
                f"kb_batch_build.build_one: {len(overlap)} item id(s) appear in BOTH the KB build "
                f"pool and eval_manifest (first few: {sorted(overlap)[:5]}) -- refusing to build a "
                "source that could hand a retrieval-time model its own eval item back as "
                "'knowledge'. Pass a disjoint pool_split/n/seed that excludes the eval slice."
            )

    source = f"{dataset_loader_key}__{embedder}__{key_org}__{value_org}"
    records, embedder_for_build = _apply_key_org(key_org, embedder, dataset_loader_key, source, rows, records)
    records = _apply_value_org(value_org, embedder, source, records, seed)

    from kb_schema import VALUE_TYPES

    value_type = value_spec if value_spec in VALUE_TYPES else "text-fact"
    return kb_build.build_source(
        source, dataset_loader_key, revision=None, records=records,
        key_modality="audio", value_type=value_type, embedder=embedder_for_build,
        audit_golds=audit_golds, scrub=True, pool_split=pool_split,
        note=note or f"kb_batch_build: {dataset_loader_key}/{pool_split} x {embedder} "
                     f"(key_org={key_org}, value_org={value_org}, value_spec={value_spec})",
    )


# ---------------------------------------------------------------------------------------------
# pseudo-question key synthesis (2026-07-13, M1 engineering-base item 3; owner directive 续21-A①
# "生成桥接匹配几何": doc2query-style offline key synthesis + query-time HyDE, S1 split into S1a
# same-distribution frozen matching / S1b generation-bridge-vs-trained-retriever). ``key_form=
# 'pseudo-question'``: for each evidence VALUE, generate k pseudo-questions a user might ask to
# retrieve it (doc2query), embed EACH question as its OWN key row pointing back at the SAME value
# (q2q matching geometry -- many keys, one value) -- as opposed to ``key_form='direct'``
# (``build_one``'s existing behavior: one audio-embedding key per row).
#
# **Black-box contract** (Decision-Log 续18, 2026-07-12: "系统接口契约唯一 = 音频/文本进、文本出、
# 多次采样" -- no white-box hook): pseudo-question generation is a plain TEXT-IN/TEXT-OUT call
# against the frozen core's resident llama-server (``run_baseline.BACKBONES``), reusing that
# module's server/env resolution but NOT its audio-attaching ``gen_llamacpp`` (there is no audio
# here -- the model is asked to read a TEXT passage and write TEXT questions).
# ---------------------------------------------------------------------------------------------

KEY_FORMS = ("direct", "pseudo-question")

PSEUDO_QUESTION_K_DEFAULT = 3
# Brand-new, never-reused-elsewhere seed constant for this generation path (mirrors this repo's
# date-coded seed convention -- SLICE_SEED=20260705, NOISE_SEED=20260712, ... -- distinct from all
# of them so a pseudo-question build can never be confused with an unrelated draw).
PSEUDO_QUESTION_SEED_DEFAULT = 20260713

PSEUDO_QUESTION_PROMPT_TEMPLATE = (
    "You are generating short SEARCH QUERIES a user might type to find the passage below. Read "
    "the passage, then output exactly {k} distinct, concise questions (one per line, no numbering "
    "or bullets) that this passage directly answers. Do not answer the questions, only write them.\n\n"
    "Passage:\n{passage}\n\nQuestions:"
)


def _parse_pseudo_questions(raw_text: str, k: int) -> list[str]:
    """Best-effort parse of the model's raw generation into <=k question strings: split on
    newlines, strip common enumeration prefixes ("1.", "-", "*", "Q:"), drop empties, dedupe
    (order-preserving), truncate to k. Never pads/fabricates a short output -- if the model
    returns fewer than k parseable lines, that shortfall is recorded (n_generated < k), never
    silently backfilled with duplicates."""
    import re

    out, seen = [], set()
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.):]|Q\d*[:.])\s*", "", line).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= k:
            break
    return out


def _llama_server_generate_text(base_url: str, prompt: str, seed: int, temperature: float = 0.7,
                                  max_tokens: int = 200, timeout: int = 300) -> str:
    """Plain TEXT-IN/TEXT-OUT call against a resident llama-server's OpenAI-compatible
    ``/v1/chat/completions`` endpoint -- the SAME endpoint ``run_baseline.gen_llamacpp`` posts to,
    minus the ``input_audio`` content block (there is no audio for a doc2query generation call).
    """
    import json as _json
    import urllib.request

    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "seed": seed, "temperature": temperature}
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=_json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def _default_generate_fn(backbone: str):
    """The REAL generate callable: resolves ``backbone``'s resident llama-server URL via
    ``run_baseline.BACKBONES`` (never hardcodes a port) and posts through
    ``_llama_server_generate_text``. Lazy -- only imported/invoked when pseudo-question synthesis
    actually runs for real (never at module import time, and never by the fake-model test, which
    injects its own ``generate_fn``)."""
    def _gen(prompt: str, seed: int, temperature: float = 0.7, max_tokens: int = 200) -> str:
        import run_baseline as rb  # scripts/baselines/run_baseline.py

        cfg = rb.BACKBONES[backbone]
        base_url = os.environ.get(cfg["server_env"], cfg["default_url"])
        return _llama_server_generate_text(base_url, prompt, seed, temperature, max_tokens)
    return _gen


def synthesize_pseudo_question_keys(
    evidence_records: list[dict], k: int = PSEUDO_QUESTION_K_DEFAULT,
    backbone: str = "qwen3-omni-30b-gguf", seed: int = PSEUDO_QUESTION_SEED_DEFAULT,
    generate_fn=None, temperature: float = 0.7,
) -> tuple[list[dict], dict]:
    """For each ``{"value": <evidence text>, "from_item_id": ...}`` in ``evidence_records``,
    generate up to ``k`` pseudo-questions via ``generate_fn`` (defaults to the real
    ``_default_generate_fn(backbone)`` -- a fake/test caller injects its own ``generate_fn(prompt,
    seed, temperature=..., max_tokens=...) -> str`` instead, the ONE seam this function depends on,
    matching the black-box text-in/text-out contract exactly so a fake stands in for the real
    server with zero other code path differences).

    Per-value generation seed = ``seed + value_index`` (deterministic, distinct per value, no two
    values share a generation call). Returns ``(pseudo_question_records, provenance)`` where the
    first element is a flat list of ``{"key_text": <question>, "value": <SAME evidence value>,
    "from_item_id": f"{parent_id}|pq{i}"}`` rows ready for ``kb_build.build_source(key_modality=
    'text', ...)``, and ``provenance`` carries the exact prompt template, k, backbone, seed, and
    (2026-07-13) the per-value generated question TEXTS themselves -- everything a manifest note
    needs to make this build's generation step fully auditable.
    """
    gen = generate_fn or _default_generate_fn(backbone)
    out: list[dict] = []
    per_value_questions: list[dict] = []
    for vi, rec in enumerate(evidence_records):
        value_text = rec["value"]
        parent_id = rec.get("from_item_id")
        value_seed = seed + vi
        prompt = PSEUDO_QUESTION_PROMPT_TEMPLATE.format(k=k, passage=value_text)
        raw = gen(prompt, value_seed, temperature=temperature)
        questions = _parse_pseudo_questions(raw, k)
        per_value_questions.append({
            "from_item_id": parent_id, "generation_seed": value_seed,
            "n_generated": len(questions), "questions": questions,
        })
        for qi, q in enumerate(questions):
            out.append({
                "key_text": q,
                "value": value_text,
                "from_item_id": f"{parent_id}|pq{qi}",
            })
    provenance = {
        "k_requested": k, "backbone": backbone, "seed_base": seed, "temperature": temperature,
        "prompt_template": PSEUDO_QUESTION_PROMPT_TEMPLATE,
        "generation_contract": ("black-box text-in/text-out (Decision-Log 续18) -- offline, "
                                 "build-time only; per-value seed = seed_base + value_index"),
        "n_values": len(evidence_records), "n_synthesized_keys": len(out),
        "per_value_questions": per_value_questions,
    }
    return out, provenance


# 2026-07-13 (ticket #37 item 5): embedders with a documented (or plausibly future) AUDIO-QUERY
# path into a TEXT-keyed index -- mirrors kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS's key set
# exactly (that table's "supported"/"pending-live-verification"/"pending-GPU-window" statuses are
# ALL for these tokens). 'auto' (MiniLM->TF-IDF) and every other kb_embed.TEXT_KEY_CAPABLE token
# are TEXT-ONLY -- no audio-query bridge exists (or is even planned) for them at all, so a
# pseudo-question q2q index built with one of them can NEVER be reached by an audio-query arm (the
# index gets built, but is unreachable cross-modally -- exactly the silent trap this item closes).
AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS = {"glap", "omni-embed-nemotron", "lco-3b", "lco-7b", "qwen3-omni-own"}


def build_pseudo_question_source(
    *,
    corpus_records: list[dict] | None = None,
    corpus_source: str | None = None,
    dataset_label: str | None = None,
    text_embedder: str = "auto",
    k: int = PSEUDO_QUESTION_K_DEFAULT,
    backbone: str = "qwen3-omni-30b-gguf", seed: int = PSEUDO_QUESTION_SEED_DEFAULT,
    n: int | None = None, generate_fn=None, note: str = "",
    eval_golds: list[str] | None = None,
    eval_manifest: list[str] | None = None, supersede: bool = False,
    value_type: str | None = None, pool_split: str = "dev",
    allow_text_only_query: bool = False, query_paths: list[str] | None = None,
    **_removed_kwargs,
) -> dict:
    """``key_form='pseudo-question'`` build: CORPUS-DOCUMENT evidence VALUES ->
    ``synthesize_pseudo_question_keys`` -> TEXT-embed each synthesized question (``text_embedder``)
    -> ``kb_build.build_source(key_modality='text', ...)``, gated by the SAME CLEAN
    leakage-verdict enforcement every other ``kb_build`` caller goes through.

    ## Input: corpus-manifest-only (2026-07-13, ticket #37 item 2 P0 fix)

    The OLD ``dataset_loader_key=`` path (pulling rows straight from
    ``scripts/loaders/registry.py``'s generic ``LOADERS`` table) has been REMOVED — it repeated the
    P0-1 wrong-object bug: for a REAL eval dataset (e.g. squtr) ``row['meta']['text']`` is the
    EVALUATION QUERY/transcript, not a corpus document, so a pseudo-question build synthesized from
    it would generate doc2query keys whose "document" is actually someone's query (see
    ``build_squtr_corpus_source``'s own module docstring for the earlier incarnation of this same
    bug class). Pass EXACTLY ONE of:

      ``corpus_records`` — a list of PRE-AUDITED, in-memory corpus-document records, each
      ``{"value": <evidence text>, "from_item_id": <id>}`` — the SAME shape
      ``build_squtr_corpus_source`` builds internally (evidence docs from qrels, i.e. the CORPUS
      side of a retrieval dataset, never a query).

      ``corpus_source`` — the NAME of an EXISTING persisted, text-keyed, CLEAN KB source (e.g. one
      built by ``build_squtr_corpus_source``) whose VALUES are corpus documents; this function
      loads it via ``kb_schema.load_values`` and uses each row's ``.value``/
      ``.provenance['from_item_id']`` as the evidence record. Raises ``ValueError`` if the
      referenced source is audio-keyed or its ``leakage_audit`` verdict is not CLEAN.

    Passing the old ``dataset_loader_key``/``value_spec`` (or any other unrecognized keyword) is a
    HARD TypeError with a pointer to the two arguments above — never silently misinterpreted.

    ## No silent 'auto' embedder (2026-07-13, ticket #37 item 5 P0 fix)

    ``text_embedder`` must be one of ``AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS`` (``glap`` /
    ``omni-embed-nemotron`` / ``lco-3b`` / ``lco-7b`` / ``qwen3-omni-own`` — mirrors
    ``kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS``'s key set) UNLESS the caller passes
    ``allow_text_only_query=True`` — an explicit escape hatch for a build that is intentionally
    text-query-only (e.g. an ASR-cascade arm, or a same-modality text-text comparison). ``'auto'``
    (MiniLM -> TF-IDF) in particular is a HARD ERROR without the escape hatch: neither tier has ANY
    audio-query bridge, so a q2q index built with it is constructed but UNREACHABLE by an
    audio-query arm — silently wasted work that looks like a working retrieval arm until someone
    actually tries to query it with audio. The escape-hatch's intentional restriction is recorded
    in the manifest's ``created_note`` as ``query_paths=['text']`` (vs. ``['audio', 'text']`` for
    an audio-query-capable build) — see the note-assembly code below.

    Source name: ``f"{dataset_label}__{text_embedder}__pseudo-question-k{k}__{value_type}"`` --
    deliberately NOT one of ``build_one``'s 4-field ``(dataset, embedder, key_org, value_org)``
    names, so it can never collide with an existing ``build_one`` source. ``dataset_label``
    defaults to ``corpus_source`` (when given) or ``"corpus_records"`` (when not) — pass it
    explicitly for a more descriptive source name.

    Provenance (task brief: "generation prompt + seeds in manifest"): the manifest's
    ``created_note`` embeds the exact prompt template, ``k``, ``backbone``, ``seed_base``, and a
    per-value log of {from_item_id, generation_seed, n_generated, questions} -- everything needed
    to audit exactly what was generated and with which seed, without re-running the generation.
    ``content_hash`` (kb_schema, RI item 8) already covers the synthesized keys' VALUES bytes as
    persisted in ``values.jsonl`` (each pseudo-question row's ``value`` = the parent evidence
    text) and, per this task's schema addition, each row's raw key TEXT itself
    (``KnowledgeValue.key_text_ref`` — see ``kb_schema.py``), so the synthesized keys are covered
    by the SAME content-addressed fingerprint every other source gets, not a bespoke one.
    """
    if _removed_kwargs:
        bad = sorted(_removed_kwargs)
        raise TypeError(
            f"kb_batch_build.build_pseudo_question_source: unexpected argument(s) {bad} -- if you "
            "meant 'dataset_loader_key'/'value_spec' (the OLD generic scripts/loaders/registry.py "
            "pull), that path was REMOVED (2026-07-13, ticket #37 item 2): for a REAL eval dataset "
            "(e.g. squtr) row['meta']['text'] is the EVALUATION QUERY, not a corpus document -- the "
            "exact P0-1 wrong-object bug this removal closes. Pass corpus_records=[{'value':..., "
            "'from_item_id':...}, ...] (pre-audited corpus documents) or corpus_source=<a persisted "
            "corpus-side KB source name> instead."
        )

    import kb_build
    import kb_embed

    if not kb_embed.can_embed_text(text_embedder):
        raise ValueError(
            f"kb_batch_build.build_pseudo_question_source: text_embedder={text_embedder!r} has no "
            f"text-embedding path wired (kb_embed.can_embed_text) -- pseudo-question KEYS are text, "
            "so this must be one of kb_embed.TEXT_KEY_CAPABLE."
        )
    if text_embedder not in AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS and not allow_text_only_query:
        raise ValueError(
            f"kb_batch_build.build_pseudo_question_source: text_embedder={text_embedder!r} has no "
            f"audio-query bridge (not in AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS="
            f"{sorted(AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS)}) -- ticket #37 item 5. A pseudo-question "
            "q2q index built with it can be searched by a TEXT query only ('auto' in particular "
            "silently falls back MiniLM->TF-IDF, neither of which any audio query can search at "
            "all) -- an audio-query arm could never reach it, even though the index builds "
            "successfully and looks like a working retrieval arm. Pass an embedder from that set "
            "for an audio-query-capable build, or explicitly opt into a text-only-query build via "
            "allow_text_only_query=True (records query_paths=['text'] in the manifest note, making "
            "the restriction an intentional, auditable choice rather than a silent trap)."
        )
    if (corpus_records is None) == (corpus_source is None):
        raise ValueError(
            "kb_batch_build.build_pseudo_question_source: pass EXACTLY ONE of corpus_records= "
            "(pre-audited in-memory corpus-document records) or corpus_source= (an existing "
            "persisted, text-keyed, CLEAN corpus-side KB source name) -- never both, never neither."
        )

    if corpus_source is not None:
        from kb_schema import leakage_verdict, load_manifest, load_values

        src_manifest = load_manifest(corpus_source)
        if src_manifest.key_modality != "text":
            raise ValueError(
                f"kb_batch_build.build_pseudo_question_source: corpus_source={corpus_source!r} is "
                f"not text-keyed (key_modality={src_manifest.key_modality!r}) -- a pseudo-question "
                "build synthesizes TEXT questions over CORPUS DOCUMENT text, so the seed source "
                "must be a text-keyed corpus-side source (e.g. built by "
                "kb_batch_build.build_squtr_corpus_source)."
            )
        verdict = leakage_verdict(src_manifest.leakage_audit)
        if verdict != "CLEAN":
            raise ValueError(
                f"kb_batch_build.build_pseudo_question_source: corpus_source={corpus_source!r} "
                f"leakage_audit verdict={verdict!r} (not CLEAN) -- refusing to seed a pseudo-"
                "question build from an unaudited/leaking corpus source."
            )
        src_values = load_values(corpus_source)
        if n is not None:
            src_values = src_values[:n]
        evidence_records = [
            {"value": v.value, "from_item_id": v.provenance.get("from_item_id")}
            for v in src_values
        ]
        resolved_dataset_label = dataset_label or src_manifest.dataset
        resolved_value_type = value_type or src_manifest.value_type
    else:
        evidence_records = list(corpus_records)
        if not evidence_records:
            raise ValueError("kb_batch_build.build_pseudo_question_source: corpus_records is empty")
        for i, r in enumerate(evidence_records):
            if "value" not in r:
                raise ValueError(
                    f"kb_batch_build.build_pseudo_question_source: corpus_records[{i}] missing "
                    "'value' -- expected {'value': <evidence text>, 'from_item_id': <id>} rows "
                    "(the same shape build_squtr_corpus_source produces)."
                )
        if n is not None:
            evidence_records = evidence_records[:n]
        resolved_dataset_label = dataset_label or "corpus_records"
        resolved_value_type = value_type or "text-fact"

    # audit_golds: corpus documents carry no per-row QA gold of their own by contract (only
    # value/from_item_id -- see build_squtr_corpus_source's "squtr's OWN qrels carry no literal
    # answer text field" note for why this is a real, not an oversight); a caller with a REAL
    # downstream QA gold to scrub the synthesized values against passes eval_golds= explicitly
    # (1:1 with evidence_records, '' where not applicable) -- mirrors build_squtr_corpus_source's
    # own eval_golds parameter. The audit still always runs (0 non-empty golds -> trivially CLEAN).
    golds_in = list(eval_golds) if eval_golds else ["" for _ in evidence_records]
    if len(golds_in) != len(evidence_records):
        raise ValueError(
            f"kb_batch_build.build_pseudo_question_source: eval_golds has {len(golds_in)} entries "
            f"but there are {len(evidence_records)} evidence records -- must be 1:1 (pass '' for an "
            "evidence record with no applicable gold)."
        )

    # ---- P3g: machine-verified source/eval disjointness (unchanged behavior/semantics) ----
    if eval_manifest is not None:
        source_ids = {rec["from_item_id"] for rec in evidence_records if rec["from_item_id"] is not None}
        overlap = source_ids & set(eval_manifest)
        if overlap:
            raise ValueError(
                f"kb_batch_build.build_pseudo_question_source: {len(overlap)} item id(s) appear in "
                f"BOTH the KB build pool and eval_manifest (first few: {sorted(overlap)[:5]}) -- "
                "refusing to build a source that could hand a retrieval-time model its own eval "
                "item back as 'knowledge'."
            )

    pq_records, generation_provenance = synthesize_pseudo_question_keys(
        evidence_records, k=k, backbone=backbone, seed=seed, generate_fn=generate_fn,
    )
    n_by_evidence_idx = [pq["n_generated"] for pq in generation_provenance["per_value_questions"]]
    pq_audit_golds = [g for g, n_gen in zip(golds_in, n_by_evidence_idx) for _ in range(n_gen)]

    from kb_schema import VALUE_TYPES

    final_value_type = resolved_value_type if resolved_value_type in VALUE_TYPES else "text-fact"
    resolved_query_paths = ["text"] if allow_text_only_query else ["audio", "text"]
    source = f"{resolved_dataset_label}__{text_embedder}__pseudo-question-k{k}__{final_value_type}"
    note_full = note or (
        f"kb_batch_build.build_pseudo_question_source (续21-A① q2q keys, ticket #37 item 2/5 fix, "
        f"2026-07-13): seeded from corpus_source={corpus_source!r} "
        f"(corpus_records_n={len(evidence_records) if corpus_source is None else None}) -- "
        f"{generation_provenance['n_values']} evidence values x k={k} pseudo-questions/value "
        f"(frozen core={backbone!r}, black-box generate mode, seed_base={seed}) = "
        f"{generation_provenance['n_synthesized_keys']} synthesized q2q keys. "
        f"query_paths={resolved_query_paths!r} (allow_text_only_query={allow_text_only_query!r}). "
        f"prompt_template={PSEUDO_QUESTION_PROMPT_TEMPLATE!r}. "
        f"per_value_questions={json.dumps(generation_provenance['per_value_questions'], ensure_ascii=False)}"
    )
    manifest = kb_build.build_source(
        source, resolved_dataset_label, revision=None, records=pq_records,
        key_modality="text", value_type=final_value_type, embedder=text_embedder,
        audit_golds=pq_audit_golds, scrub=True, pool_split=pool_split, supersede=supersede,
        note=note_full,
    )
    return {
        "status": "built", "manifest": manifest,
        "n_evidence_values": generation_provenance["n_values"],
        "n_synthesized_keys": generation_provenance["n_synthesized_keys"],
        "generation_provenance": generation_provenance,
        "corpus_source": corpus_source,
        "query_paths": resolved_query_paths,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true", help="print the build matrix; build nothing")
    ap.add_argument("--squtr-preview", action="store_true",
                    help="with --plan: also preview the squtr mini-corpus value-spec path (plan-mode only)")
    ap.add_argument("--embedder", help="a kb_embed.EMBEDDERS key")
    ap.add_argument("--dataset", help="a registry.LOADERS key")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--value-spec", default="text")
    ap.add_argument("--key-org", default="single-utt", choices=KEY_ORGS)
    ap.add_argument("--value-org", default="knowledge-passage", choices=VALUE_ORGS)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--key-form", default="direct", choices=KEY_FORMS,
                    help="'pseudo-question' builds via build_pseudo_question_source instead of "
                         "build_one -- --embedder is then read as the TEXT embedder for the "
                         "synthesized question keys, and --key-org/--value-org/--dataset are "
                         "ignored (use --corpus-source instead -- ticket #37 item 2 removed the "
                         "old --dataset generic-loader path for this build type)")
    ap.add_argument("--corpus-source", default=None,
                    help="an EXISTING persisted, text-keyed, CLEAN corpus-side KB source (e.g. "
                         "built by kb_batch_build.build_squtr_corpus_source) to seed pseudo-"
                         "question generation from -- REQUIRED for --key-form pseudo-question")
    ap.add_argument("--pq-k", type=int, default=PSEUDO_QUESTION_K_DEFAULT)
    ap.add_argument("--pq-backbone", default="qwen3-omni-30b-gguf")
    ap.add_argument("--allow-text-only-query", action="store_true",
                     help="ticket #37 item 5 escape hatch: allow --embedder to be a text-only "
                          "(no audio-query bridge) embedder for a pseudo-question build, e.g. "
                          "'auto' -- otherwise a HARD ERROR")
    args = ap.parse_args()

    if args.key_form == "pseudo-question":
        if args.plan or not (args.embedder and args.corpus_source):
            print_plan()
            if args.squtr_preview:
                print("\n=== squtr mini-corpus preview (PLAN ONLY -- nothing built) ===")
                print(json.dumps(plan_squtr_mini_corpus(), indent=2, ensure_ascii=False))
            if not args.plan:
                print("\n--key-form pseudo-question requires --embedder and --corpus-source "
                      "(the old --dataset generic-loader path was removed, ticket #37 item 2) -- "
                      "nothing built.")
            return 0
        result = build_pseudo_question_source(
            corpus_source=args.corpus_source, text_embedder=args.embedder, pool_split=args.split,
            k=args.pq_k, backbone=args.pq_backbone, seed=args.seed, n=args.n,
            allow_text_only_query=args.allow_text_only_query,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.plan or not (args.embedder and args.dataset):
        print_plan()
        if args.squtr_preview:
            print("\n=== squtr mini-corpus preview (PLAN ONLY -- nothing built) ===")
            print(json.dumps(plan_squtr_mini_corpus(), indent=2, ensure_ascii=False))
        return 0

    manifest = build_one(args.embedder, args.dataset, args.split, args.value_spec,
                          key_org=args.key_org, value_org=args.value_org, n=args.n, seed=args.seed)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

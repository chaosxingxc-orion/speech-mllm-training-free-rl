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


def build_squtr_corpus_source(embedder: str, subset: str = "fiqa", noise_level: str = "clean",
                                n: int | None = 40, n_distractors: int = 200, seed: int = 20260705,
                                eval_golds: list[str] | None = None, note: str = "",
                                supersede: bool = False) -> dict:
    """REAL (not plan-only) corpus-side KB build for squtr's qrels documents (2026-07-12, RI
    mechanical remediation item 9).

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

    ## Gold-scrub (Information-Boundary Guard)

    squtr's OWN qrels carry no literal "answer" text field (they are IR relevance judgments —
    ``(query-id, corpus-id, score)`` — not a QA answer span), so the P0-1 bug this function replaces
    was an OBJECT-MISMATCH bug (wrong field used as the value), not an eval-gold leak in the
    T7/M3 sense. This function nonetheless exercises the SAME ``kb_build.build_source``
    ``audit_golds``/``scrub=True`` pipeline every other source in this repo goes through, as a
    matter of consistent Information-Boundary discipline (an unaudited source is never treated as
    admissible elsewhere in this codebase, and this build is not an exception): pass
    ``eval_golds`` — the actual gold ANSWER strings of whatever downstream QA eval task will
    consume this corpus KB (e.g. heysquad/SQuAD-zh answer spans, if this corpus backs a QA gate
    like ``t7_rag_gate_probe.py``'s design) — to gold-scrub the VALUES against them (qrels-positive
    docs CAN by construction literally contain the answer text to their linked query, since FiQA-style
    queries are themselves questions). For pure-IR retrieval use (the Stage-1 default), leave
    ``eval_golds`` as ``None``/empty, which still runs the audit (0 golds -> trivially CLEAN)
    rather than skipping it outright. Scrub stats are ALWAYS recorded in the returned dict's
    ``gold_scrub_stats``, even when 0 golds were checked.

    Returns one of:
      ``{"status": "built", "manifest": <SourceManifest dict>, "n_gold_docs": ..., "n_corpus_docs":
      ..., "gold_scrub_stats": {...}}``
      ``{"status": "ARM-BLOCKED-cross-modal", "embedder": ..., "reason": ...}``
      ``{"status": "pending-GPU-window", "embedder": ..., "reason": ...}``
    """
    import kb_embed

    meta = kb_embed.EMBEDDERS.get(embedder, {})
    if meta.get("needs_server"):
        return {
            "status": "pending-GPU-window",
            "embedder": embedder, "subset": subset, "noise_level": noise_level,
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
            "reason": (
                f"{embedder!r} has no text-embedding path wired in kb_embed.embed_text -- cannot "
                "key a TEXT corpus with it (squtr corpus documents ship as text only, no audio -- "
                "see this function's key-modality docstring section). Recorded, not silently "
                "skipped."
            ),
        }

    import kb_build
    import squtr

    retrieval = squtr.load_squtr_retrieval(subset=subset, noise_level=noise_level, n=n, seed=seed,
                                            n_distractors=n_distractors)
    corpus_sample = retrieval["corpus_sample"]
    qrels = retrieval["qrels"]
    gold_docids = {docid for _, docid, _ in qrels}

    doc_texts = [f"{d.get('title', '')} {d.get('text', '')}".strip() for d in corpus_sample]
    records = []
    for doc, text in zip(corpus_sample, doc_texts):
        records.append({
            "key_text": text,
            "value": text,
            "from_item_id": f"squtr-{subset}-{noise_level}|corpus|{doc['_id']}",
        })

    # GLAP needs an explicit source_lang (it does not auto-detect); nemotron's encode_document does
    # not take one. Precompute GLAP's keys here (so the correct source_lang is used) and pass them
    # through as `precomputed_key` -- kb_build.build_source then skips its own embedding step
    # entirely and uses `embedder` as the manifest's ename/token verbatim (see its "every row
    # arrived precomputed" branch).
    if embedder == "glap":
        lang = _SQUTR_SUBSET_GLAP_LANG.get(squtr.SUBSETS.get(subset, "en"), "eng_Latn")
        _, precomp, _fitted = kb_embed.embed_text(doc_texts, embedder="glap", source_lang=lang)
        for r, vec in zip(records, precomp):
            r["precomputed_key"] = vec

    audit_golds = list(eval_golds) if eval_golds else []
    source = f"squtr-{subset}-{noise_level}__corpus__{embedder}__text-keyed"
    manifest = kb_build.build_source(
        source, f"squtr-{subset}-{noise_level}-corpus", revision=None, records=records,
        key_modality="text", value_type="text-fact", embedder=embedder,
        audit_golds=audit_golds, scrub=True, pool_split="test", supersede=supersede,
        note=note or (
            f"kb_batch_build.build_squtr_corpus_source (RI item 9, 2026-07-12): squtr {subset}/"
            f"{noise_level} TEXT-keyed corpus (fixes 续15 P0-1 object-mismatch -- values are the "
            "qrels-linked corpus documents themselves, NOT query text). "
            f"n_gold_docs={len(gold_docids)} n_corpus_docs={len(corpus_sample)}."
        ),
    )

    audit = manifest.get("leakage_audit", {}) or {}
    post = audit.get("post_scrub", audit)
    return {
        "status": "built",
        "manifest": manifest,
        "n_gold_docs": len(gold_docids),
        "n_corpus_docs": len(corpus_sample),
        "n_distractor_docs": len(corpus_sample) - len(gold_docids & {d["_id"] for d in corpus_sample}),
        "gold_scrub_stats": {
            "n_eval_golds_checked": len(audit_golds),
            "pre_scrub_verdict": audit.get("verdict"),
            "pre_scrub_answer_overlap_rate": audit.get("answer_overlap_rate"),
            "pre_scrub_n_golds_leaked": audit.get("n_golds_leaked"),
            "post_scrub_verdict": post.get("verdict"),
            "post_scrub_answer_overlap_rate": post.get("answer_overlap_rate"),
            "post_scrub_n_golds_leaked": post.get("n_golds_leaked"),
        },
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


def build_pseudo_question_source(
    dataset_loader_key: str, text_embedder: str = "auto", pool_split: str = "dev",
    value_spec: str = "text", k: int = PSEUDO_QUESTION_K_DEFAULT,
    backbone: str = "qwen3-omni-30b-gguf", seed: int = PSEUDO_QUESTION_SEED_DEFAULT,
    n: int | None = None, generate_fn=None, note: str = "",
    eval_manifest: list[str] | None = None, supersede: bool = False,
) -> dict:
    """``key_form='pseudo-question'`` build: loader rows -> evidence VALUES (reusing
    ``_extract_value``/``_gold_string``, same as ``build_one``) -> ``synthesize_pseudo_question_
    keys`` -> TEXT-embed each synthesized question (``text_embedder``, default ``'auto'`` = cheap
    CPU MiniLM/TF-IDF -- pseudo-question keys are TEXT, so this never needs a heavy audio
    embedder) -> ``kb_build.build_source(key_modality='text', ...)``, gated by the SAME CLEAN
    leakage-verdict enforcement every other ``kb_build`` caller goes through.

    Source name: ``f"{dataset_loader_key}__{text_embedder}__pseudo-question-k{k}__{value_type}"``
    -- deliberately NOT one of ``build_one``'s 4-field ``(dataset, embedder, key_org, value_org)``
    names (``"pseudo-question-k{k}"`` is not a ``KEY_ORGS`` token), so it can never collide with an
    existing ``build_one`` source. This is a NEW, additive capability (this build function, plus
    ``KEY_FORMS`` for documentation) -- it does NOT touch ``KEY_ORGS``/``VALUE_ORGS`` or
    ``run_mock.py``'s Phase-A grid; wiring a pseudo-question arm into that grid is a separate,
    owner-gated follow-up, not attempted here (task brief: "no bulk builds yet").

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
    import kb_build
    import kb_embed

    if not kb_embed.can_embed_text(text_embedder):
        raise ValueError(
            f"kb_batch_build.build_pseudo_question_source: text_embedder={text_embedder!r} has no "
            f"text-embedding path wired (kb_embed.can_embed_text) -- pseudo-question KEYS are text, "
            "so this must be one of kb_embed.TEXT_KEY_CAPABLE."
        )
    registry = _registry()
    if dataset_loader_key not in registry.LOADERS:
        raise KeyError(
            f"kb_batch_build.build_pseudo_question_source: {dataset_loader_key!r} not in "
            f"registry.LOADERS ({len(registry.LOADERS)} registered)."
        )

    rows = registry.LOADERS[dataset_loader_key](split=pool_split, n=n, seed=seed)
    evidence_records = []
    audit_golds = []
    for r in rows:
        evidence_records.append({
            "value": _extract_value(r, value_spec),
            "from_item_id": r.get("meta", {}).get("item_id"),
        })
        audit_golds.append(_gold_string(r.get("gold")))
    # audit_golds is per EVIDENCE row above; the actual persisted records (one per synthesized
    # question, k-per-evidence-row) repeat each evidence row's gold k times below so the audit
    # checks every persisted VALUE against its own row's gold, not a length-mismatched list.
    pq_records, generation_provenance = synthesize_pseudo_question_keys(
        evidence_records, k=k, backbone=backbone, seed=seed, generate_fn=generate_fn,
    )
    n_by_evidence_idx = [pq["n_generated"] for pq in generation_provenance["per_value_questions"]]
    pq_audit_golds = [g for g, n_gen in zip(audit_golds, n_by_evidence_idx) for _ in range(n_gen)]

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

    from kb_schema import VALUE_TYPES

    value_type = value_spec if value_spec in VALUE_TYPES else "text-fact"
    source = f"{dataset_loader_key}__{text_embedder}__pseudo-question-k{k}__{value_type}"
    note_full = note or (
        f"kb_batch_build.build_pseudo_question_source (续21-A① q2q keys, M1 item 3, 2026-07-13): "
        f"{dataset_loader_key}/{pool_split} -- {generation_provenance['n_values']} evidence values "
        f"x k={k} pseudo-questions/value (frozen core={backbone!r}, black-box generate mode, "
        f"seed_base={seed}) = {generation_provenance['n_synthesized_keys']} synthesized q2q keys. "
        f"prompt_template={PSEUDO_QUESTION_PROMPT_TEMPLATE!r}. "
        f"per_value_questions={json.dumps(generation_provenance['per_value_questions'], ensure_ascii=False)}"
    )
    manifest = kb_build.build_source(
        source, dataset_loader_key, revision=None, records=pq_records,
        key_modality="text", value_type=value_type, embedder=text_embedder,
        audit_golds=pq_audit_golds, scrub=True, pool_split=pool_split, supersede=supersede,
        note=note_full,
    )
    return {
        "status": "built", "manifest": manifest,
        "n_evidence_values": generation_provenance["n_values"],
        "n_synthesized_keys": generation_provenance["n_synthesized_keys"],
        "generation_provenance": generation_provenance,
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
                         "synthesized question keys, and --key-org/--value-org are ignored")
    ap.add_argument("--pq-k", type=int, default=PSEUDO_QUESTION_K_DEFAULT)
    ap.add_argument("--pq-backbone", default="qwen3-omni-30b-gguf")
    args = ap.parse_args()

    if args.plan or not (args.embedder and args.dataset):
        print_plan()
        if args.squtr_preview:
            print("\n=== squtr mini-corpus preview (PLAN ONLY -- nothing built) ===")
            print(json.dumps(plan_squtr_mini_corpus(), indent=2, ensure_ascii=False))
        return 0

    if args.key_form == "pseudo-question":
        result = build_pseudo_question_source(
            args.dataset, text_embedder=args.embedder, pool_split=args.split,
            value_spec=args.value_spec, k=args.pq_k, backbone=args.pq_backbone,
            seed=args.seed, n=args.n,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    manifest = build_one(args.embedder, args.dataset, args.split, args.value_spec,
                          key_org=args.key_org, value_org=args.value_org, n=args.n, seed=args.seed)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

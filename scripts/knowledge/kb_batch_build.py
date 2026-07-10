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
    args = ap.parse_args()

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

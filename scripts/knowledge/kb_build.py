"""kb_build — ingest a dataset into a PERSISTED speech-keyed source (KEY=audio, VALUE=payload).

Mainstream vector-DB shape: embed the KEY (audio) into a matrix, persist a vector index over it, and
store the VALUE (transcript / labels / answer / fact) row-aligned as the payload. Replaces the
transient TF-IDF text-passage pool. ``key_modality='text'`` is kept as a legacy path for
reading-comprehension passage pools, but AUDIO is the primary, multimodal key.

    build_source(source, dataset, revision, records, key_modality='audio', value_type='transcript', ...)
        records = [{ 'key_audio_ref': <wav path>,  'value': <payload>, 'from_item_id': <id> }, ...]
               or [{ 'key_text': <text key>,        'value': <payload>, 'from_item_id': <id> }, ...]  # legacy

    Step-2 schema evolution (2026-07-10; additive, optional per-record): each record may also carry
    ``key_granularity`` ('utterance' default | 'word' | 'segment'), ``parent_ref`` (parent
    utterance's ``kid``, for child/segment keys), ``start_s``/``end_s`` (child span offset within
    the parent utterance), and ``grain`` ('knowledge' | 'memory' | 'exemplar' | None). Omitted
    fields default exactly as ``kb_schema.KnowledgeValue`` does, so callers that don't need
    granularity/grain tagging are unaffected.

Leakage audit runs on the VALUES (value-side Information-Boundary Guard): if a value contains an eval
gold, the source is flagged; ``scrub=True`` strips golds from values before persisting.

**Enforcement gate**: when the audit's final verdict (post-scrub if scrubbed, else raw) is
``LEAKAGE``, ``build_source`` REFUSES to persist and raises ``kb_schema.KBLeakageError`` — unless the
caller passes ``force_persist=True``, an explicit PoC/debug escape hatch that logs a loud warning and
stamps the manifest ``forced=True`` so a leaking, force-built source is never mistaken for an
admissible one. ``SUSPECT`` and unaudited (``audit_golds=None``) builds are NOT gated — only a
confirmed ``LEAKAGE`` verdict blocks persistence.

2026-07-13 (ticket #38 item 2, F-7 remediation): ``enforce_leakage_gate=False`` (default ``True`` —
every pre-existing caller is unaffected) is a SEPARATE, DESCRIPTIVE-ONLY escape hatch from
``force_persist`` — it exists for a caller building a legitimate OPEN CORPUS (e.g.
``kb_batch_build.build_squtr_corpus_source``), where a value containing an eval gold's literal text
is EXPECTED and CORRECT (an open evidence corpus naturally contains the answers to real questions
about it — that is not leakage, see that function's module docstring), not a defect to gate on. With
``enforce_leakage_gate=False`` the audit still RUNS and is still stamped on the manifest (so a
descriptive overlap report stays available) but a ``LEAKAGE``/``SUSPECT`` verdict never raises and
never requires ``force_persist``; unlike ``force_persist=True`` this does NOT stamp
``manifest.forced=True`` (nothing was overridden — there was no gate to override for this call site
by design), it stamps ``manifest.leakage_gate_enforced=False`` instead, so a reader of the manifest
can always tell whether THIS build's audit was a hard gate or a descriptive-only report.

2026-07-11 (ticket #25): ``build_source`` now also (P1a) accepts ``pool_split`` (stored in the
manifest, no longer baked into the source name — see ``kb_schema.SourceManifest``), (P1c) stamps
``manifest.embedder_token`` (the registry token ``kb_retrieve`` must reuse for query-side
embedding — see ``kb_embed.resolve_embedder_token``), and (P2d) honors an optional per-record
``"precomputed_key"`` so a caller (``kb_batch_build``'s richer ``key_org``/``value_org`` arms) can
supply ready-made key vectors (composite multi-embedder keys, cluster-centroid summary keys, ...)
for some or all rows instead of every row being freshly embedded from ``key_audio_ref``.
"""
from __future__ import annotations

import json
import pickle

from kb_index import VectorIndex


def build_source(
    source: str,
    dataset: str,
    revision: str | None,
    records: list[dict],
    key_modality: str = "audio",
    value_type: str = "transcript",
    embedder: str = "auto",
    build_seed: int = 0,
    note: str = "",
    audit_golds: list[str] | None = None,
    scrub: bool = False,
    force_persist: bool = False,
    pool_split: str | None = None,
    aux_audit: bool = False,
    supersede: bool = False,
    enforce_leakage_gate: bool = True,
) -> dict:
    """Build a persisted speech-keyed knowledge source. Returns the SourceManifest as a dict.

    Raises ``kb_schema.KBLeakageError`` if the value-side audit's final verdict is ``LEAKAGE`` and
    ``force_persist`` is not set (see module docstring: the enforcement gate).

    Raises ``kb_schema.KBSourceExistsError`` (2026-07-12, RI item 8) if ``source`` already has a
    persisted build (an existing ``manifest.json``) and ``supersede`` is not set -- this function no
    longer silently overwrites an existing source in place (forensic finding 续15: "KB build_hash
    不含内容+原位覆盖"). Pass ``supersede=True`` to ARCHIVE (never delete) the existing build under
    ``<kb_root>/archive/<source>__superseded_<UTC-timestamp>/`` and proceed with a fresh build at
    the same ``source`` name; the new manifest's ``predecessor`` field records the archived path
    and the predecessor's own ``content_hash``. The other, non-destructive way to avoid this gate
    entirely is to build under a NEW, distinctly-versioned ``source`` name (e.g. ``f"{source}__v2"``)
    instead of reusing one that already exists.

    Every successful build now stamps ``manifest.content_hash`` (2026-07-12, RI item 8) --
    ``kb_schema.content_hash_of``: sha256 over the persisted ``values.jsonl``/``keys.npy`` bytes,
    the sorted ``from_item_id``s, and this repo's current git sha -- a content fingerprint that
    changes whenever the actual persisted bytes would, unlike ``build_hash`` (pure metadata).

    ``pool_split`` (2026-07-11, ticket #25 P1a) is stored in the manifest verbatim -- see
    ``kb_schema.SourceManifest``'s docstring for why this moved OUT of the source name.

    Each ``records`` entry may carry an optional ``"precomputed_key"`` (2026-07-11, ticket #25 P2d):
    a ready-made key vector (list/np.ndarray) for that row, used AS-IS instead of embedding
    ``key_audio_ref``/``key_text`` through ``kb_embed`` -- this is the substrate
    ``kb_batch_build``'s richer ``key_org``/``value_org`` arms build on (composite multi-embedder
    keys for 'ha-multi-key', cluster-centroid keys for 'raptor-lite' summary rows, ...). A row with
    ``precomputed_key`` still needs a non-``None`` ``key_field`` value (``key_audio_ref``/
    ``key_text``) for id/dedup/traceability -- it need not be a real playable file for such rows
    (e.g. a synthetic ``"raptor-cluster:<source>:<id>"`` string), since it is never passed to
    ``kb_embed``. Rows that mix (some precomputed, some not) are fully supported -- only the
    non-precomputed subset is ever sent to ``kb_embed.embed_audio``/``embed_text``, batched
    together as before.
    """
    import shutil
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np

    import kb_embed
    from kb_audit import audit_texts, scrub_golds
    from kb_schema import (
        CONTENT_HASH_SCHEMA_VERSION,
        KBLeakageError,
        KBSourceExistsError,
        KnowledgeValue,
        SourceManifest,
        build_hash,
        content_hash_of,
        entry_id,
        git_dirty_of,
        git_sha_of,
        kb_root,
        leakage_verdict,
        source_dir,
    )

    # --- refuse-overwrite gate (2026-07-12, RI item 8) -- checked BEFORE anything is embedded/
    # written, so a refusal never does wasted embedding work first ---
    predecessor_record = None
    existing_dir = source_dir(source)
    if (existing_dir / "manifest.json").exists():
        if not supersede:
            raise KBSourceExistsError(
                f"kb_build.build_source({source!r}): a build already exists at {existing_dir} "
                "(manifest.json present) -- refusing to overwrite it in place. Either build under "
                f"a NEW, distinctly-versioned source name (e.g. {source!r} + '__v2'), or pass "
                "supersede=True to archive (never delete) the existing build and record it as this "
                "build's predecessor."
            )
        old_manifest = json.load(open(existing_dir / "manifest.json", encoding="utf-8"))
        archived_dir = kb_root() / "archive" / f"{source}__superseded_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        archived_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(existing_dir), str(archived_dir))
        predecessor_record = {
            "source": source,
            "content_hash": old_manifest.get("content_hash"),
            "build_hash": old_manifest.get("build_hash"),
            "archived_path": str(archived_dir),
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"  [kb_build] {source}: superseded existing build -> archived to {archived_dir} "
              "(never deleted)", flush=True)

    # --- collect keys + values (dedup by (key_ref, value)) ---
    seen, key_refs, values, from_ids, grains, precomputed = set(), [], [], [], [], []
    key_field = "key_audio_ref" if key_modality == "audio" else "key_text"
    for r in records:
        kref, val = r.get(key_field), r.get("value", "")
        if kref is None:
            continue
        dedup = (kref, val)
        if dedup in seen:
            continue
        seen.add(dedup)
        key_refs.append(kref)
        values.append(val)
        from_ids.append(r.get("from_item_id"))
        precomputed.append(r.get("precomputed_key"))
        grains.append({
            "key_granularity": r.get("key_granularity", "utterance"),
            "parent_ref": r.get("parent_ref"),
            "start_s": r.get("start_s"),
            "end_s": r.get("end_s"),
            "grain": r.get("grain"),
        })

    # --- value-side leakage audit (does a stored value contain an eval gold?) ---
    # aux_audit (2026-07-11, ticket #25 P3h): opt-in only (default False) -- the embedding-
    # similarity aux hook (kb_audit.embedding_similarity_audit) may try a network-backed text
    # embedder (MiniLM) before its offline TF-IDF fallback, which would risk a hang on an
    # offline/CI box if this ran unconditionally on every build. The exact-substring hard gate
    # below is completely unaffected either way -- aux fields are purely additive diagnostics.
    audit, leakage_ok = {}, None
    if audit_golds is not None:
        audit = audit_texts(values, audit_golds, aux=aux_audit)
        if scrub:
            values = scrub_golds(values, audit_golds)
            audit = {**audit, "post_scrub": audit_texts(values, audit_golds, aux=aux_audit)}
        leakage_ok = audit.get("post_scrub", audit).get("verdict") == "CLEAN"

    # --- enforcement gate: refuse to PERSIST a confirmed-LEAKAGE source (Information-Boundary Guard) ---
    # 2026-07-13 (ticket #38 item 2, F-7): enforce_leakage_gate=False skips this entire gate (no
    # raise possible, ever) for a call site whose audit is DESCRIPTIVE-ONLY by design (an open
    # corpus, where gold-text presence is expected) -- see this function's module docstring.
    verdict = leakage_verdict(audit)
    forced = False
    if verdict == "LEAKAGE" and enforce_leakage_gate:
        if not force_persist:
            raise KBLeakageError(
                f"kb_build.build_source({source!r}): refusing to persist — leakage_audit verdict="
                f"LEAKAGE (answer_overlap_rate={audit.get('post_scrub', audit).get('answer_overlap_rate')}). "
                "Pass force_persist=True to override for PoC/debug only; the persisted manifest will "
                "be stamped forced=True and remains NOT admissible for a knowledge-utilization claim."
            )
        forced = True
        print(
            f"  [kb_build] WARNING: force-persisting LEAKAGE source {source!r} (force_persist=True) — "
            "manifest stamped forced=True; NOT admissible for a knowledge-utilization claim.",
            flush=True,
        )

    # --- embed the KEY into a matrix, build + persist the vector index ---
    # 2026-07-11 (ticket #25 P2d): only rows WITHOUT a precomputed_key are actually sent to
    # kb_embed -- batched together, exactly as before, so the common all-fresh-audio path (every
    # pre-existing caller) is unaffected. Rows WITH one are spliced back into the final matrix at
    # their original position, after a dim-consistency check against whatever WAS embedded (or, if
    # every row was precomputed, against each other) -- a build-time mirror of the query-time
    # dimension assertion ``kb_retrieve.retrieve`` now also enforces (P1c).
    fitted = None
    embed_idx = [i for i, p in enumerate(precomputed) if p is None]
    if embed_idx:
        sub_refs = [key_refs[i] for i in embed_idx]
        if key_modality == "audio":
            ename, embedded = kb_embed.embed_audio(sub_refs, embedder=embedder)
        else:  # legacy text key
            ename, embedded, fitted = kb_embed.embed_text(sub_refs, embedder=embedder)
        embedded = np.asarray(embedded, dtype="float32")
        dim = embedded.shape[1]
    else:
        # every row arrived precomputed (e.g. kb_batch_build's 'ha-multi-key' composite-key arm) --
        # nothing to embed here; the caller's own `embedder` string IS the descriptive name, since
        # this function never actually invoked kb_embed.
        ename = embedder
        embedded = None
        first = next(p for p in precomputed if p is not None)
        dim = np.asarray(first, dtype="float32").shape[0]

    keys = np.zeros((len(key_refs), dim), dtype="float32")
    j = 0
    for i in range(len(key_refs)):
        if precomputed[i] is None:
            keys[i] = embedded[j]
            j += 1
        else:
            pv = np.asarray(precomputed[i], dtype="float32")
            if pv.shape[0] != dim:
                raise ValueError(
                    f"kb_build.build_source({source!r}): record {i} precomputed_key dim "
                    f"{pv.shape[0]} != this source's embedded dim {dim} -- every row of ONE source "
                    "must share one key space."
                )
            keys[i] = pv
    embedder_token = kb_embed.resolve_embedder_token(embedder, ename)
    index = VectorIndex.build(keys)
    # 2026-07-13 (ticket #37 item 6): the ACTUAL construction params VectorIndex.build used --
    # kb_build never overrides its defaults today (always called as VectorIndex.build(keys), no
    # index_type/hnsw_m override), so "requested_index_type" is always "auto" for now; recorded
    # explicitly (not hardcoded elsewhere) so a future caller that DOES pass an override only needs
    # to update this one call site, and so index.backend's resulting string is never the ONLY
    # record of how the index was actually built.
    index_backend_params = {
        "requested_index_type": "auto",
        "hnsw_m": 32 if index.backend == "faiss-hnsw-ip" else None,
        "metric": "inner-product (cosine, since keys are L2-normalized)",
    }
    normalization = "l2"  # every kb_embed embedder L2-normalizes its output (see kb_embed._l2) --
    # explicit rather than assumed, see kb_schema.SourceManifest.normalization's docstring.
    embedder_revision = kb_embed.embedder_revision_of(embedder_token) if embedder_token else None

    d = source_dir(source)
    d.mkdir(parents=True, exist_ok=True)
    index.save(d)  # keys.npy (+ index.faiss)
    if fitted is not None and ename.startswith("tfidf"):
        pickle.dump(fitted, open(d / "retriever.pkl", "wb"))

    # --- write the VALUE store (payload), row-aligned to the key index ---
    with open(d / "values.jsonl", "w", encoding="utf-8") as fh:
        for i, (kref, val, fid, g) in enumerate(zip(key_refs, values, from_ids, grains)):
            kv = KnowledgeValue(
                row=i,
                kid=entry_id(source, str(kref), val),
                source=source,
                key_modality=key_modality,
                value_type=value_type,
                value=val,
                key_audio_ref=kref if key_modality == "audio" else None,
                key_text_ref=kref if key_modality == "text" else None,
                provenance={
                    "dataset": dataset,
                    "revision": revision,
                    "build_seed": build_seed,
                    "from_item_id": fid,
                    "leakage_checked": leakage_ok,
                },
                key_granularity=g["key_granularity"],
                parent_ref=g["parent_ref"],
                start_s=g["start_s"],
                end_s=g["end_s"],
                grain=g["grain"],
            )
            fh.write(kv.to_json() + "\n")

    # --- content_hash (2026-07-12, RI item 8; extended 2026-07-13, ticket #37 item 6): fingerprint
    # the ACTUAL PERSISTED BYTES + embedder identity + normalization + index-backend params, not
    # just values/keys/from_item_ids/git-sha -- computed AFTER values.jsonl/keys.npy are on disk,
    # over those real files. ---
    repo_root = Path(__file__).resolve().parents[2]  # W1 repo root
    code_git_sha = git_sha_of(repo_root)
    code_git_dirty = git_dirty_of(repo_root)
    c_hash = content_hash_of(
        d / "values.jsonl", d / "keys.npy", from_ids, code_git_sha,
        embedder_token=embedder_token, embedder_revision=embedder_revision,
        normalization=normalization, index_backend=index.backend,
        index_backend_params=index_backend_params,
    )

    manifest = SourceManifest(
        source=source,
        dataset=dataset,
        revision=revision,
        key_modality=key_modality,
        value_type=value_type,
        embedder=ename,
        dim=index.dim,
        n_entries=index.n,
        index_backend=index.backend,
        build_seed=build_seed,
        build_hash=build_hash(dataset, revision, key_modality, value_type, ename, build_seed, index.n),
        leakage_audit=audit,
        created_note=note,
        forced=forced,
        leakage_gate_enforced=enforce_leakage_gate,
        embedder_token=embedder_token,
        pool_split=pool_split,
        content_hash=c_hash,
        predecessor=predecessor_record,
        code_git_sha=code_git_sha,
        code_git_dirty=code_git_dirty,
        embedder_revision=embedder_revision,
        normalization=normalization,
        index_backend_params=index_backend_params,
        content_hash_schema_version=CONTENT_HASH_SCHEMA_VERSION,
    )
    json.dump(manifest.to_dict(), open(d / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(
        f"  [kb_build] {source}: key={key_modality} value={value_type} n={index.n} "
        f"embedder={ename} token={embedder_token} index={index.backend} hash={manifest.build_hash} "
        f"content_hash={c_hash[:16]} audit={verdict if verdict else 'n/a'}{' FORCED' if forced else ''}"
        f"{' SUPERSEDES-PRIOR' if predecessor_record else ''}",
        flush=True,
    )
    return manifest.to_dict()

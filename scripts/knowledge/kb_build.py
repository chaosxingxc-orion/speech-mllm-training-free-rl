"""kb_build — ingest a dataset into a PERSISTED speech-keyed source (KEY=audio, VALUE=payload).

Mainstream vector-DB shape: embed the KEY (audio) into a matrix, persist a vector index over it, and
store the VALUE (transcript / labels / answer / fact) row-aligned as the payload. Replaces the
transient TF-IDF text-passage pool. ``key_modality='text'`` is kept as a legacy path for
reading-comprehension passage pools, but AUDIO is the primary, multimodal key.

    build_source(source, dataset, revision, records, key_modality='audio', value_type='transcript', ...)
        records = [{ 'key_audio_ref': <wav path>,  'value': <payload>, 'from_item_id': <id> }, ...]
               or [{ 'key_text': <text key>,        'value': <payload>, 'from_item_id': <id> }, ...]  # legacy

Leakage audit runs on the VALUES (value-side Information-Boundary Guard): if a value contains an eval
gold, the source is flagged; ``scrub=True`` strips golds from values before persisting.

**Enforcement gate**: when the audit's final verdict (post-scrub if scrubbed, else raw) is
``LEAKAGE``, ``build_source`` REFUSES to persist and raises ``kb_schema.KBLeakageError`` — unless the
caller passes ``force_persist=True``, an explicit PoC/debug escape hatch that logs a loud warning and
stamps the manifest ``forced=True`` so a leaking, force-built source is never mistaken for an
admissible one. ``SUSPECT`` and unaudited (``audit_golds=None``) builds are NOT gated — only a
confirmed ``LEAKAGE`` verdict blocks persistence.
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
) -> dict:
    """Build a persisted speech-keyed knowledge source. Returns the SourceManifest as a dict.

    Raises ``kb_schema.KBLeakageError`` if the value-side audit's final verdict is ``LEAKAGE`` and
    ``force_persist`` is not set (see module docstring: the enforcement gate).
    """
    import numpy as np

    import kb_embed
    from kb_audit import audit_texts, scrub_golds
    from kb_schema import (
        KBLeakageError,
        KnowledgeValue,
        SourceManifest,
        build_hash,
        entry_id,
        leakage_verdict,
        source_dir,
    )

    # --- collect keys + values (dedup by (key_ref, value)) ---
    seen, key_refs, values, from_ids = set(), [], [], []
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

    # --- value-side leakage audit (does a stored value contain an eval gold?) ---
    audit, leakage_ok = {}, None
    if audit_golds is not None:
        audit = audit_texts(values, audit_golds)
        if scrub:
            values = scrub_golds(values, audit_golds)
            audit = {**audit, "post_scrub": audit_texts(values, audit_golds)}
        leakage_ok = audit.get("post_scrub", audit).get("verdict") == "CLEAN"

    # --- enforcement gate: refuse to PERSIST a confirmed-LEAKAGE source (Information-Boundary Guard) ---
    verdict = leakage_verdict(audit)
    forced = False
    if verdict == "LEAKAGE":
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
    fitted = None
    if key_modality == "audio":
        ename, keys = kb_embed.embed_audio(key_refs, embedder=embedder)
    else:  # legacy text key
        ename, keys, fitted = kb_embed.embed_text(key_refs, embedder=embedder)
    keys = np.asarray(keys, dtype="float32")
    index = VectorIndex.build(keys)

    d = source_dir(source)
    d.mkdir(parents=True, exist_ok=True)
    index.save(d)  # keys.npy (+ index.faiss)
    if fitted is not None and ename.startswith("tfidf"):
        pickle.dump(fitted, open(d / "retriever.pkl", "wb"))

    # --- write the VALUE store (payload), row-aligned to the key index ---
    with open(d / "values.jsonl", "w", encoding="utf-8") as fh:
        for i, (kref, val, fid) in enumerate(zip(key_refs, values, from_ids)):
            kv = KnowledgeValue(
                row=i,
                kid=entry_id(source, str(kref), val),
                source=source,
                key_modality=key_modality,
                value_type=value_type,
                value=val,
                key_audio_ref=kref if key_modality == "audio" else None,
                provenance={
                    "dataset": dataset,
                    "revision": revision,
                    "build_seed": build_seed,
                    "from_item_id": fid,
                    "leakage_checked": leakage_ok,
                },
            )
            fh.write(kv.to_json() + "\n")

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
    )
    json.dump(manifest.to_dict(), open(d / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(
        f"  [kb_build] {source}: key={key_modality} value={value_type} n={index.n} "
        f"embedder={ename} index={index.backend} hash={manifest.build_hash} "
        f"audit={verdict if verdict else 'n/a'}{' FORCED' if forced else ''}",
        flush=True,
    )
    return manifest.to_dict()

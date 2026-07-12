"""kb_retrieve — load a PERSISTED speech-keyed source and retrieve values by key similarity.

Reloads the persisted vector index (``kb_index.VectorIndex.load``: FAISS or numpy-flat) + the value
store, and re-instantiates the KEY embedder from the manifest — so retrieval reproduces without
re-embedding the corpus. Query flow: embed query AUDIO -> ANN search over key index -> return VALUES.
(For legacy text-keyed sources, the query is text and the persisted TF-IDF/MiniLM key embedder is used.)

**Enforcement gate**: ``load_source`` reads ``manifest.leakage_audit`` and raises
``kb_schema.KBLeakageError`` unless the final verdict is ``CLEAN`` (an unaudited source, verdict
``None``, is treated as NOT admissible either — audit silence is not a clean bill). The escape hatch
is ``allow_unclean=True`` (PoC/debug only), which logs a loud warning on every load.

**Cross-modal routing (2026-07-13, M1 engineering-base item 2 — fixes the RI item 9 follow-up
flagged below in the 2026-07-12 version of this docstring)**: ``load_source``'s default
``embed_query`` still resolves the query embedder from ``manifest.key_modality`` alone (audio-keyed
-> audio query; text-keyed -> SAME-MODALITY text query, now correctly wired for the omni-family
tokens too — see ``_query_embedder``'s text branch). For the genuinely CROSS-MODAL case
(``kb_batch_build.build_squtr_corpus_source``'s TEXT-keyed corpus sources, whose intended query
side for an omni-family asymmetric bi-encoder is an AUDIO query against those TEXT keys), use
``retrieve_cross_modal_audio`` below instead of ``retrieve``/``load_source``'s default
``embed_query`` — it routes per ``CROSS_MODAL_AUDIO_QUERY_STATUS`` and raises
``kb_schema.KBCrossModalBlockedError`` (status ``"ARM-BLOCKED-cross-modal"`` or
``"pending-GPU-window"``) cleanly for any embedder that cannot (yet) support it, rather than the
old silent misroute / opaque ``ValueError``.
"""
from __future__ import annotations

import pickle

# ---------------------------------------------------------------------------------------------
# cross-modal AUDIO-query -> TEXT-keyed-source routing table (2026-07-13, M1 item 2). Maps a
# source's manifest.embedder_token to whether an AUDIO query can be embedded into that TEXT-keyed
# source's key space, mirroring the status vocabulary kb_batch_build.build_squtr_corpus_source
# already uses at BUILD time for the identical embedders (the analogous decision at QUERY time):
#   "supported"           -- kb_embed.embed_query_crossmodal_audio has a real path for this token.
#   "pending-GPU-window"   -- the embedder's audio-query tower needs a resident llama-server not up
#                              this session (temporary — matches build-time's same status).
#   "ARM-BLOCKED-cross-modal" (the default for any token NOT listed here) -- no text tower / no
#                              audio-query path exists for this embedder at all (permanent).
# ---------------------------------------------------------------------------------------------
CROSS_MODAL_AUDIO_QUERY_STATUS: dict[str, str] = {
    "glap": "supported",                 # joint audio-text space (_glap_embed / _glap_embed_text)
    "omni-embed-nemotron": "supported",  # asymmetric encode_query accepts {"audio": ...} too (see
                                          # kb_embed._omni_embed_query_audio's docstring caveat)
    "lco-3b": "pending-GPU-window",
    "lco-7b": "pending-GPU-window",
    "qwen3-omni-own": "pending-GPU-window",
}

# manifest.embedder_token (the DOCUMENT/KEY-side registry token) -> the kb_embed.embed_text
# `embedder=` argument to use for a SAME-MODALITY (text) query against a text-keyed source built
# with that token. For a true asymmetric bi-encoder the query tower is a DIFFERENT call from the
# document tower even when both land in the same cosine space -- nemotron's document-side token
# ("omni-embed-nemotron", _omni_embed_document_text) must be queried via its QUERY-side token
# ("omni-embed", _omni_embed_query), never itself. GLAP is a genuinely SYMMETRIC joint space (like
# CLAP) -- same call either side, so it maps to itself.
_TEXT_KEY_SAME_MODALITY_QUERY_TOKEN = {
    "omni-embed-nemotron": "omni-embed",
    "glap": "glap",
}


def _l2(X):
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def _query_embedder(manifest, source_path):
    """Rebuild the KEY embedder so queries land in the SAME space as the stored keys.

    Correctness: retrieval MUST use the exact embedder the source was BUILT with — querying a
    logmel-keyed KB with CLAP (or vice-versa) compares vectors from different spaces.

    2026-07-11 (ticket #25 P1c fix): this used to guess the embedder ARG back from
    ``manifest.embedder`` (the descriptive ``ename``, e.g. ``"glap:GLAP"``) via fragile prefix
    matching that defaulted to ``'auto'`` whenever the guess didn't match one of 3 hardcoded
    prefixes — silently landing a query on CLAP for any source built with any of the 14 unified
    embedders (glap/meralion-se2/wavlm-*/eres2netv2/campplus/emotion2vec-*/dasheng/clsp/sense/
    sensevoice-small/lco-3b/lco-7b/qwen3-omni-own), a real correctness bug (comparing CLAP query
    vectors against e.g. GLAP-space keys). Now ``manifest.embedder_token`` — the exact registry
    token ``kb_build`` stamped at build time (see its docstring) — is used directly, and this
    function RAISES rather than falling back to CLAP/'auto' when no token is available and
    ``kb_embed.infer_embedder_token`` can't recover one from ``manifest.embedder`` either (only
    possible for a source built before ``embedder_token`` existed).
    """
    if manifest.key_modality == "audio":
        import kb_embed

        token = manifest.embedder_token or kb_embed.infer_embedder_token(manifest.embedder)
        if not token:
            raise ValueError(
                f"kb_retrieve._query_embedder: cannot determine the query embedder for source "
                f"{manifest.source!r} (manifest.embedder={manifest.embedder!r}, no embedder_token "
                "recorded and kb_embed.infer_embedder_token could not recover one either) -- "
                "refusing to silently fall back to CLAP/'auto' (ticket #25 P1c). Rebuild this "
                "source with a current kb_build so embedder_token is recorded."
            )
        if token.startswith("composite:"):
            parts = token.split(":", 1)[1].split("+")
            return lambda qs: kb_embed.embed_audio_composite(qs, parts)[1]
        return lambda qs: kb_embed.embed_audio(qs, embedder=token)[1]
    # text-keyed source: legacy TF-IDF/MiniLM passage pool, OR (2026-07-13, M1 item 2 fix) one of
    # kb_batch_build.build_squtr_corpus_source's newer omni-family TEXT-keyed corpus builds queried
    # SAME-MODALITY (a text query, e.g. an ASR-cascade hypothesis — see that function's docstring's
    # "TEXT-TEXT, not cross-modal" case). Before this fix, ANY such source (manifest.embedder like
    # "glap:GLAP" / "omni-embed-nemotron:...") fell through to the final `raise ValueError` below —
    # `load_source` could not even be called on one of these sources at all, cross-modal or not.
    if manifest.embedder.startswith("tfidf"):
        v = pickle.load(open(source_path / "retriever.pkl", "rb"))
        return lambda qs: _l2(v.transform(list(qs)).toarray())
    if manifest.embedder.startswith("minilm"):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(manifest.embedder.split(":", 1)[1], device="cpu")
        return lambda qs: _l2(m.encode(list(qs), normalize_embeddings=True))
    import kb_embed

    token = manifest.embedder_token or kb_embed.infer_embedder_token(manifest.embedder)
    query_token = _TEXT_KEY_SAME_MODALITY_QUERY_TOKEN.get(token) if token else None
    if query_token:
        return lambda qs: kb_embed.embed_text(list(qs), embedder=query_token)[1]
    raise ValueError(
        f"unknown embedder {manifest.embedder!r} (token={token!r}) -- no SAME-MODALITY text-query "
        "path wired for this text-keyed source. If this source's intended query is AUDIO (the "
        "cross-modal case), use kb_retrieve.retrieve_cross_modal_audio instead of load_source's "
        "default embed_query."
    )


def load_source(source: str, allow_unclean: bool = False) -> dict:
    """Return {index, values, manifest, embed_query} for a persisted source.

    Raises ``kb_schema.KBLeakageError`` unless ``manifest.leakage_audit``'s final verdict is
    ``CLEAN`` — unless ``allow_unclean=True`` (PoC/debug escape hatch; logs a loud warning).
    """
    from kb_index import VectorIndex
    from kb_schema import KBLeakageError, leakage_verdict, load_manifest, load_values, source_dir

    d = source_dir(source)
    manifest = load_manifest(source)
    verdict = leakage_verdict(manifest.leakage_audit)
    if verdict != "CLEAN":
        if not allow_unclean:
            raise KBLeakageError(
                f"kb_retrieve.load_source({source!r}): refusing to load — leakage_audit verdict="
                f"{verdict!r} (not CLEAN). This source is NOT admissible for a knowledge-utilization "
                "claim. Pass allow_unclean=True to override for PoC/debug only."
            )
        print(
            f"  [kb_retrieve] WARNING: loading UNCLEAN source {source!r} (verdict={verdict!r}) "
            "with allow_unclean=True — NOT admissible for a knowledge-utilization claim.",
            flush=True,
        )
    # 2026-07-13 (M1 item 2): resolving the DEFAULT (same-modality) query embedder is deferred to
    # first USE, not load time -- a source whose embedder has no same-modality query path wired at
    # all (e.g. a text-keyed source built with an audio-only embedder token, valid input for the
    # CROSS-MODAL-only retrieve_cross_modal_audio bridge) must still be loadable; it should only
    # fail if a caller actually tries source_obj['embed_query'](...) without one.
    try:
        embed_query = _query_embedder(manifest, d)
    except ValueError as resolve_err:
        def embed_query(_qs, _e=resolve_err):
            raise RuntimeError(
                f"kb_retrieve.load_source({source!r}): no default (same-modality) query embedder "
                f"is available for this source ({_e}). If you intended a CROSS-MODAL audio query "
                "against a text-keyed source, call kb_retrieve.retrieve_cross_modal_audio(source_obj, "
                "...) instead of source_obj['embed_query']."
            ) from _e
    return {
        "index": VectorIndex.load(d),
        "values": load_values(source),
        "manifest": manifest,
        "embed_query": embed_query,
    }


def retrieve(source_obj: dict, queries: list, topk: int = 5) -> list[list[dict]]:
    """For each query (audio path or text) return topk [{value, sim}] from the value store.

    2026-07-11 (ticket #25 P1c): asserts the query embedding's dimension matches the index's key
    dimension before searching — a mismatched embedder (wrong token resolved, or a caller bypassing
    ``load_source``'s own ``embed_query``) would otherwise fail silently (FAISS/numpy would either
    crash with an opaque shape error deep inside the ANN call, or — worse, for a numpy-flat backend
    with a coincidentally-broadcastable shape — silently produce garbage similarities). Raising here,
    with the source/manifest identity in the message, replaces both failure modes with one clear one.
    """
    import numpy as np

    index, values, embed_query = source_obj["index"], source_obj["values"], source_obj["embed_query"]
    q_vec = np.asarray(embed_query(queries), dtype="float32")
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)
    if q_vec.shape[1] != index.dim:
        manifest = source_obj.get("manifest")
        src_name = getattr(manifest, "source", "?")
        token = getattr(manifest, "embedder_token", None)
        raise ValueError(
            f"kb_retrieve.retrieve: query embedding dim {q_vec.shape[1]} != index dim {index.dim} "
            f"for source {src_name!r} (embedder_token={token!r}) -- the query embedder does not "
            "match the one this source's keys were built with."
        )
    idx, sims = index.search(q_vec, topk)
    out = []
    for i in range(len(queries)):
        out.append([{"value": values[int(j)], "sim": float(sims[i][c])} for c, j in enumerate(idx[i])])
    return out


def _search_and_build(index, values, q_vec, topk: int) -> list[list[dict]]:
    """Shared ANN-search + output-shaping tail of ``retrieve``/``retrieve_cross_modal_audio``."""
    idx, sims = index.search(q_vec, topk)
    out = []
    for i in range(q_vec.shape[0]):
        out.append([{"value": values[int(j)], "sim": float(sims[i][c])} for c, j in enumerate(idx[i])])
    return out


def cross_modal_audio_query_status(manifest) -> str:
    """Whether an AUDIO query can be embedded into ``manifest``'s (TEXT-keyed) key space — one of
    ``"supported"`` / ``"pending-GPU-window"`` / ``"ARM-BLOCKED-cross-modal"``. See
    ``CROSS_MODAL_AUDIO_QUERY_STATUS``'s module-level docstring for what each status means.
    ``token=None`` (embedder_token unrecoverable) is treated as ``"ARM-BLOCKED-cross-modal"`` — no
    routing decision can be made without knowing which embedder built the keys.
    """
    import kb_embed

    token = manifest.embedder_token or kb_embed.infer_embedder_token(manifest.embedder)
    if not token:
        return "ARM-BLOCKED-cross-modal"
    return CROSS_MODAL_AUDIO_QUERY_STATUS.get(token, "ARM-BLOCKED-cross-modal")


def retrieve_cross_modal_audio(source_obj: dict, audio_query_paths: list[str], topk: int = 5) -> list[list[dict]]:
    """Cross-modal AUDIO query into a TEXT-keyed source (2026-07-13, M1 engineering-base item 2 —
    fixes the "known follow-up" gap ``kb_retrieve.py``'s 2026-07-12 docstring flagged: an audio
    query against ``kb_batch_build.build_squtr_corpus_source``'s TEXT-keyed corpus sources had NO
    wired routing at all — a caller had to bypass ``load_source`` entirely and call
    ``kb_embed.embed_audio(queries, embedder=manifest.embedder_token)`` by hand).

    ``source_obj`` is ``load_source(...)``'s return value for a ``key_modality=='text'`` source.
    Raises ``ValueError`` if ``source_obj`` is audio-keyed (that direction is already served by
    ``retrieve``'s default ``embed_query`` — this bridge is specifically for TEXT-keyed sources).

    Raises ``kb_schema.KBCrossModalBlockedError`` (``.status`` = ``"ARM-BLOCKED-cross-modal"`` or
    ``"pending-GPU-window"`` — see ``CROSS_MODAL_AUDIO_QUERY_STATUS``) when the source's embedder
    cannot (yet) embed an audio query into its text-key space — a clean, specific, catchable
    signal instead of an opaque failure deep inside ``kb_embed``.
    """
    import numpy as np

    from kb_schema import KBCrossModalBlockedError

    manifest = source_obj["manifest"]
    if manifest.key_modality != "text":
        raise ValueError(
            "kb_retrieve.retrieve_cross_modal_audio: source is not text-keyed "
            f"(key_modality={manifest.key_modality!r}) -- an audio-keyed source already accepts "
            "an audio query via retrieve()/load_source's default embed_query; this bridge is for "
            "TEXT-keyed sources only."
        )

    import kb_embed

    token = manifest.embedder_token or kb_embed.infer_embedder_token(manifest.embedder)
    status = cross_modal_audio_query_status(manifest)
    if status != "supported":
        raise KBCrossModalBlockedError(
            f"kb_retrieve.retrieve_cross_modal_audio: source {manifest.source!r} "
            f"(embedder_token={token!r}) cannot embed an audio query into its text-key space -- "
            f"status={status!r}. "
            + ("No text tower / audio-query path exists for this embedder at all (permanent)."
               if status == "ARM-BLOCKED-cross-modal" else
               "This embedder's audio-query tower needs a resident llama-server not up this "
               "session (temporary -- retry once the GPU server is serving)."),
            status,
        )

    q_vec = np.asarray(kb_embed.embed_query_crossmodal_audio(list(audio_query_paths), embedder=token)[1],
                        dtype="float32")
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)
    index = source_obj["index"]
    if q_vec.shape[1] != index.dim:
        raise ValueError(
            f"kb_retrieve.retrieve_cross_modal_audio: query embedding dim {q_vec.shape[1]} != "
            f"index dim {index.dim} for source {manifest.source!r} (embedder_token={token!r})."
        )
    return _search_and_build(index, source_obj["values"], q_vec, topk)


def gate(hits: list[dict], gate_frac: float = 0.9) -> list[dict]:
    """R1 admission gate (sim >= gate_frac*max). STUDY-ONLY — T7 found precision-gating HURTS a strong omni."""
    if not hits:
        return hits
    thr = gate_frac * max(h["sim"] for h in hits)
    return [h for h in hits if h["sim"] >= thr] or hits[:1]

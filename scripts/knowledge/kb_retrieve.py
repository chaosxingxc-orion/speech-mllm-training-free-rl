"""kb_retrieve — load a PERSISTED speech-keyed source and retrieve values by key similarity.

Reloads the persisted vector index (``kb_index.VectorIndex.load``: FAISS or numpy-flat) + the value
store, and re-instantiates the KEY embedder from the manifest — so retrieval reproduces without
re-embedding the corpus. Query flow: embed query AUDIO -> ANN search over key index -> return VALUES.
(For legacy text-keyed sources, the query is text and the persisted TF-IDF/MiniLM key embedder is used.)

**Enforcement gate**: ``load_source`` reads ``manifest.leakage_audit`` and raises
``kb_schema.KBLeakageError`` unless the final verdict is ``CLEAN`` (an unaudited source, verdict
``None``, is treated as NOT admissible either — audit silence is not a clean bill). The escape hatch
is ``allow_unclean=True`` (PoC/debug only), which logs a loud warning on every load.

**Known follow-up (2026-07-12, RI item 9)**: ``_query_embedder`` below picks the query embedder from
``manifest.key_modality`` alone, which is fine while key_modality always matched the EXPECTED query
modality (audio-keyed sources queried by audio; legacy text-keyed sources queried by text). That
assumption breaks for ``kb_batch_build.build_squtr_corpus_source``'s new TEXT-keyed corpus sources,
whose intended query side is CROSS-MODAL for omni-family embedders (an AUDIO query against these
TEXT keys, via the same asymmetric encode_query/encode_document space) — ``_query_embedder`` would
currently route such a source's queries through its text branch regardless of the caller's actual
query modality. Not fixed in this pass (out of RI item 9's scope) — a caller needing cross-modal
querying against a text-keyed omni source should call ``kb_embed.embed_audio(queries,
embedder=manifest.embedder_token)`` directly rather than via ``load_source(...)["embed_query"]``
until this is wired.
"""
from __future__ import annotations

import pickle


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
    # legacy text key
    if manifest.embedder.startswith("tfidf"):
        v = pickle.load(open(source_path / "retriever.pkl", "rb"))
        return lambda qs: _l2(v.transform(list(qs)).toarray())
    if manifest.embedder.startswith("minilm"):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(manifest.embedder.split(":", 1)[1], device="cpu")
        return lambda qs: _l2(m.encode(list(qs), normalize_embeddings=True))
    raise ValueError(f"unknown embedder {manifest.embedder!r}")


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
    return {
        "index": VectorIndex.load(d),
        "values": load_values(source),
        "manifest": manifest,
        "embed_query": _query_embedder(manifest, d),
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


def gate(hits: list[dict], gate_frac: float = 0.9) -> list[dict]:
    """R1 admission gate (sim >= gate_frac*max). STUDY-ONLY — T7 found precision-gating HURTS a strong omni."""
    if not hits:
        return hits
    thr = gate_frac * max(h["sim"] for h in hits)
    return [h for h in hits if h["sim"] >= thr] or hits[:1]

"""kb_retrieve — load a PERSISTED speech-keyed source and retrieve values by key similarity.

Reloads the persisted vector index (``kb_index.VectorIndex.load``: FAISS or numpy-flat) + the value
store, and re-instantiates the KEY embedder from the manifest — so retrieval reproduces without
re-embedding the corpus. Query flow: embed query AUDIO -> ANN search over key index -> return VALUES.
(For legacy text-keyed sources, the query is text and the persisted TF-IDF/MiniLM key embedder is used.)

**Enforcement gate**: ``load_source`` reads ``manifest.leakage_audit`` and raises
``kb_schema.KBLeakageError`` unless the final verdict is ``CLEAN`` (an unaudited source, verdict
``None``, is treated as NOT admissible either — audit silence is not a clean bill). The escape hatch
is ``allow_unclean=True`` (PoC/debug only), which logs a loud warning on every load.
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
    logmel-keyed KB with CLAP (or vice-versa) compares vectors from different spaces. So we map the
    manifest's recorded embedder back to its explicit arg, never defaulting to 'auto' (which could pick
    a different real embedder than the one keyed).
    """
    if manifest.key_modality == "audio":
        import kb_embed

        emb = manifest.embedder
        arg = (
            "logmel-stats" if emb.startswith("logmel")
            else "clap" if emb.startswith("clap")
            else "omni-embed" if emb.startswith("omni")
            else "auto"
        )
        return lambda qs: kb_embed.embed_audio(qs, embedder=arg)[1]
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
    """For each query (audio path or text) return topk [{value, sim}] from the value store."""
    index, values, embed_query = source_obj["index"], source_obj["values"], source_obj["embed_query"]
    q_vec = embed_query(queries)
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

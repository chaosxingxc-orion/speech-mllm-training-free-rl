"""kb_retrieve — load a PERSISTED speech-keyed source and retrieve values by key similarity.

Reloads the persisted vector index (``kb_index.VectorIndex.load``: FAISS or numpy-flat) + the value
store, and re-instantiates the KEY embedder from the manifest — so retrieval reproduces without
re-embedding the corpus. Query flow: embed query AUDIO -> ANN search over key index -> return VALUES.
(For legacy text-keyed sources, the query is text and the persisted TF-IDF/MiniLM key embedder is used.)
"""
from __future__ import annotations

import pickle


def _l2(X):
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def _query_embedder(manifest, source_path):
    """Rebuild the KEY embedder so queries land in the same space as the stored keys."""
    if manifest.key_modality == "audio":
        import kb_embed

        emb = manifest.embedder
        return lambda qs: kb_embed.embed_audio(qs, embedder="omni-embed" if emb.startswith("omni") else "auto")[1]
    # legacy text key
    if manifest.embedder.startswith("tfidf"):
        v = pickle.load(open(source_path / "retriever.pkl", "rb"))
        return lambda qs: _l2(v.transform(list(qs)).toarray())
    if manifest.embedder.startswith("minilm"):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(manifest.embedder.split(":", 1)[1], device="cpu")
        return lambda qs: _l2(m.encode(list(qs), normalize_embeddings=True))
    raise ValueError(f"unknown embedder {manifest.embedder!r}")


def load_source(source: str) -> dict:
    """Return {index, values, manifest, embed_query} for a persisted source."""
    from kb_index import VectorIndex
    from kb_schema import load_manifest, load_values, source_dir

    d = source_dir(source)
    manifest = load_manifest(source)
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

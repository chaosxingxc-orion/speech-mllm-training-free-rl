"""kb_index — the KEY matrix as a PERSISTED vector index (the modern retrieval core).

Owner's framing: "把 key 当做 matrix,把 index 存下来,再把 value 存储下来." So the key embeddings are a
matrix ``keys.npy`` [N×d] and a persisted vector index sits on top. FAISS (``IndexFlatIP`` over
L2-normalized vectors = exact cosine) if available — the standard ANN backbone the old transient
TF-IDF pool never had; otherwise a numpy flat-cosine fallback with identical results so the pipeline
runs even without FAISS installed. Values are stored separately (see kb_schema / kb_build).
"""
from __future__ import annotations

from pathlib import Path


class VectorIndex:
    """A persisted cosine index over an L2-normalized key matrix. FAISS-backed if available."""

    def __init__(self, keys, backend: str, faiss_index=None):
        self.keys = keys  # np.ndarray [N×d], L2-normalized
        self.backend = backend  # "faiss-flat-ip" | "numpy-flat"
        self._faiss = faiss_index

    @property
    def n(self) -> int:
        return int(self.keys.shape[0])

    @property
    def dim(self) -> int:
        return int(self.keys.shape[1])

    @staticmethod
    def build(keys, index_type: str = "auto", hnsw_m: int = 32) -> "VectorIndex":
        """Build a persisted cosine index.

        index_type: "flat" (exact; small/medium) | "hnsw" (graph ANN; the 2025-26 default for large,
        dynamic corpora — Qdrant/Weaviate/pg/ES default) | "auto" (flat if n<50k else hnsw). Keys must
        be L2-normalized so inner-product == cosine (CLAP/CLIP/WavRAG convention). For billion-scale /
        out-of-RAM corpora prefer IVF-PQ / DiskANN, or a managed store (Qdrant/Milvus) whose
        {id, vector, payload} model gives value-separation + filtering + persistence for free.
        """
        import numpy as np

        keys = np.ascontiguousarray(np.asarray(keys, dtype="float32"))
        n, d = keys.shape
        want = "hnsw" if (index_type == "hnsw" or (index_type == "auto" and n >= 50_000)) else "flat"
        try:
            import faiss

            if want == "hnsw":
                idx = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                idx.add(keys)
                return VectorIndex(keys, "faiss-hnsw-ip", idx)
            idx = faiss.IndexFlatIP(d)  # inner product on normalized == cosine
            idx.add(keys)
            return VectorIndex(keys, "faiss-flat-ip", idx)
        except Exception as e:
            print(f"  [kb_index] faiss unavailable ({type(e).__name__}); numpy-flat fallback", flush=True)
            return VectorIndex(keys, "numpy-flat")

    def search(self, queries, topk: int = 5):
        """Return (idx[Nq×k], sims[Nq×k]) — top-k nearest key rows per query (cosine)."""
        import numpy as np

        q = np.ascontiguousarray(np.asarray(queries, dtype="float32"))
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, topk)
            return idx, sims
        sims_full = q @ self.keys.T
        idx = np.argsort(-sims_full, axis=1)[:, :topk]
        sims = np.take_along_axis(sims_full, idx, axis=1)
        return idx, sims

    def save(self, dir_path) -> None:
        import numpy as np

        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "keys.npy", self.keys)
        if self._faiss is not None:
            try:
                import faiss

                faiss.write_index(self._faiss, str(d / "index.faiss"))
            except Exception as e:
                print(f"  [kb_index] faiss write failed ({type(e).__name__}); keys.npy still persisted", flush=True)

    @staticmethod
    def load(dir_path) -> "VectorIndex":
        import numpy as np

        d = Path(dir_path)
        keys = np.load(d / "keys.npy")
        faiss_path = d / "index.faiss"
        if faiss_path.exists():
            try:
                import faiss

                return VectorIndex(keys, "faiss-flat-ip", faiss.read_index(str(faiss_path)))
            except Exception:
                pass
        return VectorIndex(keys, "numpy-flat")

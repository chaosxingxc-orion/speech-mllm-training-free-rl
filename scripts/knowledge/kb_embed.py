"""kb_embed — modality embedders. In a MULTIMODAL KB the KEY is SPEECH (audio) by default.

Design (owner, 2026-07-07): the knowledge base is keyed on SPEECH. A key = a dense audio embedding;
the value is another modality (transcript / labels / text / other-modal). So the primary embedder is
an AUDIO embedder that turns a waveform into a vector; text embedding is the legacy path (only for
text-keyed reading-comprehension passages, kept for back-compat, not the centerpiece).

Audio embedder tiers (auto):
  omni-embed   : ``omni-embed-nemotron-3b`` (on disk; a real neural audio embedder) — run in WSL/GPU.
                 This is THE key embedder for Stage-2.
  logmel-stats : CPU/offline PoC fallback — mean+std of a log-mel spectrogram. Runnable anywhere with
                 no GPU/network so the pipeline is testable; NOT a Stage-2-grade key. Loudly flagged.

Text embedder tiers (legacy, text-keyed knowledge only): MiniLM -> TF-IDF.
Lazy imports throughout (CLAUDE.md).
"""
from __future__ import annotations

import os

OMNI_EMBED_DIRNAME = "omni-embed-nemotron-3b"  # under $SPEECHRL_DATA_DIR/models


def _l2(X):
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)


def _logmel_stats(wav_path: str, n_mels: int = 64):
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=16000)
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels) + 1e-9)
    return np.concatenate([logmel.mean(axis=1), logmel.std(axis=1)]).astype("float32")


def _omni_embed(wav_paths):
    """Real audio key embedder (omni-embed-nemotron-3b). WSL/GPU. Returns (name, matrix[N×d])."""
    import numpy as np
    from sentence_transformers import SentenceTransformer  # omni-embed ships a ST-compatible loader

    model_dir = os.path.join(os.environ.get("SPEECHRL_DATA_DIR", ""), "models", OMNI_EMBED_DIRNAME)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"omni-embed model not found at {model_dir}")
    m = SentenceTransformer(model_dir, device="cuda")
    embs = m.encode(list(wav_paths), normalize_embeddings=True)  # loader handles wav->features
    return f"omni-embed:{OMNI_EMBED_DIRNAME}", _l2(np.asarray(embs, dtype="float32"))


def embed_audio(wav_paths, embedder: str = "auto"):
    """Embed audio -> (name, key_matrix[N×d], L2-normalized). Primary KB key embedder."""
    import numpy as np

    if embedder in ("auto", "omni-embed"):
        try:
            return _omni_embed(wav_paths)
        except Exception as e:
            if embedder == "omni-embed":
                raise
            print(
                f"  [kb_embed] omni-embed unavailable ({type(e).__name__}: {str(e)[:80]}); "
                f"logmel-stats PoC fallback (NOT Stage-2 grade)",
                flush=True,
            )
    keys = np.stack([_logmel_stats(p) for p in wav_paths]).astype("float32")
    return "logmel-stats-64", _l2(keys)


def embed_text(texts, embedder: str = "auto"):
    """Legacy TEXT key embedder (text-keyed passages only). Returns (name, matrix, fitted_or_None)."""
    import numpy as np

    if embedder in ("auto", "minilm"):
        try:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
            return "minilm:paraphrase-multilingual-MiniLM-L12-v2", _l2(m.encode(list(texts), normalize_embeddings=True)), m
        except Exception as e:
            if embedder == "minilm":
                raise
            print(f"  [kb_embed] minilm unavailable ({type(e).__name__}); TF-IDF legacy fallback", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50000)
    kb = _l2(v.fit_transform(list(texts)).toarray())
    return "tfidf-word12", kb, v

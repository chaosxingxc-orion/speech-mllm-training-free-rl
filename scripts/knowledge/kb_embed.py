"""kb_embed — modality embedders. In a MULTIMODAL KB the KEY is SPEECH (audio) by default.

Design (owner, 2026-07-07): the knowledge base is keyed on SPEECH. A key = a dense audio embedding;
the value is another modality (transcript / labels / text / other-modal).

RELIABILITY CONTRACT (owner: "得保证工程可靠性和稳定性"): ``embedder='auto'`` uses a REAL semantic
audio embedder (CLAP, a stable well-supported audio encoder; or omni-embed if wired) and **RAISES if
none loads** — it NEVER silently degrades to the crude logmel-stats key. Silently building a production
KB on a non-semantic key (mean+std of a log-mel) would make retrieval unreliable without the caller
knowing; that failure mode is now impossible. logmel-stats is a PoC/offline-smoke key available ONLY via
the explicit ``embedder='logmel-stats'`` opt-in, and it warns loudly.

Text embedder tiers (legacy, text-keyed knowledge only): MiniLM -> TF-IDF.
Lazy imports throughout (CLAUDE.md).
"""
from __future__ import annotations

import os

OMNI_EMBED_DIRNAME = "omni-embed-nemotron-3b"  # under $SPEECHRL_DATA_DIR/models
CLAP_MODEL = "laion/clap-htsat-unfused"  # stable, documented audio embedder; 48 kHz mono


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


def _clap_embed(wav_paths):
    """Reliable, semantic audio key embedder (CLAP). CPU-capable. Returns (name, matrix[N×d])."""
    import librosa
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL)
    model.eval()
    proc = ClapProcessor.from_pretrained(CLAP_MODEL)
    audios = [librosa.load(p, sr=48000)[0] for p in wav_paths]  # CLAP expects 48 kHz mono
    embs = []
    B = 16
    for i in range(0, len(audios), B):
        inp = proc(audios=audios[i : i + B], sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            embs.append(model.get_audio_features(**inp).cpu().numpy())
    return f"clap:{CLAP_MODEL}", _l2(np.concatenate(embs, axis=0).astype("float32"))


def _omni_embed(wav_paths):
    """Optional acoustic embedder (omni-embed-nemotron-3b). WSL/GPU; loader wiring is model-specific."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_dir = os.path.join(os.environ.get("SPEECHRL_DATA_DIR", ""), "models", OMNI_EMBED_DIRNAME)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"omni-embed model not found at {model_dir}")
    m = SentenceTransformer(model_dir, device="cuda")
    embs = m.encode(list(wav_paths), normalize_embeddings=True)
    return f"omni-embed:{OMNI_EMBED_DIRNAME}", _l2(np.asarray(embs, dtype="float32"))


def embed_audio(wav_paths, embedder: str = "auto"):
    """Embed audio -> (name, key_matrix[N×d], L2-normalized). Primary KB key embedder.

    RELIABILITY: ``auto`` tries real semantic embedders (CLAP, then omni-embed) and RAISES if none is
    available — it does NOT silently fall back to logmel-stats. Use ``embedder='logmel-stats'`` ONLY for
    an offline smoke test (it warns). ``embedder='clap'`` / ``'omni-embed'`` force a specific real one.
    """
    import numpy as np

    if embedder == "logmel-stats":
        print(
            "  [kb_embed] WARNING: logmel-stats is a PoC/offline-smoke key (mean+std log-mel), "
            "NOT a semantic embedder — do NOT build a real KB on it.",
            flush=True,
        )
        keys = np.stack([_logmel_stats(p) for p in wav_paths]).astype("float32")
        return "logmel-stats-64", _l2(keys)

    tried = []
    if embedder in ("auto", "clap"):
        try:
            return _clap_embed(wav_paths)
        except Exception as e:
            tried.append(f"clap({type(e).__name__}: {str(e)[:60]})")
            if embedder == "clap":
                raise
    if embedder in ("auto", "omni-embed"):
        try:
            return _omni_embed(wav_paths)
        except Exception as e:
            tried.append(f"omni-embed({type(e).__name__}: {str(e)[:60]})")
            if embedder == "omni-embed":
                raise
    raise RuntimeError(
        "kb_embed: no reliable semantic audio embedder available (tried: "
        + "; ".join(tried)
        + "). For a real KB, install/cache CLAP (`pip install transformers` + cache "
        + f"'{CLAP_MODEL}') or wire omni-embed. For an OFFLINE SMOKE TEST ONLY, pass "
        + "embedder='logmel-stats' explicitly (it is NOT semantic and must not back a real KB)."
    )


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

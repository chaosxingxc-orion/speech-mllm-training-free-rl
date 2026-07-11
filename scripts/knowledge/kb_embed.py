"""kb_embed — modality embedders. In a MULTIMODAL KB the KEY is SPEECH (audio) by default.

Design (owner, 2026-07-07): the knowledge base is keyed on SPEECH. A key = a dense audio embedding;
the value is another modality (transcript / labels / text / other-modal).

RELIABILITY CONTRACT (owner: "得保证工程可靠性和稳定性"): ``embedder='auto'`` uses a REAL semantic
audio embedder (CLAP, a stable well-supported audio encoder; or omni-embed if wired) and **RAISES if
none loads** — it NEVER silently degrades to the crude logmel-stats key. Silently building a production
KB on a non-semantic key (mean+std of a log-mel) would make retrieval unreliable without the caller
knowing; that failure mode is now impossible. logmel-stats is a PoC/offline-smoke key available ONLY via
the explicit ``embedder='logmel-stats'`` opt-in, and it warns loudly.

omni-embed (``nvidia/omni-embed-nemotron-3b``) is wired via its OFFICIAL asymmetric
``sentence-transformers`` API (``trust_remote_code=True`` is required — custom ``NVOmniEmbedModel``
code): ``encode_document`` for the document/KEY side (``_omni_embed`` — audio keys, called from
``embed_audio``) and ``encode_query`` for the query side (``_omni_embed_query`` — text queries,
reachable via ``embed_text(embedder='omni-embed')`` for cross-modal text->audio-KB retrieval; NOT
part of ``embed_text``'s ``'auto'`` tier, which stays MiniLM -> TF-IDF unchanged). Device defaults to
**CPU**: the GPU is held by the resident llama-server and must never be touched (CLAUDE.md); pass
``device='cuda'`` explicitly only when the GPU is confirmed free. A CPU load of this ~4.7B-param model
is expected to be slow (multi-minute cold start; unverified in this session) — see
``scripts/knowledge/README.md`` for the documented manual smoke-test step; do not add an
automated/CI call to ``_omni_embed``/``_omni_embed_query`` without first timing it manually and
setting a generous timeout.

Text embedder tiers (legacy, text-keyed knowledge only): MiniLM -> TF-IDF (``'auto'``); omni-embed
``encode_query`` is available as an explicit opt-in (see above).
Lazy imports throughout (CLAUDE.md).
"""
from __future__ import annotations

import os

OMNI_EMBED_DIRNAME = "omni-embed-nemotron-3b"  # under $SPEECHRL_DATA_DIR/models
CLAP_MODEL = "laion/clap-htsat-unfused"  # stable, documented audio embedder; 48 kHz mono

# ---------------------------------------------------------------------------------------------
# Step-2 unified embedder interface (2026-07-10; wiki/2026-07-09-q1q2-embedder-granularity-
# decision-memo.md S2.5 + wiki/2026-07-10-step2-grid-draft.md S2/S5). Each entry below is a
# LAZY-imported loader over a model already on disk at $SPEECHRL_DATA_DIR/models/<dirname> (the
# "candidate" fetch campaign, docs/datasets.lock.json S+9 pending owner lock-approval). All are
# CPU-default (GPU is held by the resident llama-server -- CLAUDE.md) and return L2-normalized
# np.float32 arrays. ``EMBEDDERS`` is the planning-facing metadata registry (dim / license /
# domain / needs_server) consumed by kb_batch_build.py's --plan mode; ``_LOADER_FNS`` is the
# name -> callable(wav_paths, device="cpu") dispatch table wired into ``embed_audio`` below.
# ---------------------------------------------------------------------------------------------

MODEL_DIRNAMES = {
    "glap": "glap",
    "meralion-se2": "meralion-speech-encoder-2",
    "wavlm-base-plus": "wavlm-base-plus",
    "wavlm-large": "wavlm-large",
    "eres2netv2": "eres2netv2-zh",
    "campplus": "campplus-zh",
    "emotion2vec-s": "emotion2vec-s",
    "emotion2vec-plus-large": "emotion2vec-plus-large",
    "dasheng": "dasheng",
    "clsp": "clsp",
    "sensevoice-small": "sensevoice-small",
    "sense": "sense",
    "lco-3b": "lco-embedding-omni-3b-gguf",
    "lco-7b": "lco-embedding-omni-7b-gguf",
}

# Planning metadata only (dim/license/domain) -- NOT a promise every entry loads on this box this
# session; see each loader's docstring + the CPU-smoke report for what actually ran. ``dim: None``
# means "unmeasured" (blocked this session; see ``note``). License strings are per the model's own
# README/HF metadata, read this session -- not an independent legal audit.
EMBEDDERS: dict[str, dict] = {
    "glap": {
        "dim": 1024, "license": "apache-2.0", "domain": "content zh+en + sound (H-a content-key candidate)",
        "needs_server": False,
    },
    "meralion-se2": {
        "dim": 1024, "license": "custom-gated (MERaLiON Public License; SEA-focused, non-Apache)",
        "domain": "multilingual (en/zh/ms/ta/th/id/vi) speech encoder, general-purpose backbone",
        "needs_server": False,
    },
    "wavlm-base-plus": {
        "dim": 768, "license": "MIT", "domain": "speaker/general SSL baseline (small)", "needs_server": False,
    },
    "wavlm-large": {
        "dim": 1024, "license": "MIT", "domain": "speaker/general SSL baseline (large)", "needs_server": False,
    },
    "eres2netv2": {
        "dim": 192, "license": "apache-2.0", "domain": "speaker-ID zh (H-a speaker-key candidate, 3D-Speaker)",
        "needs_server": False,
    },
    "campplus": {
        "dim": 192, "license": "apache-2.0", "domain": "speaker-ID zh (H-a speaker-key alt, 3D-Speaker)",
        "needs_server": False,
    },
    "emotion2vec-plus-large": {
        "dim": 1024, "license": "apache-2.0", "domain": "SER (H-a emotion-key candidate, FunASR)",
        "needs_server": False,
    },
    "emotion2vec-s": {
        "dim": None, "license": "apache-2.0", "domain": "SER (H-a emotion-key candidate, C2SER/fairseq variant)",
        "needs_server": False,
        "note": (
            "needs fairseq (checkpoint is a raw data2vec/fairseq .pt, not FunASR-format) -- SKIPPED "
            "this session: fairseq's pinned omegaconf==2.0.6/hydra-core==1.0.7 would DOWNGRADE the "
            "newer omegaconf/hydra-core this session's funasr install already brought in, a real "
            "conflict risk in the shared venv. Install in a DEDICATED venv: "
            "`pip install fairseq==0.12.2` (+ the C2SER repo's `user_dir` module for the custom "
            "data2vec task/model registration -- ASLP-lab/Emotion2Vec-S ships no FunASR config)."
        ),
    },
    "dasheng": {
        "dim": 1536, "license": "apache-2.0", "domain": "general audio (HEAR; wins CREMA-D) -- SER/general-audio candidate",
        "needs_server": False,
    },
    "clsp": {
        "dim": 512, "license": "apache-2.0", "domain": "speaking-style / paralinguistic contrastive key",
        "needs_server": False,
    },
    "sense": {
        "dim": None, "license": "cc0-1.0", "domain": "ST/multilingual (VoxPopuli R@1 96.55 > SONAR, per survey)",
        "needs_server": False,
        "note": (
            "speechbrain is installed (this session), but the local snapshot ships only checkpoint "
            "files (CKPT.yaml, brain.ckpt, wav2vec2.ckpt, model.ckpt, ...) -- NOT the source repo's "
            "hyperparams.yaml (+ any custom_model.py) speechbrain's `from_hparams`/EncoderClassifier "
            "needs to reconstruct the architecture. SKIPPED: re-fetch hyperparams.yaml alongside the "
            "existing checkpoints from the model's original release before this can load."
        ),
    },
    "sensevoice-small": {
        "dim": None, "license": "apache-2.0", "domain": "zh+en ASR encoder (SenseVoice) -- acoustic-key probe (K1/K2)",
        "needs_server": False,
        "note": (
            "funasr is installed (this session) and the model LOADS via `funasr.AutoModel`, but "
            "`generate(..., extract_embedding=True)` is honored only by data2vec/emotion2vec-family "
            "checkpoints -- this CTC/attention ASR architecture returns no 'feats' key. A pooled "
            "embedding needs a custom forward hook on `model.model.encoder` (bypassing the "
            "generate() convenience API and replicating its internal frontend+LFR+CMVN "
            "preprocessing) -- not attempted this session; flagged NeedsCustomHook."
        ),
    },
    "lco-3b": {
        "dim": None, "license": "apache-2.0", "domain": "content key (MAEB-lineage; H-a/H-b candidate, GGUF)",
        "needs_server": True, "server_env": "LCO3B_EMBED_SERVER",
    },
    "lco-7b": {
        "dim": None, "license": "apache-2.0", "domain": "content key (larger MAEB-lineage variant, GGUF)",
        "needs_server": True, "server_env": "LCO7B_EMBED_SERVER",
    },
    "qwen3-omni-own": {
        "dim": 2048, "license": "apache-2.0",
        "domain": "H-b own-hidden-state content key (frozen-omni single-space thesis; dim confirmed live per step2-grid-draft S0)",
        "needs_server": True, "server_env": "QWEN3OMNI_EMBED_SERVER",
    },
    # Pre-existing legacy tiers (not new this session; added HERE only as metadata so the step-2
    # "8 content-key embedders" axis -- wiki/2026-07-10-step2-grid-draft.md S2 -- can be enumerated
    # from this ONE registry: glap / lco-3b / lco-7b / qwen3-omni-own / sense / meralion-se2 / clap
    # (negative control) / omni-embed-nemotron (NC-license contrast). Both already dispatch via the
    # pre-existing 'clap'/'omni-embed' embedder args in ``embed_audio`` -- see module docstring.
    "clap": {
        "dim": 512, "license": "apache-2.0", "domain": "sound-event/general audio -- NEGATIVE CONTROL (content R@1~0.1%, survey F1)",
        "needs_server": False,
        "note": "laion/clap-htsat-unfused is HF-hub-fetched (not a local models/ snapshot) -- this "
                "session's WSL box could not reach the Hub to re-verify dim=512 live; value is the "
                "well-documented checkpoint constant, not independently re-measured this session.",
    },
    "omni-embed-nemotron": {
        "dim": 2048, "license": "NVIDIA OneWay Noncommercial (NC) -- excluded from any default key, contrast-only",
        "domain": "asymmetric encode_document/encode_query content key -- NC-license contrast candidate",
        "needs_server": False,
    },
}


def _model_dir(name: str) -> str:
    dirname = MODEL_DIRNAMES.get(name, name)
    d = os.path.join(os.environ.get("SPEECHRL_DATA_DIR", ""), "models", dirname)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"kb_embed: model dir not found for embedder {name!r}: {d}")
    return d


def _reset_meta_pe_caches(model) -> int:
    """Workaround for a real transformers-v5 fast-init bug hit by 2 of our trust_remote_code
    checkpoints (MERaLiON-SpeechEncoder-2's ``ConformerRelPositionalEmbedding``; CLSP's
    zipformer2 ``RelPositionalEncoding``): both cache a lazily-extended sinusoidal
    positional-encoding table in a plain ``self.pe`` attribute (NOT a registered buffer/parameter),
    built once inside ``__init__``. transformers v5's ``from_pretrained`` always constructs the
    model class under a ``meta`` device context (independent of ``low_cpu_mem_usage``); only
    REGISTERED params/buffers get restored from the state dict afterward, so this plain attribute
    is permanently stuck on ``meta`` and the first real forward pass crashes inside the module's own
    ``extend_pe`` with ``NotImplementedError: Cannot copy out of meta tensor; no data!``. Resetting
    the cached ``.pe`` to ``None`` forces a real (CPU) recompute on next use -- numerically identical
    to a freshly-constructed model, since the table is a deterministic sinusoidal function, never a
    learned parameter. Returns the number of caches reset (0 is normal for architectures without
    this pattern, e.g. GLAP / WavLM / Dasheng -- calling this on them is a harmless no-op scan).
    """
    import torch

    n = 0
    for m in model.modules():
        v = getattr(m, "pe", None)
        if isinstance(v, torch.Tensor) and v.is_meta:
            m.pe = None
            n += 1
    return n


def _glap_embed(wav_paths, device: str = "cpu"):
    """GLAP (mispeech/GLAP, Apache) -- content key candidate (zh+en+sound), 1024-d.
    Official API: ``AutoModel(trust_remote_code=True).encode_audio(raw_waveform_tensor)``.
    """
    import librosa
    import numpy as np
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(_model_dir("glap"), trust_remote_code=True).to(device).eval()
    _reset_meta_pe_caches(model)
    embs = []
    for p in wav_paths:
        y, _ = librosa.load(p, sr=16000, mono=True)
        audio = torch.from_numpy(y).float().unsqueeze(0).to(device)
        with torch.no_grad():
            e = model.encode_audio(audio)
        embs.append(e.cpu().numpy()[0])
    return "glap:GLAP", _l2(np.stack(embs).astype("float32"))


def _meralion_se2_embed(wav_paths, device: str = "cpu"):
    """MERaLiON-SpeechEncoder-2 (custom-gated, SEA-focused multilingual) -- 1024-d, mean-pooled
    ``last_hidden_state``. Needs ``_reset_meta_pe_caches`` (see its docstring for why).
    """
    import librosa
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    d = _model_dir("meralion-se2")
    fe = AutoFeatureExtractor.from_pretrained(d, trust_remote_code=True)
    model = AutoModel.from_pretrained(d, trust_remote_code=True).to(device).eval()
    _reset_meta_pe_caches(model)
    embs = []
    for p in wav_paths:
        y, _ = librosa.load(p, sr=16000, mono=True)
        inputs = fe([y], sampling_rate=16000, return_attention_mask=True, return_tensors="pt", do_normalize=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(input_values=inputs["input_values"], attention_mask=inputs.get("attention_mask"),
                        output_hidden_states=True)
        embs.append(out.last_hidden_state.mean(dim=1).cpu().numpy()[0])
    return "meralion-se2:MERaLiON-SpeechEncoder-2", _l2(np.stack(embs).astype("float32"))


def _wavlm_embed(variant: str, wav_paths, device: str = "cpu"):
    """WavLM base-plus (768-d) / large (1024-d) (MIT) -- speaker/general SSL baseline, mean-pooled
    ``last_hidden_state``. First-party ``transformers.WavLMModel`` (no trust_remote_code).
    """
    import librosa
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    d = _model_dir(f"wavlm-{variant}")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(d)
    model = WavLMModel.from_pretrained(d).to(device).eval()
    embs = []
    for p in wav_paths:
        y, _ = librosa.load(p, sr=16000, mono=True)
        inputs = fe(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = model(input_values=inputs.input_values.to(device))
        embs.append(out.last_hidden_state.mean(dim=1).cpu().numpy()[0])
    return f"wavlm-{variant}:{MODEL_DIRNAMES[f'wavlm-{variant}']}", _l2(np.stack(embs).astype("float32"))


def _dasheng_embed(wav_paths, device: str = "cpu"):
    """Dasheng-1.2B (mispeech/dasheng-1.2B, Apache) -- general audio (HEAR; wins CREMA-D), 1536-d.
    ``outputs.logits`` IS the mean-pooled embedding when ``outputdim=None`` (per model README).
    """
    import librosa
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    d = _model_dir("dasheng")
    fe = AutoFeatureExtractor.from_pretrained(d, trust_remote_code=True)
    model = AutoModel.from_pretrained(d, outputdim=None, trust_remote_code=True).to(device).eval()
    _reset_meta_pe_caches(model)
    embs = []
    for p in wav_paths:
        y, _ = librosa.load(p, sr=16000, mono=True)
        inputs = fe([y], sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        embs.append(out.logits.cpu().numpy()[0])
    return "dasheng:dasheng-1.2B", _l2(np.stack(embs).astype("float32"))


def _clsp_embed(wav_paths, device: str = "cpu"):
    """CLSP (yfyeung/CLSP, Apache) -- speaking-style/paralinguistic contrastive key, 512-d.

    Two transformers-v5 compatibility fixes needed beyond plain ``AutoModel.from_pretrained``:
    (1) ``CLSPModel.__init__`` never calls ``self.post_init()``, so the class is missing
    ``all_tied_weights_keys`` that v5's loading path reads unconditionally -- patched to ``{}``
    (correct: this contrastive model ties no weights) via ``get_class_from_dynamic_module`` BEFORE
    ``.from_pretrained`` so the same (module-cached) class object is patched. (2) the zipformer2
    encoder hits the same ``.pe`` meta-tensor bug as MERaLiON -- see ``_reset_meta_pe_caches``.
    """
    import librosa
    import numpy as np
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    d = _model_dir("clsp")
    cls = get_class_from_dynamic_module("modeling_clsp.CLSPModel", d)
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    model = cls.from_pretrained(d, trust_remote_code=True).to(device).eval()
    _reset_meta_pe_caches(model)
    embs = []
    for p in wav_paths:
        y, _ = librosa.load(p, sr=16000, mono=True)
        audio = torch.from_numpy(y).float().unsqueeze(0).to(device)
        audio_lens = torch.tensor([audio.shape[1]]).to(device)
        with torch.no_grad():
            audio_embedding, _text_embedding, _logit_scale = model(audio, audio_lens, ["a placeholder caption"])
        embs.append(audio_embedding.cpu().numpy()[0])
    return "clsp:CLSP", _l2(np.stack(embs).astype("float32"))


def _modelscope_sv_embed(name: str, wav_paths, device: str = "cpu"):
    """eres2netv2-zh / campplus-zh (3D-Speaker, Apache) -- zh speaker-ID, 192-d.
    ModelScope's own ``speaker-verification`` pipeline (native to the ``configuration.json`` these
    checkpoints ship, NOT FunASR). Needs 3 tiny pure-python deps beyond modelscope itself
    (``addict``, ``simplejson``, ``sortedcontainers`` -- installed this session, together < 1 MB).
    """
    import numpy as np
    from modelscope.pipelines import pipeline

    d = _model_dir(name)
    sv = pipeline(task="speaker-verification", model=d, device=device)
    embs = []
    for p in wav_paths:
        out = sv([p], output_emb=True)
        embs.append(np.asarray(out["embs"][0], dtype="float32"))
    return f"{name}:{MODEL_DIRNAMES[name]}", _l2(np.stack(embs))


def _emotion2vec_plus_large_embed(wav_paths, device: str = "cpu"):
    """Emotion2Vec-plus-large (iic/emotion2vec_plus_large lineage, Apache) -- SER, 1024-d.
    FunASR ``AutoModel(...).generate(wav, granularity='utterance', extract_embedding=True)`` ->
    the 'feats' key IS the pooled utterance embedding (data2vec-family convention).
    """
    import numpy as np
    from funasr import AutoModel as FunASRAutoModel

    d = _model_dir("emotion2vec-plus-large")
    model = FunASRAutoModel(model=d, device=device, disable_update=True)
    embs = []
    for p in wav_paths:
        res = model.generate(p, granularity="utterance", extract_embedding=True)
        embs.append(np.asarray(res[0]["feats"], dtype="float32"))
    return "emotion2vec-plus-large:FunASR", _l2(np.stack(embs).reshape(len(wav_paths), -1))


def _emotion2vec_s_embed(wav_paths, device: str = "cpu"):
    raise RuntimeError(
        "emotion2vec-s needs fairseq (raw data2vec/fairseq .pt checkpoint, no FunASR config shipped). "
        "SKIPPED this session: see EMBEDDERS['emotion2vec-s']['note'] for the exact install + the "
        "reason it was not installed into this shared venv (omegaconf/hydra-core version conflict "
        "with funasr, already installed here for emotion2vec-plus-large/sensevoice-small)."
    )


def _sense_embed(wav_paths, device: str = "cpu"):
    raise RuntimeError(
        "sense needs speechbrain hyperparams.yaml (+ any custom_model.py) not shipped in this "
        "snapshot -- only raw checkpoints (CKPT.yaml/*.ckpt) are on disk. See "
        "EMBEDDERS['sense']['note']."
    )


def _sensevoice_small_embed(wav_paths, device: str = "cpu"):
    raise RuntimeError(
        "sensevoice-small loads via funasr.AutoModel but generate(extract_embedding=True) returns "
        "no 'feats' for this CTC/attention ASR architecture (only data2vec/emotion2vec-family "
        "checkpoints are honored) -- pooled embedding needs a custom encoder hook. See "
        "EMBEDDERS['sensevoice-small']['note']."
    )


def _llama_server_embed(wav_paths, server_env: str, base_url: str | None = None, timeout: int = 300):
    """HTTP-client embedder for a GGUF audio embedder served by ``llama-server --embedding``
    (lco-3b / lco-7b / qwen3-omni-own's own hidden states). **NeedsServer**: this function talks to
    an ALREADY-RESIDENT server -- it never starts one (GPU is busy this session with the wave-1
    marathon; CLAUDE.md forbids touching it). ``base_url`` (else ``os.environ[server_env]``) and
    ``server_env`` are caller-supplied per embedder/config, never hardcoded to one port, since
    step-2 plans multiple GGUF embedders potentially time-sharing the GPU across separate launches.

    The media_marker gotcha: llama.cpp's multimodal ``/props`` response carries a ``media_marker``
    string (e.g. ``"<__media__>"``) that MUST be spliced into the request content at the exact
    position the audio should occupy. Some llama.cpp builds RANDOMIZE this marker per-launch, so it
    is fetched fresh from ``/props`` on every call here -- never hardcoded or cached across launches.
    """
    import base64
    import json
    import urllib.request

    import numpy as np

    url = base_url or os.environ.get(server_env)
    if not url:
        raise RuntimeError(
            f"kb_embed._llama_server_embed: no server URL -- pass base_url= or set ${server_env} "
            "to a resident `llama-server --embedding` instance. NeedsServer: not started here."
        )

    def _get(path):
        with urllib.request.urlopen(url + path, timeout=timeout) as r:
            return json.loads(r.read().decode())

    props = _get("/props")
    media_marker = props.get("media_marker") or (props.get("mmproj") or {}).get("media_marker") or "<__media__>"

    embs = []
    for p in wav_paths:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        payload = {
            "content": [
                {"type": "text", "text": media_marker},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ],
        }
        req = urllib.request.Request(url + "/embedding", data=json.dumps(payload).encode(),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        vec = out[0]["embedding"] if isinstance(out, list) else out.get("embedding")
        vec = np.asarray(vec, dtype="float32")
        if vec.ndim == 2:  # some llama.cpp builds return one row per token -- pool by mean
            vec = vec.mean(axis=0)
        embs.append(vec)
    return f"llama-server-embed:{server_env}", _l2(np.stack(embs))


_LOADER_FNS = {
    "glap": _glap_embed,
    "meralion-se2": _meralion_se2_embed,
    "wavlm-base-plus": lambda wp, device="cpu": _wavlm_embed("base-plus", wp, device=device),
    "wavlm-large": lambda wp, device="cpu": _wavlm_embed("large", wp, device=device),
    "eres2netv2": lambda wp, device="cpu": _modelscope_sv_embed("eres2netv2", wp, device=device),
    "campplus": lambda wp, device="cpu": _modelscope_sv_embed("campplus", wp, device=device),
    "emotion2vec-plus-large": _emotion2vec_plus_large_embed,
    "emotion2vec-s": _emotion2vec_s_embed,
    "dasheng": _dasheng_embed,
    "clsp": _clsp_embed,
    "sense": _sense_embed,
    "sensevoice-small": _sensevoice_small_embed,
    "lco-3b": lambda wp, device="cpu": _llama_server_embed(wp, server_env="LCO3B_EMBED_SERVER"),
    "lco-7b": lambda wp, device="cpu": _llama_server_embed(wp, server_env="LCO7B_EMBED_SERVER"),
    "qwen3-omni-own": lambda wp, device="cpu": _llama_server_embed(wp, server_env="QWEN3OMNI_EMBED_SERVER"),
    # 2026-07-11 (G2 control-arm prep, CPU embedder-smoke finding): "omni-embed-nemotron" is an
    # EMBEDDERS metadata key (the step-2 NC-license contrast arm -- run_mock.EMBEDDERS and
    # kb_batch_build.CONTENT_KEY_EMBEDDERS both enumerate it) but was never dispatchable here:
    # embed_audio only reached _omni_embed via the LEGACY "omni-embed" tier name, so a real
    # kb_batch_build.build_one("omni-embed-nemotron", ...) build (or a kb_retrieve query against a
    # source whose embedder_token is "omni-embed-nemotron") raised the "no reliable semantic audio
    # embedder available" RuntimeError. Wire the registry token to the SAME official
    # encode_document path the legacy name uses.
    "omni-embed-nemotron": lambda wp, device="cpu": _omni_embed(wp, device=device),
}


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


def _omni_model(device: str):
    """Lazily load the omni-embed SentenceTransformer (official asymmetric API).

    ``trust_remote_code=True`` is REQUIRED — the checkpoint ships custom modeling code
    (``modeling_nv_omni_embed.py``, ``NVOmniEmbedModel``). Not cached across calls: callers that embed
    many batches should hold the returned model themselves; kb_build/kb_retrieve each call this once
    per process-lifetime op, which is acceptable at Stage-1 corpus sizes.
    """
    from sentence_transformers import SentenceTransformer

    model_dir = os.path.join(os.environ.get("SPEECHRL_DATA_DIR", ""), "models", OMNI_EMBED_DIRNAME)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"omni-embed model not found at {model_dir}")
    return SentenceTransformer(model_dir, trust_remote_code=True, device=device)


def _omni_embed(wav_paths, device: str = "cpu"):
    """Real semantic audio KEY/document embedder (omni-embed-nemotron-3b) via the model's OFFICIAL
    asymmetric bi-encoder API: ``encode_document`` for the document/KEY side. CPU by default — the GPU
    is held by the resident llama-server and must never be touched (CLAUDE.md). See ``_omni_embed_query``
    for the QUERY-side (``encode_query``) half of the same API.
    """
    import numpy as np

    m = _omni_model(device)
    docs = [{"audio": p} for p in wav_paths]
    embs = m.encode_document(docs, convert_to_numpy=True, normalize_embeddings=True)
    return f"omni-embed:{OMNI_EMBED_DIRNAME}", _l2(np.asarray(embs, dtype="float32"))


def _omni_embed_query(texts, device: str = "cpu"):
    """omni-embed QUERY-side embedder — the ``encode_query`` half of the SAME official asymmetric API
    ``_omni_embed`` uses for documents/keys. Lands TEXT queries in the identical 2048-d cosine space as
    ``_omni_embed``'s audio keys, enabling cross-modal (text query -> audio-keyed KB) retrieval. CPU by
    default (GPU held by llama-server). Reachable via ``embed_text(embedder='omni-embed')``.
    """
    import numpy as np

    m = _omni_model(device)
    embs = m.encode_query(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return f"omni-embed:{OMNI_EMBED_DIRNAME}", _l2(np.asarray(embs, dtype="float32"))


# ---------------------------------------------------------------------------------------------
# 2026-07-11 (ticket #25 P1c + P2d): registry-token <-> descriptive-ename resolution, so
# kb_build can store the exact TOKEN kb_retrieve needs to re-invoke the SAME embedder for a query
# (never guessed from ename via fragile prefix matching, never silently defaulting to 'auto'/CLAP),
# plus a composite (multi-embedder, concatenated-key) embedder used by kb_batch_build's
# 'ha-multi-key' key_org arm (2026-07-11 ticket #25 P2d).
# ---------------------------------------------------------------------------------------------

# ename-prefix -> registry token, for sources built before embedder_token existed (kb_schema
# SourceManifest 2026-07-11 additive field) or built with embedder='auto'/'clap'/'omni-embed' where
# the CALLER's own arg isn't itself the resolved token. Every _LOADER_FNS-dispatched embedder's own
# ename already starts with its own registry key (see each _xxx_embed's `return f"{key}:..."`), so
# this table only needs the few names that do NOT start with their own token verbatim.
_SERVER_ENV_TOKEN = {
    "LCO3B_EMBED_SERVER": "lco-3b",
    "LCO7B_EMBED_SERVER": "lco-7b",
    "QWEN3OMNI_EMBED_SERVER": "qwen3-omni-own",
}


def infer_embedder_token(ename: str) -> str | None:
    """Best-effort recovery of a registry TOKEN from a descriptive ``ename`` (e.g. ``"glap:GLAP"``
    -> ``"glap"``). Used ONLY as a fallback for manifests built before ``embedder_token`` existed
    (see ``kb_schema.SourceManifest``'s docstring) -- returns ``None`` (never a guess) when the
    mapping is genuinely ambiguous, so the caller (``kb_retrieve._query_embedder``) can raise
    instead of silently defaulting to a different embedder.
    """
    if not ename:
        return None
    if ename.startswith("logmel-stats"):
        return "logmel-stats"
    if ename.startswith("llama-server-embed:"):
        return _SERVER_ENV_TOKEN.get(ename.split(":", 1)[1])
    if ename.startswith("composite:"):
        return ename  # composite tokens are already self-describing (see embed_audio_composite)
    prefix = ename.split(":", 1)[0]
    if prefix in _LOADER_FNS or prefix in ("clap", "omni-embed"):
        return prefix
    return None


def resolve_embedder_token(explicit: str, ename: str) -> str | None:
    """The token ``kb_build.build_source`` should stamp into ``manifest.embedder_token``.

    If the caller passed a concrete, non-'auto' embedder arg (the normal case for every
    ``kb_batch_build`` call -- ``CONTENT_KEY_EMBEDDERS`` never includes 'auto'), that IS the token:
    trust it verbatim, it is exactly what a future query must re-pass to ``embed_audio``/
    ``embed_text``. Only when the caller asked for 'auto' (legacy convenience tier, resolves
    internally to CLAP or omni-embed) do we need to recover the ACTUAL token from the returned
    ``ename`` via ``infer_embedder_token``.
    """
    if explicit and explicit != "auto":
        return explicit
    return infer_embedder_token(ename)


def embed_audio_composite(wav_paths, embedders: list[str], device: str = "cpu"):
    """Concatenate several embedders' (already L2-normalized) audio vectors into ONE composite key
    per wav, then re-normalize the concatenation as a whole. Backs ``kb_batch_build``'s
    ``'ha-multi-key'`` key_org arm (2026-07-11, ticket #25 P2d): the grid draft's H-a calls for
    "2-3 specialized key SPACES" (content / speaker / emotion); a true H-a would need 2-3
    INDEPENDENTLY-searchable indices plus a fusion/routing layer at retrieval time. This is the
    documented PRAGMATIC approximation used instead: ONE physical index whose key row is the
    concatenation of the content/speaker/emotion embedders' own vectors, so ``kb_retrieve`` still
    only needs to search ONE index (matching ``source_name_for``'s one-physical-source-per-
    (dataset,embedder,key_org,value_org) convention, ticket #25 P1a) — cosine similarity in the
    concatenated space is a (crude) proxy for "similar across all 2-3 signals at once", not a
    routed per-space retrieval. Returns ``(name, matrix)`` where ``name`` is
    ``"composite:" + "+".join(embedders)`` -- self-describing and re-parsed verbatim by
    ``kb_retrieve._query_embedder`` to rebuild the identical query-side concatenation.
    """
    import numpy as np

    parts = []
    for emb in embedders:
        _name, vecs = embed_audio(wav_paths, embedder=emb, device=device)
        parts.append(np.asarray(vecs, dtype="float32"))
    combined = np.concatenate(parts, axis=1)
    name = "composite:" + "+".join(embedders)
    return name, _l2(combined)


def embed_audio(wav_paths, embedder: str = "auto", device: str = "cpu"):
    """Embed audio -> (name, key_matrix[N×d], L2-normalized). Primary KB key embedder.

    RELIABILITY: ``auto`` tries real semantic embedders (CLAP, then omni-embed) and RAISES if none is
    available — it does NOT silently fall back to logmel-stats. Use ``embedder='logmel-stats'`` ONLY for
    an offline smoke test (it warns). ``embedder='clap'`` / ``'omni-embed'`` force a specific real one.
    ``device`` only affects the omni-embed tier (CPU default; GPU held by llama-server — CLAUDE.md).

    Step-2 unified interface: any key in ``EMBEDDERS`` (glap / meralion-se2 / wavlm-base-plus /
    wavlm-large / eres2netv2 / campplus / emotion2vec-plus-large / emotion2vec-s / dasheng / clsp /
    sense / sensevoice-small / lco-3b / lco-7b / qwen3-omni-own) dispatches to its dedicated loader
    via ``_LOADER_FNS`` — this is checked FIRST, before the legacy 'auto'/'clap'/'omni-embed'/
    'logmel-stats' tiers, so existing callers of those four names are unaffected.
    """
    import numpy as np

    if embedder in _LOADER_FNS:
        return _LOADER_FNS[embedder](wav_paths, device=device)

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
            return _omni_embed(wav_paths, device=device)
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


def embed_text(texts, embedder: str = "auto", device: str = "cpu"):
    """TEXT embedder. Returns (name, matrix, fitted_or_None).

    ``'auto'`` (default, UNCHANGED): the legacy TEXT-KEYED tier — MiniLM -> TF-IDF. ``embedder=
    'omni-embed'`` is a separate, explicit opt-in: routes texts through omni-embed's QUERY-side
    ``encode_query`` (``_omni_embed_query``), landing them in the SAME 2048-d space as
    ``embed_audio``'s omni-embed-built audio keys — for cross-modal (text query -> audio-keyed KB)
    retrieval. Not part of ``'auto'`` and has no ``fitted`` vectorizer (returns ``None`` for it).
    """
    import numpy as np

    if embedder == "omni-embed":
        name, kb = _omni_embed_query(texts, device=device)
        return name, kb, None

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

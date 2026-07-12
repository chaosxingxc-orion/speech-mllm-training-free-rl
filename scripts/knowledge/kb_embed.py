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

    16 kHz REQUIREMENT (2026-07-11, G2 CPU smoke + Phase-A KB build): the pipeline resamples any
    non-16 kHz input via ``torchaudio.sox_effects``, which this box's torchaudio build LACKS
    (AttributeError mid-call). Pre-resampling to 16 kHz mono here (librosa -> temp wav) skips that
    broken path entirely -- the fix the G2 CPU-smoke session validated (its ``smoke_sv_16k.py``:
    both embedders PASS on 16 kHz input), now applied at the one shared entry point instead of
    being every caller's job.
    """
    import os as _os
    import shutil
    import tempfile

    import librosa
    import numpy as np
    import soundfile as sf
    from modelscope.pipelines import pipeline

    d = _model_dir(name)
    sv = pipeline(task="speaker-verification", model=d, device=device)
    embs = []
    tmpdir = None
    try:
        for i, p in enumerate(wav_paths):
            if librosa.get_samplerate(p) == 16000:
                p16 = p
            else:
                if tmpdir is None:
                    tmpdir = tempfile.mkdtemp(prefix=f"kbembed_{name}_16k_")
                y, _sr = librosa.load(p, sr=16000, mono=True)
                p16 = _os.path.join(tmpdir, f"sv16k_{i}.wav")
                sf.write(p16, y, 16000)
            out = sv([p16], output_emb=True)
            embs.append(np.asarray(out["embs"][0], dtype="float32"))
    finally:
        # Clean up the resampled-16kHz scratch dir (if one was created) -- the original diff left
        # this leaking one temp dir per _modelscope_sv_embed call/embedder; cleaned up 2026-07-12
        # (RI item 7 pre-commit review) so a batch build over many non-16kHz sources doesn't
        # accumulate temp directories.
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
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
        # Payload shape FIXED 2026-07-11 (Phase-A step-1 live smoke): llama-server's /embeddings
        # endpoint rejects OpenAI-chat-style content parts ({"type": "input_audio", ...}) with a
        # 500 ('"prompt" elements must be a string, ... or a JSON object containing a prompt
        # string'). The shape its server.cpp actually accepts for multimodal embedding -- and the
        # one the 2026-07-09 G2 smoke that confirmed dim=2048 live actually used (embed_flow.sh)
        # -- is {"prompt_string": <media_marker>, "multimodal_data": [<b64>]}.
        payload = {"content": [{"prompt_string": media_marker, "multimodal_data": [b64]}]}
        req = urllib.request.Request(url + "/embeddings", data=json.dumps(payload).encode(),
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
        # transformers renamed ClapProcessor's `audios=` kwarg to `audio=` (newer versions RAISE
        # ValueError on the legacy name; older versions don't know the new one) -- try the new name
        # first, fall back to the legacy kwarg (2026-07-11, Phase-A KB build hit the ValueError
        # live on this box's transformers version).
        try:
            inp = proc(audio=audios[i : i + B], sampling_rate=48000, return_tensors="pt")
        except (TypeError, ValueError):
            inp = proc(audios=audios[i : i + B], sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            out = model.get_audio_features(**inp)
        # transformers >= 5 returns BaseModelOutputWithPooling whose pooler_output IS the
        # L2-normalized projected audio embedding (modeling_clap.py get_audio_features, verified
        # live on 5.12.1); older versions returned the tensor directly. Handle both.
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        embs.append(feats.cpu().numpy())
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


def _omni_embed_document_text(texts, device: str = "cpu"):
    """omni-embed-nemotron-3b's ``encode_document`` DOCUMENT side, over TEXT documents (2026-07-12,
    RI item 9) — the model's official multimodal input contract accepts ``{"text": ...}`` (as well
    as ``{"audio": ...}``/``{"video": ...}``; see the model card's usage example), so this is the
    SAME API/space ``_omni_embed`` uses for audio KEYS, just fed pure text instead. Backs
    ``kb_batch_build.build_squtr_corpus_source``'s TEXT-KEYED corpus-document build (squtr's
    corpus documents ship as text only — there is no audio for a corpus document, only for
    queries). Reachable via ``embed_text(embedder='omni-embed-nemotron')`` — a DIFFERENT registry
    token from ``embedder='omni-embed'`` (which is the QUERY side, ``_omni_embed_query`` above);
    this is the DOCUMENT/key side, matching the token ``_LOADER_FNS['omni-embed-nemotron']`` already
    uses for the audio-key path, so a manifest's ``embedder_token`` is the same string regardless of
    whether the KEY happened to be audio or text.
    """
    import numpy as np

    m = _omni_model(device)
    docs = [{"text": t} for t in texts]
    embs = m.encode_document(docs, convert_to_numpy=True, normalize_embeddings=True)
    return f"omni-embed-nemotron:{OMNI_EMBED_DIRNAME}", _l2(np.asarray(embs, dtype="float32"))


def _omni_embed_query_audio(wav_paths, device: str = "cpu"):
    """omni-embed QUERY-side embedder, AUDIO input (2026-07-13, kb_retrieve cross-modal routing
    fix -- M1 engineering-base item 2). The SAME ``encode_query`` tower ``_omni_embed_query`` uses
    for TEXT queries, fed ``{"audio": ...}`` dicts instead: the model's official multimodal input
    contract accepts audio/text/video dicts on ``encode_document`` (already exercised both ways —
    see ``_omni_embed``/``_omni_embed_document_text``); per that same API's documented symmetry,
    ``encode_query`` accepts the identical dict shapes. This lets an AUDIO query land in the exact
    query-tower space a TEXT-keyed source built via ``_omni_embed_document_text`` (registry token
    ``omni-embed-nemotron``) was persisted against — the cross-modal bridge
    ``kb_retrieve.retrieve_cross_modal_audio`` needs.

    NOT independently verified live this session (loading + running the ~4.7B model is a
    multi-minute CPU cold start — see ``scripts/knowledge/README.md``); if the installed
    checkpoint's ``encode_query`` rejects a raw ``{"audio": ...}`` dict, this raises the underlying
    error directly rather than masking it — treat as an open verification item (documented in
    ``kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS``'s docstring), not a silent guess.
    """
    import numpy as np

    m = _omni_model(device)
    docs = [{"audio": p} for p in wav_paths]
    embs = m.encode_query(docs, convert_to_numpy=True, normalize_embeddings=True)
    return f"omni-embed-nemotron:{OMNI_EMBED_DIRNAME}", _l2(np.asarray(embs, dtype="float32"))


def embed_query_crossmodal_audio(wav_paths, embedder: str, device: str = "cpu"):
    """AUDIO query embedded into a TEXT-keyed source's key space (2026-07-13, kb_retrieve
    cross-modal routing fix — M1 engineering-base item 2). The mirror-image of ``embed_text``'s
    existing cross-modal bridge (``embed_text(embedder='omni-embed')``: TEXT query -> AUDIO-keyed
    KB) but for the OTHER direction: AUDIO query -> TEXT-keyed KB (the gap
    ``kb_retrieve.py``'s 2026-07-12 docstring flagged as a known follow-up).

    Only the embedders ``kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS`` marks ``"supported"`` are
    dispatchable here — callers should consult that table (via
    ``kb_retrieve.retrieve_cross_modal_audio``, which does) rather than call this directly with an
    arbitrary token. Returns ``(name, matrix)`` — same shape as ``embed_audio``/``embed_text``.
    """
    if embedder == "glap":
        # GLAP is a JOINT audio-text space (like CLAP) -- _glap_embed (audio) and _glap_embed_text
        # already land in the SAME 1024-d cosine space (see _glap_embed_text's docstring), so the
        # plain AUDIO-key embedder IS the cross-modal query embedder here, unchanged.
        return _glap_embed(list(wav_paths), device=device)
    if embedder == "omni-embed-nemotron":
        return _omni_embed_query_audio(wav_paths, device=device)
    raise ValueError(
        f"kb_embed.embed_query_crossmodal_audio: {embedder!r} has no audio-query-into-text-key "
        "bridge wired (expected 'glap' or 'omni-embed-nemotron') -- see "
        "kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS for the full per-embedder status."
    )


def _glap_embed_text(texts, source_lang: str = "eng_Latn", device: str = "cpu"):
    """GLAP's native ``encode_text`` — GLAP (mispeech/GLAP) is a JOINT audio-text model (like CLAP),
    not audio-only: its ``GlapModel.encode_text(text, source_lang=...)`` (see
    ``modeling_glap.py``) lands text in the SAME 1024-d cosine space as ``_glap_embed``'s
    ``encode_audio`` (verified 2026-07-12, RI item 9 — the model ships an NLLB-family tokenizer +
    a dedicated ``SonarTextEncoder`` + ``text_proj`` head precisely for this). CPU-capable, no GPU
    server needed. ``source_lang`` follows GLAP's NLLB-style language codes (e.g. ``"eng_Latn"``
    for English, ``"zho_Hans"`` for simplified Chinese) — pick per the text's actual language;
    GLAP does NOT auto-detect it. Backs ``kb_batch_build.build_squtr_corpus_source``'s TEXT-KEYED
    corpus-document build for the ``'glap'`` embedder arm. Reachable via
    ``embed_text(embedder='glap', source_lang=...)``.
    """
    import numpy as np
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(_model_dir("glap"), trust_remote_code=True).to(device).eval()
    with torch.no_grad():
        e = model.encode_text(list(texts), source_lang=source_lang)
    return "glap:GLAP", _l2(e.cpu().numpy().astype("float32"))


# Embedders with a genuine TEXT-embedding path wired into `embed_text` (2026-07-12, RI item 9) --
# used by `build_squtr_corpus_source` to decide ARM-BLOCKED-cross-modal for an embedder that has
# none. Does NOT include lco-3b/7b/qwen3-omni-own: those are GPU-server embedders (needs_server) --
# whether they COULD embed text is a separate question from whether a server is up this session;
# callers should check `kb_embed.EMBEDDERS[x]["needs_server"]` for that, not this set.
TEXT_KEY_CAPABLE = {"glap", "omni-embed-nemotron", "omni-embed", "minilm", "auto"}


def can_embed_text(embedder: str) -> bool:
    """Whether ``embedder`` has a genuine TEXT-embedding path wired into ``embed_text`` (2026-07-12,
    RI item 9) — i.e. can key/embed a TEXT document with it, as opposed to needing an unavailable
    GPU server (a separate concern; see ``TEXT_KEY_CAPABLE``'s docstring note) or having no text
    tower at all (e.g. wavlm/eres2netv2/campplus/emotion2vec*/dasheng/clsp/sense/sensevoice-small
    and, in THIS repo's current wiring, clap — CLAP's own model does have a text tower upstream,
    but it is not wired into ``embed_text`` here; out of scope for this pass).
    """
    return embedder in TEXT_KEY_CAPABLE


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


def embedder_revision_of(embedder: str) -> str | None:
    """Best-effort "model name + checkpoint path/revision string" for ``embedder`` (2026-07-13,
    ticket #37 item 6: ``manifest.embedder_revision``). Never raises -- returns ``None`` when
    nothing is knowable offline (e.g. a server-based embedder whose exact served checkpoint isn't
    queryable without a live server, or a token this repo doesn't recognize at all).

    For a local on-disk snapshot (``MODEL_DIRNAMES``), the string is ``"<token>@<resolved dir>"``
    -- the resolved directory IS the checkpoint's identity for a pinned local snapshot (matches
    ``docs/datasets.lock.json``'s own pinned-revision convention: the directory only ever holds
    ONE revision at a time in this repo's data layout). For the two Hub-fetched
    legacy tiers (``clap``/``omni-embed``/``omni-embed-nemotron``), the string names the fixed
    upstream repo id this codebase hardcodes. For a GPU-server embedder, the string names the
    server env var (NOT a specific checkpoint hash -- the actual served revision is only knowable
    by querying the live server's ``/props``, out of scope for this offline helper).
    """
    if embedder in MODEL_DIRNAMES:
        try:
            d = _model_dir(embedder)
        except FileNotFoundError:
            return None
        return f"{embedder}@{d}"
    if embedder == "clap":
        return f"clap@{CLAP_MODEL}"
    if embedder in ("omni-embed", "omni-embed-nemotron"):
        return f"{embedder}@{OMNI_EMBED_DIRNAME}"
    meta = EMBEDDERS.get(embedder)
    if meta and meta.get("needs_server"):
        return f"{embedder}@server:{meta.get('server_env')}"
    return None


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


def embed_text(texts, embedder: str = "auto", device: str = "cpu", source_lang: str = "eng_Latn"):
    """TEXT embedder. Returns (name, matrix, fitted_or_None).

    ``'auto'`` (default, UNCHANGED): the legacy TEXT-KEYED tier — MiniLM -> TF-IDF. ``embedder=
    'omni-embed'`` is a separate, explicit opt-in: routes texts through omni-embed's QUERY-side
    ``encode_query`` (``_omni_embed_query``), landing them in the SAME 2048-d space as
    ``embed_audio``'s omni-embed-built audio keys — for cross-modal (text query -> audio-keyed KB)
    retrieval. Not part of ``'auto'`` and has no ``fitted`` vectorizer (returns ``None`` for it).

    ``embedder='omni-embed-nemotron'`` (2026-07-12, RI item 9) is the DOCUMENT side of that same
    asymmetric API (``_omni_embed_document_text``, ``encode_document`` over ``{"text": ...}``) —
    for keying a TEXT corpus (e.g. ``build_squtr_corpus_source``'s corpus documents), as opposed to
    ``'omni-embed'`` which is the query side. ``embedder='glap'`` (2026-07-12) routes through
    GLAP's native joint audio-text space (``_glap_embed_text``, ``encode_text``) — pass
    ``source_lang`` (NLLB-style code, e.g. ``"eng_Latn"``/``"zho_Hans"``) matching the text's
    actual language; GLAP does not auto-detect it. See ``can_embed_text``/``TEXT_KEY_CAPABLE`` for
    which embedder tokens have a text path wired at all.
    """
    import numpy as np

    if embedder == "omni-embed":
        name, kb = _omni_embed_query(texts, device=device)
        return name, kb, None

    if embedder == "omni-embed-nemotron":
        name, kb = _omni_embed_document_text(texts, device=device)
        return name, kb, None

    if embedder == "glap":
        name, kb = _glap_embed_text(texts, source_lang=source_lang, device=device)
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

"""CP-1 MULTIMODAL (b) — feature-audited acoustic conditioning on MMAU (GPU + FBank/MFCC audit).

MINI-PREREG (committed before generation; Stage-1 directional; owner corrections 2026-07-04):
  Audio semantics live in the high-order features (log-mel/FBank/MFCC), so leakage/equivalence is
  judged at the FEATURE level. This experiment (i) AUDITS each candidate acoustic transform for
  FBank invariance (cosine similarity of the time-averaged log-mel spectrum vs the original), (ii)
  keeps only feature-PRESERVING transforms as a semantically-equivalent presentation family, and
  (iii) measures, on the SAME MMAU n=150 slice as E3, both a leakage-free per-transform UNIFORM
  greedy accuracy (M1: is a single deployable acoustic conditioning a lever?) and — only over the
  audited feature-invariant subset — oracle-over-K (legitimate, since those variants are
  semantically equivalent, exactly like text rewording).

  Transforms audited: [original, rms_norm, preemphasis, denoise, telephone, speed0.9, speed1.1, trim].
  Feature-invariance metric: cosine similarity of the mean log-mel (128 mel bands) of transformed vs
  original; INVARIANT iff sim >= FEAT_TAU (default 0.98). Length-changing transforms (speed/trim) are
  compared on the time-averaged mel (length-robust). The audit is REPORTED alongside every number; no
  oracle is taken over feature-altering transforms (that would be content leakage / cherry-picking).
  Readouts: per-transform mel-similarity + uniform greedy accuracy; oracle-over-invariant-subset vs
  oracle_fix (sampling) — the multimodal-conditioning contribution within the invariant manifold.
  Grade: [directional | n=150 | single-touch | not significance-bearing].
  Budget: audit is CPU; GPU = 150 x (8 transforms greedy + already-have E3 sampling) ~ 1,200 gens, ~30 min.

Requires the resident llama-server.
reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp1_multimodal_feature_audited_mmau.py
"""
import base64, collections, glob, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
N_UTTS = int(os.environ.get("M3_N", "150"))
TEMP = float(os.environ.get("CP1_TEMP", "0.7"))
MAXTOK = 48
SLICE_SEED = 20260704
FEAT_TAU = float(os.environ.get("FEAT_TAU", "0.98"))
OUT = DATA / "_repro" / "cp1_multimodal_feature_audited_mmau.json"
BASE_WAV = DATA / "_repro" / "cp1_mmau_wavs"
WAV_DIR = DATA / "_repro" / "cp1_mmau_mmaudit_wavs"
NBOOT = 10000
LETTERS = ["A", "B", "C", "D", "E", "F"]
VARIANTS = ["original", "rms_norm", "preemphasis", "denoise", "telephone", "speed0.9", "speed1.1", "trim"]


def _rms_norm(y, sr):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    return np.clip(y * (10 ** (-20.0 / 20.0) / rms), -1.0, 1.0).astype("float32")

def _preemph(y, sr):
    return np.append(y[0], y[1:] - 0.97 * y[:-1]).astype("float32")

def _denoise(y, sr):
    import scipy.signal as ss
    f, tt, Z = ss.stft(y, sr, nperseg=512)
    mag, ph = np.abs(Z), np.angle(Z)
    mag = np.maximum(mag - np.median(mag, axis=1, keepdims=True) * 1.5, 0.0)
    _, yr = ss.istft(mag * np.exp(1j * ph), sr, nperseg=512)
    return yr.astype("float32")

def _telephone(y, sr):
    import scipy.signal as ss
    b, a = ss.butter(4, [300 / (sr / 2), min(3400, sr / 2 - 100) / (sr / 2)], btype="band")
    return ss.lfilter(b, a, y).astype("float32")

def _speed(f):
    def g(y, sr):
        import librosa
        return librosa.effects.time_stretch(y, rate=f).astype("float32")
    return g

def _trim(y, sr):
    import librosa
    yt, _ = librosa.effects.trim(y, top_db=25)
    return (yt if len(yt) > sr * 0.2 else y).astype("float32")

TRANSFORMS = [lambda y, sr: y.astype("float32"), _rms_norm, _preemph, _denoise, _telephone,
              _speed(0.9), _speed(1.1), _trim]


def mean_logmel(y, sr):
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return librosa.power_to_db(S + 1e-10).mean(axis=1)   # time-averaged, length-robust


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fmt_choices(choices):
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def instr_for(choices):
    return (f"Listen to the audio and answer the multiple-choice question. Choose the single best option.\n"
            f"Options:\n{fmt_choices(choices)}\nAnswer with only the option letter and text, e.g. 'A. ...'.")


def parse_choice(text, choices):
    t = text.strip().lower()
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()
    cn = [norm(c) for c in choices]
    for i in range(len(choices)):
        L = LETTERS[i].lower()
        if t[:2] in (L + ".", L + ")", L + ":", L + " ") or t.strip() == L:
            return i
    for i in sorted(range(len(choices)), key=lambda k: -len(cn[k])):
        if cn[i] and cn[i] in norm(text):
            return i
    toks = set(norm(text).split()); best, bi = 0, -1
    for i in range(len(choices)):
        ov = len(toks & set(cn[i].split()))
        if ov > best:
            best, bi = ov, i
    return bi


def load_slice():
    import pyarrow.parquet as pq
    import soundfile as sf
    fs = sorted(glob.glob(str(DATA / "datasets/mmau-mini/data/test_mini*.parquet")))
    rows = []
    for f in fs:
        rows.extend(pq.read_table(f, columns=["question", "choices", "answer", "task", "audio"]).to_pylist())
    seen, uniq = set(), []
    for r in rows:
        k = (r["question"], r["task"], tuple(r["choices"]))
        if k not in seen:
            seen.add(k); uniq.append(r)
    idx = np.random.default_rng(SLICE_SEED).permutation(len(uniq))[:N_UTTS]
    BASE_WAV.mkdir(parents=True, exist_ok=True)
    out = []
    for j in idx:
        r = uniq[int(j)]
        wp = BASE_WAV / f"mmau_{int(j)}.wav"
        if not wp.exists():
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            sf.write(str(wp), wav, sr)
        gt = [i for i, c in enumerate(r["choices"]) if c == r["answer"]]
        if gt:
            out.append({"base": str(wp), "idx": int(j), "choices": list(r["choices"]), "gt": gt[0], "task": r["task"]})
    return out


def build_variants(u):
    import soundfile as sf
    y, sr = sf.read(u["base"], dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    base_mel = mean_logmel(y, sr)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    paths, sims = [], []
    for name, tf in zip(VARIANTS, TRANSFORMS):
        wp = WAV_DIR / f"mmau_{u['idx']}__{name}.wav"
        try:
            yv = tf(y, sr)
            if yv.size == 0 or not np.isfinite(yv).all():
                yv = y.astype("float32")
        except Exception:
            yv = y.astype("float32")
        if not wp.exists():
            sf.write(str(wp), yv, sr)
        sims.append(cos(base_mel, mean_logmel(yv, sr)))
        paths.append(str(wp))
    return paths, sims


def gen(clip, instruction, seed=42):
    b64 = base64.b64encode(open(clip, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instruction}]}], "max_tokens": MAXTOK, "seed": seed, "temperature": 0.0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def cboot(vals, seed=42):
    keys = sorted(vals); arr = [vals[k] for k in keys]
    rng = np.random.default_rng(seed)
    ms = np.array([float(np.mean([arr[i] for i in rng.integers(0, len(arr), len(arr))])) for _ in range(NBOOT)])
    return [round(float(np.quantile(ms, 0.025)), 4), round(float(np.quantile(ms, 0.975)), 4)]


def main():
    t0 = time.time()
    utts = load_slice()
    print(f"server={BASE} n={len(utts)} variants={VARIANTS} FEAT_TAU={FEAT_TAU}", flush=True)
    rows, skipped = [], []
    sim_acc = collections.defaultdict(list)   # per-variant mel similarities
    for k, u in enumerate(utts):
        ch, gt = u["choices"], u["gt"]
        vpaths, sims = build_variants(u)
        instr = instr_for(ch)
        try:
            preds = [parse_choice(gen(vpaths[vi], instr), ch) for vi in range(len(VARIANTS))]
        except Exception as e:
            print(f"  [{k+1}] SKIP ({type(e).__name__}: {str(e)[:70]})", flush=True); skipped.append(k); continue
        for vi, name in enumerate(VARIANTS):
            sim_acc[name].append(sims[vi])
        rows.append({"task": u["task"], "gt": gt, "sims": sims, "preds": preds})
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(utts)}] orig_acc={np.mean([r['preds'][0]==r['gt'] for r in rows]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # feature-invariance audit + per-variant uniform accuracy
    audit = {}
    for vi, name in enumerate(VARIANTS):
        msim = float(np.mean(sim_acc[name]))
        acc = float(np.mean([r["preds"][vi] == r["gt"] for r in rows]))
        audit[name] = {"mean_mel_cosine": round(msim, 4), "feature_invariant": bool(msim >= FEAT_TAU),
                       "uniform_greedy_acc": round(acc, 4)}
    invariant = [vi for vi, name in enumerate(VARIANTS) if audit[name]["feature_invariant"]]
    orig_acc = audit["original"]["uniform_greedy_acc"]
    # oracle-over-invariant-subset (LEGITIMATE: semantically-equivalent presentations only)
    oracle_inv = float(np.mean([int(r["gt"] in [r["preds"][vi] for vi in invariant]) for r in rows])) if invariant else orig_acc
    H_mm_invariant = round(oracle_inv - orig_acc, 4)
    ci = cboot({i: int(r["gt"] in [r["preds"][vi] for vi in invariant]) - int(r["preds"][0] == r["gt"])
                for i, r in enumerate(rows)}) if len(invariant) > 1 else [0.0, 0.0]
    summary = {
        "problem": "CP-1 MULTIMODAL (b) feature-audited — acoustic conditioning on MMAU (frozen Qwen3-Omni-30B)",
        "stage": "1-directional", "grade": "[directional | n=%d | single-touch | not significance-bearing]" % len(rows),
        "n_utts": len(rows), "n_skipped": len(skipped), "FEAT_TAU": FEAT_TAU, "slice_seed": SLICE_SEED,
        "feature_audit": audit,
        "invariant_transforms": [VARIANTS[vi] for vi in invariant],
        "original_greedy_acc": orig_acc,
        "oracle_over_invariant_subset": round(oracle_inv, 4),
        "H_mm_over_invariant_manifold": H_mm_invariant,
        "ci_oracle_inv_minus_original_descriptive": ci,
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "note": ("oracle is taken ONLY over feature-invariant transforms (semantically-equivalent presentations); "
                 "feature-ALTERING transforms are audited + their uniform accuracy reported, but never oracle-pooled "
                 "(that would be content leakage / cherry-picking). Semantics judged at the FBank/mel level per owner."),
        "prereg": "module docstring, committed before generation (git history is the anchor)",
    }
    summary["verdict"] = (
        f"feature-invariant transforms (mel-cos>={FEAT_TAU}): {summary['invariant_transforms']}. "
        f"Over the invariant manifold, oracle {oracle_inv:.3f} vs original greedy {orig_acc:.3f} -> "
        f"H_mm = {H_mm_invariant:+.4f} (CI {ci}). Per-transform uniform greedy acc: "
        f"{ {n: audit[n]['uniform_greedy_acc'] for n in VARIANTS} }. Directional: within the leakage-free "
        f"feature-invariant family, acoustic conditioning "
        f"{'opens headroom' if ci[0] > 0 else 'does NOT open headroom'} beyond the fixed presentation; "
        f"feature-altering transforms (audited) are reported for robustness only, not as a conditioning gain.")
    out = {"summary": summary, "per_utt": [{"task": r["task"], "gt": r["gt"], "preds": r["preds"],
                                             "sims": [round(s, 3) for s in r["sims"]]} for r in rows],
           "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp1_multimodal_feature_audited_mmau.py (llama-server resident)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("CP1_MM_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()

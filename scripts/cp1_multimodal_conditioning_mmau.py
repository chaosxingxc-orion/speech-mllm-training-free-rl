"""CP-1 MULTIMODAL-conditioning arm — does acoustic signal injection move the rollout? (GPU).

MINI-PREREG (committed before generation; Stage-1 directional check; owner steer 2026-07-04):
  E1/E3 measured H_prompt - H_fix over the TEXT sub-channel only and found it ~ 0. For an OMNI model
  the conditioning c includes MULTIMODAL signal injection. Question: on the SAME non-saturated MMAU
  slice as E3, does varying the ACOUSTIC conditioning (denoise/normalize/speed/filter/trim) open oracle
  headroom beyond sampling the original — i.e. H_mm - H_fix — and how SENSITIVE is the rollout to
  acoustic conditioning (does the prediction shift when we perturb the signal)?

  Slice: mmau-mini test_mini, SAME seed-frozen n=150 as E3 (directly paired). Fixed text instruction
  (the E3 'fix' prompt). Arms matched at BUDGET=8:
    - greedy   : original audio, temp 0
    - FIX      : original audio x 8 temp samples          -> oracle_fix (the sampling-support baseline)
    - MULTIMODAL: 8 acoustic conditioning variants x 1 sample each -> oracle-over-8-acoustic
      variants: [0 original, 1 RMS-normalize, 2 pre-emphasis, 3 spectral-denoise, 4 speed 0.9x,
                 5 speed 1.1x, 6 telephone band-pass 300-3400 Hz, 7 trim silence]
  Readouts (per-utt MCQ-correct 0/1):
    greedy/oracle_fix/oracle_multimodal acc; H_fix = oracle_fix - greedy; H_mm = oracle_mm - greedy;
    **H_mm - H_fix** (the multimodal-conditioning contribution beyond sampling);
    rollout sensitivity = mean fraction of variants whose greedy prediction differs from the original's;
    per-variant greedy accuracy + flip rate; paired cluster-bootstrap 95% CIs (descriptive).
  Grade: [directional | n=150 | single-touch | not significance-bearing].
  Budget: 150 x (1 + 8 + 8) = 2,550 short-output generations, ~40-60 min GPU.

Requires the resident llama-server.
reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp1_multimodal_conditioning_mmau.py
"""
import base64, collections, glob, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
N_UTTS = int(os.environ.get("M3_N", "150"))
BUDGET = 8
TEMP = float(os.environ.get("CP1_TEMP", "0.7"))
MAXTOK = 48
SLICE_SEED = 20260704  # SAME as E3 -> paired items
OUT = DATA / "_repro" / "cp1_multimodal_conditioning_mmau.json"
WAV_DIR = DATA / "_repro" / "cp1_mmau_mm_wavs"
NBOOT = 10000
LETTERS = ["A", "B", "C", "D", "E", "F"]
VARIANTS = ["original", "rms_norm", "preemphasis", "denoise", "speed0.9", "speed1.1", "telephone", "trim"]


# ---------- acoustic conditioning transforms (each: wav float32, sr -> wav float32) ----------
def _rms_norm(y, sr, target_dbfs=-20.0):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    return np.clip(y * (10 ** (target_dbfs / 20.0) / rms), -1.0, 1.0).astype("float32")

def _preemph(y, sr, a=0.97):
    return np.append(y[0], y[1:] - a * y[:-1]).astype("float32")

def _denoise(y, sr):
    import scipy.signal as ss
    f, tt, Z = ss.stft(y, sr, nperseg=512)
    mag, ph = np.abs(Z), np.angle(Z)
    noise = np.median(mag, axis=1, keepdims=True) * 1.5  # crude noise floor
    mag = np.maximum(mag - noise, 0.0)
    _, yr = ss.istft(mag * np.exp(1j * ph), sr, nperseg=512)
    return yr.astype("float32")

def _speed(factor):
    def f(y, sr):
        import librosa
        return librosa.effects.time_stretch(y, rate=factor).astype("float32")
    return f

def _telephone(y, sr):
    import scipy.signal as ss
    lo, hi = 300.0, min(3400.0, sr / 2.0 - 100)
    b, a = ss.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return ss.lfilter(b, a, y).astype("float32")

def _trim(y, sr):
    import librosa
    yt, _ = librosa.effects.trim(y, top_db=25)
    return (yt if len(yt) > sr * 0.2 else y).astype("float32")

TRANSFORMS = [lambda y, sr: y.astype("float32"), _rms_norm, _preemph, _denoise,
              _speed(0.9), _speed(1.1), _telephone, _trim]


def build_variants(base_wav):
    """base_wav = path to the original; write the 8 conditioned variants once, return their paths."""
    import soundfile as sf
    y, sr = sf.read(base_wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    paths = []
    stem = Path(base_wav).stem
    for vi, (name, tf) in enumerate(zip(VARIANTS, TRANSFORMS)):
        wp = WAV_DIR / f"{stem}__{name}.wav"
        if not wp.exists():
            try:
                yv = tf(y, sr)
            except Exception:
                yv = y.astype("float32")
            if yv.size == 0 or not np.isfinite(yv).all():
                yv = y.astype("float32")
            sf.write(str(wp), yv, sr)
        paths.append(str(wp))
    return paths


def fmt_choices(choices):
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def fix_instruction(choices):
    return (f"Listen to the audio and answer the multiple-choice question. Choose the single best option.\n"
            f"Options:\n{fmt_choices(choices)}\nAnswer with only the option letter and text, e.g. 'A. ...'.")


def parse_choice(text, choices):
    t = text.strip().lower()
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()
    ch_norm = [norm(c) for c in choices]
    for i in range(len(choices)):
        L = LETTERS[i].lower()
        if t[:2] in (L + ".", L + ")", L + ":", L + " ") or t.strip() == L:
            return i
    for i in sorted(range(len(choices)), key=lambda k: -len(ch_norm[k])):
        if ch_norm[i] and ch_norm[i] in norm(text):
            return i
    toks = set(norm(text).split())
    best, bi = 0, -1
    for i in range(len(choices)):
        ov = len(toks & set(ch_norm[i].split()))
        if ov > best:
            best, bi = ov, i
    return bi


def load_slice():
    import pyarrow.parquet as pq
    import soundfile as sf
    fs = sorted(glob.glob(str(DATA / "datasets/mmau-mini/data/test_mini*.parquet")))
    cols = ["question", "choices", "answer", "task", "audio"]
    rows = []
    for f in fs:
        rows.extend(pq.read_table(f, columns=cols).to_pylist())
    seen, uniq = set(), []
    for r in rows:
        k = (r["question"], r["task"], tuple(r["choices"]))
        if k not in seen:
            seen.add(k); uniq.append(r)
    idx = np.random.default_rng(SLICE_SEED).permutation(len(uniq))[:N_UTTS]
    base_dir = DATA / "_repro" / "cp1_mmau_wavs"   # E3 already wrote the originals here
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for j in idx:
        r = uniq[int(j)]
        base = base_dir / f"mmau_{int(j)}.wav"
        if not base.exists():
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            base.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(base), wav, sr)
        gt = [i for i, c in enumerate(r["choices"]) if c == r["answer"]]
        if not gt:
            continue
        out.append({"base": str(base), "choices": list(r["choices"]), "gt": gt[0], "task": r["task"]})
    return out


def gen(clip, instruction, greedy, seed):
    b64 = base64.b64encode(open(clip, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instruction}]}],
        "max_tokens": MAXTOK, "seed": seed, "temperature": 0.0 if greedy else TEMP}
    if not greedy:
        payload["top_p"] = 0.95
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
    print(f"server={BASE} n={len(utts)} budget={BUDGET} variants={VARIANTS}", flush=True)
    rows, skipped = [], []
    var_correct = collections.Counter(); var_total = collections.Counter(); var_flip = collections.Counter()
    for k, u in enumerate(utts):
        ch, gt, instr = u["choices"], u["gt"], fix_instruction(u["choices"])
        vpaths = build_variants(u["base"])
        try:
            g = parse_choice(gen(vpaths[0], instr, True, 42), ch)              # original greedy
            fix = [parse_choice(gen(vpaths[0], instr, False, 100 + i), ch) for i in range(BUDGET)]  # original x8 temp
            mm_greedy = [parse_choice(gen(vpaths[vi], instr, True, 42), ch) for vi in range(BUDGET)]  # 8 variants, greedy each
        except Exception as e:
            print(f"  [{k+1}] SKIP ({type(e).__name__}: {str(e)[:80]})", flush=True); skipped.append(k); continue
        for vi in range(BUDGET):
            var_total[vi] += 1
            var_correct[vi] += int(mm_greedy[vi] == gt)
            if vi > 0 and mm_greedy[vi] != mm_greedy[0]:
                var_flip[vi] += 1
        rows.append({"task": u["task"], "gt": gt, "greedy_correct": int(g == gt),
                     "oracle_fix": int(gt in fix), "oracle_mm": int(gt in mm_greedy),
                     "sensitivity": np.mean([mm_greedy[vi] != mm_greedy[0] for vi in range(1, BUDGET)])})
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(utts)}] greedy={np.mean([r['greedy_correct'] for r in rows]):.3f} "
                  f"o_fix={np.mean([r['oracle_fix'] for r in rows]):.3f} "
                  f"o_mm={np.mean([r['oracle_mm'] for r in rows]):.3f} "
                  f"sens={np.mean([r['sensitivity'] for r in rows]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    m = lambda key: round(float(np.mean([r[key] for r in rows])), 4)
    idxd = lambda f: {i: f(r) for i, r in enumerate(rows)}
    greedy, ofix, omm = m("greedy_correct"), m("oracle_fix"), m("oracle_mm")
    summary = {
        "problem": "CP-1 MULTIMODAL-conditioning arm — H_mm - H_fix on MMAU-mini (frozen Qwen3-Omni-30B)",
        "stage": "1-directional", "grade": "[directional | n=%d | single-touch | not significance-bearing]" % len(rows),
        "n_utts": len(rows), "n_skipped": len(skipped), "budget_per_arm": BUDGET, "temp": TEMP,
        "dataset": "mmau-mini test_mini (SAME n=150 slice as E3)", "slice_seed": SLICE_SEED,
        "variants": VARIANTS,
        "acc": {"greedy_original": greedy, "oracle_fix_sampling": ofix, "oracle_multimodal_acoustic": omm},
        "H_fix": round(ofix - greedy, 4), "H_mm": round(omm - greedy, 4),
        "H_mm_minus_H_fix": round(omm - ofix, 4),
        "ci_mm_minus_fix_descriptive": cboot(idxd(lambda r: r["oracle_mm"] - r["oracle_fix"])),
        "rollout_sensitivity_to_acoustic": round(m("sensitivity"), 4),
        "per_variant_greedy_acc": {VARIANTS[vi]: round(var_correct[vi] / max(1, var_total[vi]), 4) for vi in range(BUDGET)},
        "per_variant_flip_rate_vs_original": {VARIANTS[vi]: round(var_flip[vi] / max(1, var_total[vi]), 4) for vi in range(1, BUDGET)},
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "prereg": "module docstring, committed before generation (git history is the anchor)",
    }
    ci = summary["ci_mm_minus_fix_descriptive"]
    summary["verdict"] = (
        f"greedy(orig) {greedy:.3f}; oracle_fix (sampling) {ofix:.3f} vs oracle_multimodal (acoustic) {omm:.3f} "
        f"-> H_mm - H_fix = {omm-ofix:+.4f} (CI {ci}). Rollout sensitivity to acoustic conditioning = "
        f"{summary['rollout_sensitivity_to_acoustic']:.3f} (fraction of variants flipping the greedy pick). "
        f"Directional: acoustic conditioning {'DOES open headroom beyond sampling' if ci[0] > 0 else 'does not clearly open headroom beyond sampling'}; "
        f"the model is {'sensitive' if summary['rollout_sensitivity_to_acoustic'] > 0.15 else 'largely insensitive'} to acoustic signal injection.")
    out = {"summary": summary, "per_utt": rows, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp1_multimodal_conditioning_mmau.py (llama-server resident)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("CP1_MM_DONE", flush=True)


if __name__ == "__main__":
    main()

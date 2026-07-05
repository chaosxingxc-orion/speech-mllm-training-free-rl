"""M3 — cross-modal injection: does adding an INDEPENDENT text transcript alongside the audio realize
the headroom that audio-only ICL could not? (the first genuinely-new-signal lever; tests the Q1b verdict).

For surfaces where the audio is the spoken question AND a ground-truth text transcript exists, compare,
greedy, with paired-bootstrap CIs:
  audio-only  vs  audio + ground-truth-text-transcript.
The transcript is the dataset's ground-truth text (external to M), so a gain is a genuinely new signal —
and it localizes whether the frozen omni's realizable gap is audio-perception (fixed by text) vs reasoning.

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/m3_crossmodal.py
"""
import base64, glob, io, json, os, sys, time, urllib.request
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
import p2_baselines as p2

BASE = p2.BASE
N = int(os.environ.get("M3_NQUERY", "60"))
SEED = 20260705
NBOOT = 10000
OUT = p2.DATA / "_repro" / "m3_crossmodal.json"
DS = p2.DS
WAV = p2.WAV


def gen(wav, instr):
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    p = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instr}]}], "max_tokens": p2.MAXTOK, "seed": 42, "temperature": 0.0}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as z:
        return json.loads(z.read().decode())["choices"][0]["message"]["content"].strip()


def load_squad_zh(rng):
    f = next((DS / "uro-bench/SQuAD-zh").glob("*.parquet"))
    rows = pq.read_table(f).to_pylist()
    out = []
    for j in rng.permutation(len(rows))[:N]:
        r = rows[int(j)]
        wp = WAV / f"sqzh_{int(j)}.wav"
        if not wp.exists():
            WAV.mkdir(parents=True, exist_ok=True)
            y, sr = sf.read(io.BytesIO(r["source_wav"]["bytes"]), dtype="float32")
            sf.write(str(wp), y if y.ndim == 1 else y.mean(1), sr)
        out.append({"wav": str(wp), "transcript": r["source_text"], "gold": str(r["target_text"]), "task": "qa"})
    return out


def load_vocalbench_zh(rng):
    fs = sorted((DS / "vocalbench-zh").rglob("*.parquet"))
    rows = []
    for ff in fs:
        cols = pq.read_table(ff).column_names
        if "Answer" in cols and "audio" in cols and "Question" in cols:
            rows += [r for r in pq.read_table(ff).to_pylist() if r.get("Answer") and r.get("audio")]
    out = []
    for j in rng.permutation(len(rows))[:N]:
        r = rows[int(j)]
        wp = WAV / f"vbzh_{int(j)}.wav"
        if not wp.exists():
            WAV.mkdir(parents=True, exist_ok=True)
            y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            sf.write(str(wp), y if y.ndim == 1 else y.mean(1), sr)
        out.append({"wav": str(wp), "transcript": r["Question"], "gold": str(r["Answer"]), "task": "qa"})
    return out


def boot(d, seed=1):
    rng = np.random.default_rng(seed)
    ms = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(NBOOT)])
    return [round(float(np.quantile(ms, .025)), 4), round(float(np.quantile(ms, .975)), 4)]


AUDIO_ONLY = "Listen to the spoken question and answer with a short answer only."
WITH_TEXT = "Transcript of the spoken question: \"{t}\"\nUsing the audio and the transcript, answer with a short answer only."


def run(name, loader):
    rng = np.random.default_rng(SEED)
    items = loader(rng)
    t0 = time.time()
    a, at = [], []
    for k, it in enumerate(items):
        try:
            ra = p2.reward(it, gen(it["wav"], AUDIO_ONLY))
            rt = p2.reward(it, gen(it["wav"], WITH_TEXT.format(t=it["transcript"])))
        except Exception:
            continue
        a.append(ra); at.append(rt)
        if (k + 1) % 30 == 0:
            print(f"  [{name} {k+1}] audio_only={np.mean(a):.3f} audio+text={np.mean(at):.3f} ({time.time()-t0:.0f}s)", flush=True)
    a, at = np.array(a), np.array(at)
    res = {"n": len(a), "audio_only": round(float(a.mean()), 4), "audio_plus_text": round(float(at.mean()), 4),
           "gain": round(float((at - a).mean()), 4), "gain_CI": boot(at - a),
           "rel_gain": round(float((at.mean() - a.mean()) / max(1e-9, a.mean())), 4)}
    print(f"  == {name}: audio_only={res['audio_only']} audio+text={res['audio_plus_text']} "
          f"gain={res['gain']:+.3f} CI{res['gain_CI']} rel={res['rel_gain']:+.1%}", flush=True)
    return res


def main():
    print(f"server={BASE} n={N}", flush=True)
    loaders = {"SQuAD-zh": load_squad_zh, "vocalbench-zh": load_vocalbench_zh}
    out = {}
    for name, ld in loaders.items():
        try:
            out[name] = run(name, ld)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "lever": "M3 cross-modal (ground-truth transcript) injection",
                               "n_query": N, "seed": SEED}, "results": out,
                   "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/m3_crossmodal.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== M3 CROSS-MODAL ===\n" + json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("M3_DONE", flush=True)


if __name__ == "__main__":
    main()

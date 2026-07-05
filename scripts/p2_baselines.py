"""P2 — Stage-1 baseline sweep: greedy vs oracle-over-N on the ready zh+en coverage set.

For each dataset (MCQ / short-answer QA / SLU intent), under ONE fixed instruction:
  - greedy (temp 0, 1 decode) accuracy
  - oracle-over-N (N samples @ temp 0.7): 1 iff any sample is correct  -> the (a)-support ceiling
  - oracle_delta = oracle - greedy   (the latent headroom every A-lever is later measured against)
  - saturated flag (greedy >= 0.90 -> demote to easy-anchor per P1)

This freezes the per-surface baselines the A-realization program (E7/E8/E9/E10) optimizes toward.
Stage-1 directional: n small per set; grade [directional | small-n | not significance-bearing].

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/p2_baselines.py [--only mmau-mini,vocalbench-zh]
"""
import base64, io, json, os, re, sys, time, unicodedata, urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
DS = DATA / "datasets"
N_UTTS = int(os.environ.get("P2_N", "150"))
NSAMP = int(os.environ.get("P2_NSAMP", "8"))
TEMP = float(os.environ.get("P2_TEMP", "0.7"))
MAXTOK = 64
SLICE_SEED = 20260705
WAV = DATA / "_repro" / "p2_wavs"
OUT = DATA / "_repro" / "p2_baselines.json"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
MINDS14 = ["abroad", "address", "app_error", "atm_limit", "balance", "business_loan", "card_issues",
           "cash_deposit", "direct_debit", "freeze", "high_value_payment", "joint_account",
           "latest_transactions", "pay_bill"]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return "".join(c for c in s if c.isalnum() or c.isspace()).strip()


def gen(wav_path, instruction, seed, temp):
    b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instruction}]}],
        "max_tokens": MAXTOK, "seed": seed, "temperature": temp}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def _wav_from_bytes(b, key):
    WAV.mkdir(parents=True, exist_ok=True)
    p = WAV / f"{key}.wav"
    if not p.exists():
        y, sr = sf.read(io.BytesIO(b), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        sf.write(str(p), y, sr)
    return str(p)


# ---- per-dataset loaders: return list of {wav, instr, gold, task} ----
def _mcq_instr(question, options):
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (f"Listen to the audio and answer the multiple-choice question.\n{question}\n"
            f"Options:\n{body}\nAnswer with only the option letter and text, e.g. 'A. ...'.")


def _parse_choice(text, options):
    t = text.strip().lower()
    for i in range(len(options)):
        L = LETTERS[i].lower()
        if t[:2] in (L + ".", L + ")", L + ":", L + " ") or t.strip() == L:
            return i
    cn = [norm(o) for o in options]
    for i in sorted(range(len(options)), key=lambda k: -len(cn[k])):
        if cn[i] and cn[i] in norm(text):
            return i
    return -1


def load_mmau(rng):
    fs = sorted((DS / "mmau-mini/data").glob("test_mini*.parquet"))
    rows = []
    for f in fs:
        rows += pq.read_table(f, columns=["question", "choices", "answer", "audio"]).to_pylist()
    seen, uniq = set(), []
    for r in rows:
        k = (r["question"], tuple(r["choices"]))
        if k not in seen:
            seen.add(k); uniq.append(r)
    out = []
    for j in rng.permutation(len(uniq))[:N_UTTS]:
        r = uniq[int(j)]; gt = [i for i, c in enumerate(r["choices"]) if c == r["answer"]]
        if gt:
            out.append({"wav": _wav_from_bytes(r["audio"]["bytes"], f"mmau_{int(j)}"),
                        "instr": _mcq_instr(r["question"], r["choices"]), "gold": gt[0],
                        "opts": list(r["choices"]), "task": "mcq"})
    return out


def load_openbookqa_zh(rng):
    f = next((DS / "uro-bench/OpenbookQA-zh").glob("*.parquet"))
    rows = pq.read_table(f).to_pylist()
    out = []
    for j in rng.permutation(len(rows))[:N_UTTS]:
        r = rows[int(j)]
        # options embedded in source_text like "...A. x B. y C. z D. w"; gold letter in target_text
        m = re.split(r"(?=[A-D][\.\、])", r["source_text"])
        opts = [re.sub(r"^[A-D][\.\、]\s*", "", s).strip() for s in m if re.match(r"^[A-D][\.\、]", s.strip())]
        gl = r["target_text"].strip()[0].upper()
        gi = LETTERS.index(gl) if gl in LETTERS[:len(opts)] else -1
        if opts and gi >= 0:
            q = m[0].strip()
            out.append({"wav": _wav_from_bytes(r["source_wav"]["bytes"], f"oqa_{int(j)}"),
                        "instr": _mcq_instr(q, opts), "gold": gi, "opts": opts, "task": "mcq"})
    return out


def _qa_row(rng, dsdir, wav_key, audio_col, q_col, a_col, prefix):
    fs = sorted((DS / dsdir).rglob("*.parquet"))
    rows = []
    for ff in fs:
        try:
            cols = pq.read_table(ff).column_names
        except Exception:
            continue
        if a_col not in cols or audio_col not in cols:
            continue  # skip files with a different schema (e.g. non-verifiable splits)
        rows += [r for r in pq.read_table(ff).to_pylist() if r.get(a_col) and r.get(audio_col)]
    out = []
    for j in rng.permutation(len(rows))[:N_UTTS]:
        r = rows[int(j)]
        audio = r[audio_col]
        b = audio["bytes"] if isinstance(audio, dict) else audio
        q = r[q_col] if q_col else None
        instr = (f"{q}\nAnswer the question above based on the audio, with a short answer only."
                 if q_col == "instruction"
                 else "Listen and answer with a short answer only." if q is None
                 else f"Listen to the spoken question and answer with a short answer only.")
        out.append({"wav": _wav_from_bytes(b, f"{prefix}_{int(j)}"), "instr": instr,
                    "gold": str(r[a_col]), "task": "qa"})
    return out


def load_vocalbench_zh(rng):
    return _qa_row(rng, "vocalbench-zh", None, "audio", None, "Answer", "vbzh")


def load_squad_zh(rng):
    return _qa_row(rng, "uro-bench/SQuAD-zh", None, "source_wav", None, "target_text", "sqzh")


def load_spoken_squad(rng):
    return _qa_row(rng, "spoken-squad", None, "context", "instruction", "answer", "ssq")


def load_minds14_zh(rng):
    f = next((DS / "minds14/zh-CN").glob("*.parquet"))
    rows = pq.read_table(f, columns=["audio", "intent_class"]).to_pylist()
    opts = "\n".join(f"{i}. {n}" for i, n in enumerate(MINDS14))
    instr = ("Listen to the banking request and classify its intent. Choose ONE intent by number.\n"
             f"Intents:\n{opts}\nAnswer with only the intent number and name, e.g. '4. balance'.")
    out = []
    for j in rng.permutation(len(rows))[:N_UTTS]:
        r = rows[int(j)]
        out.append({"wav": _wav_from_bytes(r["audio"]["bytes"], f"m14zh_{int(j)}"),
                    "instr": instr, "gold": int(r["intent_class"]), "task": "slu"})
    return out


def reward(item, text):
    if item["task"] == "mcq":
        return int(_parse_choice(text, item["opts"]) == item["gold"])
    if item["task"] == "qa":
        g = norm(item["gold"])
        return int(bool(g) and g in norm(text))
    if item["task"] == "slu":
        t = text.strip()
        m = re.match(r"\s*(\d+)", t)
        if m and int(m.group(1)) == item["gold"]:
            return 1
        return int(norm(MINDS14[item["gold"]]) in norm(text))
    return 0


LOADERS = {"mmau-mini": load_mmau, "OpenbookQA-zh": load_openbookqa_zh,
           "vocalbench-zh": load_vocalbench_zh, "SQuAD-zh": load_squad_zh,
           "spoken-squad": load_spoken_squad, "minds14-zh": load_minds14_zh}


def run_ds(name, loader):
    rng = np.random.default_rng(SLICE_SEED)
    items = loader(rng)
    t0 = time.time(); greedy, oracle, skip = [], [], 0
    for k, it in enumerate(items):
        try:
            g = reward(it, gen(it["wav"], it["instr"], 42, 0.0))
            samp = [reward(it, gen(it["wav"], it["instr"], 100 + s, TEMP)) for s in range(NSAMP)]
        except Exception as e:
            skip += 1; continue
        greedy.append(g); oracle.append(int(g or any(samp)))
        if (k + 1) % 50 == 0:
            print(f"    [{name} {k+1}/{len(items)}] greedy={np.mean(greedy):.3f} "
                  f"oracle={np.mean(oracle):.3f} ({time.time()-t0:.0f}s)", flush=True)
    n = len(greedy)
    gacc, oacc = float(np.mean(greedy)), float(np.mean(oracle))
    res = {"n": n, "n_skip": skip, "greedy": round(gacc, 4), "oracle": round(oacc, 4),
           "oracle_delta": round(oacc - gacc, 4), "saturated": bool(gacc >= 0.90),
           "task": items[0]["task"] if items else "?"}
    print(f"  == {name}: greedy={gacc:.3f} oracle={oacc:.3f} delta={oacc-gacc:+.3f} "
          f"n={n} {'[SATURATED]' if res['saturated'] else ''}", flush=True)
    return res


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print(f"server={BASE} n={N_UTTS} N={NSAMP} temp={TEMP}", flush=True)
    results = {}
    if OUT.exists():
        results = json.load(open(OUT)).get("results", {})
    for name, loader in LOADERS.items():
        if only and name not in only:
            continue
        try:
            results[name] = run_ds(name, loader)
        except Exception as e:
            print(f"  !! {name} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": {"stage": "1-directional", "n_per_set": N_UTTS, "nsamp": NSAMP,
                               "slice_seed": SLICE_SEED, "model": "qwen3-omni-30b Q8_0 GGUF llama.cpp"},
                   "results": results,
                   "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/p2_baselines.py"},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n=== P2 BASELINES ===\n" + json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print("wrote", OUT, "\nP2_DONE", flush=True)


if __name__ == "__main__":
    main()

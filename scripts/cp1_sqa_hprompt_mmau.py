"""CP-1 SQA arm — quantify H_prompt − H_fix for audio-MCQ on frozen Qwen3-Omni-30B (GPU).

MINI-PREREG (committed before generation; Stage-1 directional worth-investment check on CP-1, SQA arm):
  Question: on a HARDER, less-saturated surface than intent (MMAU audio reasoning, where models score
  well below ceiling), how much oracle headroom does INSTRUCTION diversity add beyond SAMPLING at
  matched budget — i.e. H_prompt − H_fix on MCQ accuracy? Chosen because E1 showed MInDS intent is
  near-saturated (greedy 0.95), giving no method room; MMAU gives the prompt-space a fair chance.

  Dataset/slice: mmau-mini test_mini (1,000 items over sound/music/speech), n = M3_N (default 150),
  deterministic seed-frozen slice. Audio = the clip; MCQ with per-question `choices` + `answer` (full
  choice text). Arms matched at BUDGET=8 generations/utt:
    - greedy   : 1 fixed instruction, temp 0
    - FIX      : 1 fixed instruction x 8 temp samples -> oracle-over-8 = H_fix
    - PROMPT   : 8 genuine task-definition instructions x 1 each -> oracle-over-8
    - GENERIC (b1 floor): 8 minimal 'here are the choices, pick one' instructions x 1 each -> if
                 PROMPT ~ GENERIC oracle, elaborate task-definition adds nothing (b1, not b2).
  Readouts (per-utt MCQ-correct 0/1; parse the output to one of the question's choices):
    greedy/majority acc; oracle_fix/prompt/generic; H_fix, H_prompt, **H_prompt - H_fix**;
    b2-share = oracle_prompt - oracle_generic; paired cluster-bootstrap 95% CIs (descriptive);
    per-difficulty and per-task (sound/music/speech) breakdown.
  Grade: [directional | n=150 | single-touch | not significance-bearing]. Stage-2 (only if this
  directional check warrants investment) = full 1,000 + label-sensitivity + choice-order control.
  Budget: 150 x (1 + 8 + 8 + 8) = 3,750 short-output generations, ~45-75 min GPU.

Requires the resident llama-server.
reproduce:
  SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/cp1_sqa_hprompt_mmau.py
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
SLICE_SEED = 20260704
OUT = DATA / "_repro" / "cp1_sqa_hprompt_mmau.json"
WAV_DIR = DATA / "_repro" / "cp1_mmau_wavs"
NBOOT = 10000
LETTERS = ["A", "B", "C", "D", "E", "F"]


def fmt_choices(choices):
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def build_instructions(choices):
    cl = fmt_choices(choices)
    fix = (f"Listen to the audio and answer the multiple-choice question. Choose the single best option.\n"
           f"Options:\n{cl}\nAnswer with only the option letter and text, e.g. 'A. ...'.")
    prompt = [
        fix,
        (f"You are an expert audio analyst. Based only on what you hear, select the correct answer.\n"
         f"{cl}\nRespond with the best option."),
        (f"Carefully reason about the audio, then pick the one correct choice:\n{cl}\n"
         f"Give the option letter and its text."),
        (f"Audio question answering. Attend to the acoustic content and choose exactly one option.\n"
         f"{cl}\nOutput the chosen option."),
        (f"Think step by step about the sound, then decide which option is correct:\n{cl}\n"
         f"State only the final option."),
        (f"From the audio evidence, which option is right?\n{cl}\nAnswer with one option."),
        (f"Determine the answer from the recording and select the matching choice:\n{cl}\n"
         f"Return the option letter and text."),
        (f"Analyze the audio and identify the correct multiple-choice answer:\n{cl}\n"
         f"Provide only your chosen option."),
    ]
    # b1 floor: minimal / generic framing (no task-specific reasoning cues), choices still shown
    generic = [
        (f"Pick one:\n{cl}\nAnswer with an option."),
        (f"Choose an option:\n{cl}\nOutput one."),
        (f"Select one of the following:\n{cl}\nReply with a choice."),
        (f"Here are the options:\n{cl}\nGive one."),
        (f"Options below. Return one:\n{cl}"),
        (f"One of these:\n{cl}\nAnswer."),
        (f"Which one?\n{cl}"),
        (f"Answer with an option:\n{cl}"),
    ]
    return fix, prompt, generic


def parse_choice(text, choices):
    t = text.strip().lower()
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()
    ch_norm = [norm(c) for c in choices]
    # 1) leading letter "A." / "A)" / "A:"
    for i in range(len(choices)):
        L = LETTERS[i].lower()
        if t[:2] in (L + ".", L + ")", L + ":", L + " ") or t.strip() == L:
            return i
    # 2) exact / substring choice-text match (longest choice first to avoid prefix collisions)
    for i in sorted(range(len(choices)), key=lambda k: -len(ch_norm[k])):
        if ch_norm[i] and ch_norm[i] in norm(text):
            return i
    # 3) any choice word overlap
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
    cols = ["question", "choices", "answer", "task", "difficulty", "audio"]
    rows = []
    for f in fs:
        rows.extend(pq.read_table(f, columns=cols).to_pylist())
    # dedupe by (question, task) — the v05 re-release duplicates the 1,000 items
    seen, uniq = set(), []
    for r in rows:
        k = (r["question"], r["task"], tuple(r["choices"]))
        if k not in seen:
            seen.add(k); uniq.append(r)
    idx = np.random.default_rng(SLICE_SEED).permutation(len(uniq))[:N_UTTS]
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for n, j in enumerate(idx):
        r = uniq[int(j)]
        wp = WAV_DIR / f"mmau_{int(j)}.wav"
        if not wp.exists():
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            sf.write(str(wp), wav, sr)
        gt = [i for i, c in enumerate(r["choices"]) if c == r["answer"]]
        out.append({"wav": str(wp), "choices": list(r["choices"]),
                    "gt_idx": gt[0] if gt else -1, "answer": r["answer"],
                    "task": r["task"], "difficulty": r.get("difficulty")})
    return [u for u in out if u["gt_idx"] >= 0]  # drop any answer-not-in-choices


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
    keys = sorted(vals)
    arr = [vals[k] for k in keys]
    rng = np.random.default_rng(seed)
    ms = np.array([float(np.mean([arr[i] for i in rng.integers(0, len(arr), len(arr))])) for _ in range(NBOOT)])
    return [round(float(np.quantile(ms, 0.025)), 4), round(float(np.quantile(ms, 0.975)), 4)]


def main():
    t0 = time.time()
    utts = load_slice()
    print(f"server={BASE} n={len(utts)} budget={BUDGET} temp={TEMP} (fix 1x8 / prompt 8x1 / generic 8x1)", flush=True)
    rows, skipped = [], []
    for k, u in enumerate(utts):
        ch, gt = u["choices"], u["gt_idx"]
        fixI, prI, genI = build_instructions(ch)
        try:
            g = parse_choice(gen(u["wav"], fixI, True, 42), ch)
            fix = [parse_choice(gen(u["wav"], fixI, False, 100 + i), ch) for i in range(BUDGET)]
            pr = [parse_choice(gen(u["wav"], prI[i], False, 200 + i), ch) for i in range(BUDGET)]
            gn = [parse_choice(gen(u["wav"], genI[i], False, 300 + i), ch) for i in range(BUDGET)]
        except Exception as e:
            print(f"  [{k+1}] SKIP ({type(e).__name__}: {str(e)[:80]})", flush=True); skipped.append(k); continue
        maj = collections.Counter(fix).most_common(1)[0][0]
        rows.append({"task": u["task"], "difficulty": u["difficulty"], "gt": gt,
                     "greedy_correct": int(g == gt), "majority_correct": int(maj == gt),
                     "oracle_fix": int(gt in fix), "oracle_prompt": int(gt in pr),
                     "oracle_generic": int(gt in gn)})
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(utts)}] greedy={np.mean([r['greedy_correct'] for r in rows]):.3f} "
                  f"o_fix={np.mean([r['oracle_fix'] for r in rows]):.3f} "
                  f"o_prompt={np.mean([r['oracle_prompt'] for r in rows]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    m = lambda key: round(float(np.mean([r[key] for r in rows])), 4)
    idxd = lambda f: {i: f(r) for i, r in enumerate(rows)}
    greedy, ofix, opr, ogn = m("greedy_correct"), m("oracle_fix"), m("oracle_prompt"), m("oracle_generic")
    bytask = {tk: round(float(np.mean([r["greedy_correct"] for r in rows if r["task"] == tk])), 3)
              for tk in sorted(set(r["task"] for r in rows))}
    summary = {
        "problem": "CP-1 SQA arm — H_prompt - H_fix on MMAU-mini audio-MCQ (frozen Qwen3-Omni-30B)",
        "stage": "1-directional", "grade": "[directional | n=%d | single-touch | not significance-bearing]" % len(rows),
        "n_utts": len(rows), "n_skipped": len(skipped), "budget_per_arm": BUDGET, "temp": TEMP,
        "dataset": "mmau-mini test_mini (sound/music/speech MCQ)", "slice_seed": SLICE_SEED,
        "acc": {"greedy": greedy, "majority": m("majority_correct"), "oracle_fix": ofix,
                "oracle_prompt": opr, "oracle_generic_b1floor": ogn},
        "greedy_by_task": bytask,
        "H_fix": round(ofix - greedy, 4), "H_prompt": round(opr - greedy, 4),
        "H_prompt_minus_H_fix": round(opr - ofix, 4),
        "b2_share_prompt_minus_generic": round(opr - ogn, 4),
        "ci_prompt_minus_fix_descriptive": cboot(idxd(lambda r: r["oracle_prompt"] - r["oracle_fix"])),
        "ci_prompt_minus_generic_descriptive": cboot(idxd(lambda r: r["oracle_prompt"] - r["oracle_generic"])),
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "prereg": "module docstring, committed before generation (git history is the anchor)",
    }
    ci, cib2 = summary["ci_prompt_minus_fix_descriptive"], summary["ci_prompt_minus_generic_descriptive"]
    summary["verdict"] = (
        f"greedy MCQ-acc {greedy:.3f} (by task {bytask}); oracle-over-8 fix {ofix:.3f} vs prompt {opr:.3f} -> "
        f"H_prompt-H_fix = {opr-ofix:+.4f} (CI {ci}); b1 generic-floor oracle {ogn:.3f}, b2-share "
        f"prompt-minus-generic {opr-ogn:+.4f} (CI {cib2}). H_fix={ofix-greedy:+.4f} (headroom present: "
        f"{'yes' if ofix-greedy > 0.03 else 'small'}). Directional: instruction diversity "
        f"{'adds' if ci[0] > 0 else 'does not clearly add'} oracle headroom on this harder surface; "
        f"{'gain survives the b1 floor (task-definition helps, b2)' if cib2[0] > 0 else 'does not clearly survive the b1 floor'}.")
    out = {"summary": summary, "per_utt": rows, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data python scripts/cp1_sqa_hprompt_mmau.py (llama-server resident)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("CP1_SQA_DONE", flush=True)


if __name__ == "__main__":
    main()

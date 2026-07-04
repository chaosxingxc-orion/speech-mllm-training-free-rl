"""CP-1 SLU arm — quantify H_prompt − H_fix for intent classification on frozen Qwen3-Omni-30B (GPU).

MINI-PREREG (committed before generation; Stage-1 directional worth-investment check on K2-selected problem CP-1, SLU arm):
  Question: on the schema-rich intent-classification surface, how much oracle headroom does INSTRUCTION
  diversity add beyond SAMPLING diversity at matched budget — i.e. H_prompt − H_fix on MInDS-14 intent?
  Directly measures the survey's central quantity on the family the Stage-1 probe weakly favored and
  where the in-house +0.126 [scoped] precedent lives.

  Dataset/slice: MInDS-14 en-US (563 utts, 14 balanced banking intents); n = M3_N env (default 150),
  deterministic seed-frozen slice. Audio = the utterance; the model classifies intent from speech.
  Arms, matched at BUDGET=8 generations/utt:
    - greedy         : 1 fixed instruction, temp 0 (the deployed default)
    - FIX            : 1 fixed instruction x 8 temp samples  -> oracle-over-8 = H_fix support
    - PROMPT         : 8 genuine task-definition instructions x 1 sample each -> oracle-over-8
    - RANDOM (b1)    : 8 instructions with the 14 category NAMES replaced by nonsense tokens x 1 each
                       -> b1 floor: if PROMPT ~ RANDOM oracle, the gain is instruction-diversity/format,
                       not genuine task-definition (b2).
  Readouts (per-utt intent-correct 0/1, parsed to one of 14 labels):
    - greedy_acc, majority_acc (deployable); oracle_fix_acc, oracle_prompt_acc, oracle_random_acc;
    - H_fix = oracle_fix − greedy ; H_prompt = oracle_prompt − greedy ; **H_prompt − H_fix** (the target);
    - Delta_BM (matched-budget) = oracle_prompt − oracle_fix ; b2-share = oracle_prompt − oracle_random ;
    - paired cluster-bootstrap 95% CIs (descriptive) on per-utt (correct_prompt − correct_fix) etc.
  Grade: [directional | n=150 | single-touch | not significance-bearing]. Stage-2 (only if this directional check warrants investment) = full
  563 + multilingual + label-sensitivity control on a fresh touch.
  Budget: 150 x (1 + 8 + 8 + 8) = 3,750 short-output generations, ~45-75 min GPU.

Requires the resident llama-server (see repro_asr_best_of_n_llamacpp.py header).
reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp1_slu_hprompt_minds14.py
"""
import base64, collections, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
N_UTTS = int(os.environ.get("M3_N", "150"))
BUDGET = 8
TEMP = float(os.environ.get("CP1_TEMP", "0.7"))
MAXTOK = 32
SLICE_SEED = 20260704
OUT = DATA / "_repro" / "cp1_slu_hprompt_minds14.json"
WAV_DIR = DATA / "_repro" / "cp1_minds14_wavs"
NBOOT = 10000

# MInDS-14 intent id -> (canonical name, gloss) — the choice surface
INTENTS = {
    0: ("abroad", "using cards or banking abroad / travel notification"),
    1: ("address", "changing or updating the address on the account"),
    2: ("app_error", "a bug or error in the mobile/online banking app"),
    3: ("atm_limit", "the cash withdrawal limit at an ATM"),
    4: ("balance", "checking the account balance"),
    5: ("business_loan", "applying for or asking about a business loan"),
    6: ("card_issues", "a lost, stolen, blocked or broken card"),
    7: ("cash_deposit", "depositing cash into the account"),
    8: ("direct_debit", "setting up or cancelling a direct debit / standing order"),
    9: ("freeze", "freezing or blocking the account or a card"),
    10: ("high_value_payment", "making a large / high-value payment or transfer"),
    11: ("joint_account", "opening or managing a joint account"),
    12: ("latest_transactions", "recent transactions or transaction history"),
    13: ("pay_bill", "paying a bill"),
}
NAMES = [INTENTS[i][0] for i in range(14)]
CATLIST = "\n".join(f"- {INTENTS[i][0]}: {INTENTS[i][1]}" for i in range(14))

# 8 genuine task-definition instructions (varied phrasings; all list the 14 categories)
FIX_INSTR = (f"You are a banking-assistant intent classifier. Listen to the customer's spoken request "
             f"and classify it into exactly ONE of these intent categories:\n{CATLIST}\n"
             f"Answer with only the category name, nothing else.")
PROMPT_INSTR = [
    FIX_INSTR,
    (f"Task: spoken-language understanding for a bank call centre. The audio is a customer utterance. "
     f"Decide which single intent it expresses:\n{CATLIST}\nReply with just the category name."),
    (f"Classify the intent of this banking voice message into one label from the list. "
     f"Categories:\n{CATLIST}\nOutput format: the category name only."),
    (f"The recording is a customer speaking to their bank. Identify the ONE intent that best matches.\n"
     f"{CATLIST}\nGive only the intent name."),
    (f"Intent detection. Map the spoken request to one of the 14 banking intents below.\n{CATLIST}\n"
     f"Respond with a single category name."),
    (f"Listen carefully. What does the caller want? Choose exactly one:\n{CATLIST}\n"
     f"Answer with the category name and nothing else."),
    (f"You label banking audio by intent. Pick the single best-matching category:\n{CATLIST}\n"
     f"Return only the label."),
    (f"Determine the customer's goal from the audio and select one intent category:\n{CATLIST}\n"
     f"Output: category name only."),
]
# 8 b1-floor instructions: same structure, category NAMES replaced by nonsense tokens (glosses kept
# so the task is still 'classify' but the labels carry no task meaning -> tests format-vs-task-def)
_NONSENSE = ["zorp", "quen", "flim", "brax", "vunt", "kite", "morl", "dweb", "plax", "grint",
             "yolt", "snee", "crov", "wumf"]
_NCATLIST = "\n".join(f"- {_NONSENSE[i]}: {INTENTS[i][1]}" for i in range(14))
RANDOM_INSTR = [ins.replace(CATLIST, _NCATLIST) for ins in PROMPT_INSTR]
# map a nonsense label back to the intent id for scoring the RANDOM arm
_NONSENSE_TO_ID = {_NONSENSE[i]: i for i in range(14)}


def parse_label(text, nonsense=False):
    t = text.strip().lower()
    vocab = _NONSENSE if nonsense else NAMES
    # exact / substring match against the (possibly nonsense) label vocabulary
    for i, name in enumerate(vocab):
        if name in t:
            return _NONSENSE_TO_ID[name] if nonsense else i
    # fallback: match against canonical names' words for the genuine arms
    if not nonsense:
        toks = set(t.replace("_", " ").split())
        best, bi = 0, -1
        for i in range(14):
            words = set(NAMES[i].replace("_", " ").split())
            ov = len(toks & words)
            if ov > best:
                best, bi = ov, i
        return bi
    return -1


def load_slice():
    import pyarrow.parquet as pq
    import soundfile as sf
    f = DATA / "datasets/minds14/en-US/train-00000-of-00001.parquet"
    t = pq.read_table(str(f), columns=["audio", "intent_class", "english_transcription"])
    audio = t.column("audio").to_pylist()
    intent = t.column("intent_class").to_pylist()
    txt = t.column("english_transcription").to_pylist()
    idx = np.random.default_rng(SLICE_SEED).permutation(len(intent))[:N_UTTS]
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    utts = []
    for j in idx:
        j = int(j)
        wp = WAV_DIR / f"minds_{j}.wav"
        if not wp.exists():
            wav, sr = sf.read(io.BytesIO(audio[j]["bytes"]), dtype="float32")
            sf.write(str(wp), wav, sr)
        utts.append({"idx": j, "wav": str(wp), "intent": int(intent[j]), "text": txt[j]})
    return utts


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


def cboot(vals_by_utt, seed=42):
    utts = sorted(vals_by_utt)
    arr = [vals_by_utt[u] for u in utts]
    rng = np.random.default_rng(seed)
    ms = np.array([float(np.mean([arr[i] for i in rng.integers(0, len(arr), len(arr))])) for _ in range(NBOOT)])
    return [round(float(np.quantile(ms, 0.025)), 4), round(float(np.quantile(ms, 0.975)), 4)]


def main():
    t0 = time.time()
    utts = load_slice()
    print(f"server={BASE} n={len(utts)} budget={BUDGET} temp={TEMP} (fix 1x8 / prompt 8x1 / random 8x1)", flush=True)
    rows, skipped = [], []
    for k, u in enumerate(utts):
        gt = u["intent"]
        try:
            g = parse_label(gen(u["wav"], FIX_INSTR, True, 42))
            fix = [parse_label(gen(u["wav"], FIX_INSTR, False, 100 + i)) for i in range(BUDGET)]
            pr = [parse_label(gen(u["wav"], PROMPT_INSTR[i], False, 200 + i)) for i in range(BUDGET)]
            rnd = [parse_label(gen(u["wav"], RANDOM_INSTR[i], False, 300 + i), nonsense=True) for i in range(BUDGET)]
        except Exception as e:
            print(f"  [{k+1}] idx{u['idx']} SKIP ({type(e).__name__}: {str(e)[:80]})", flush=True)
            skipped.append(u["idx"]); continue
        maj = collections.Counter(fix).most_common(1)[0][0]
        rows.append({"idx": u["idx"], "gt": gt, "greedy_pred": g,
                     "greedy_correct": int(g == gt), "majority_correct": int(maj == gt),
                     "oracle_fix": int(gt in fix), "oracle_prompt": int(gt in pr),
                     "oracle_random": int(gt in rnd),
                     "fix": fix, "prompt": pr, "random": rnd,
                     "best_instr": next((i for i in range(BUDGET) if pr[i] == gt), -1)})
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(utts)}] greedy_acc={np.mean([r['greedy_correct'] for r in rows]):.3f} "
                  f"oracle_fix={np.mean([r['oracle_fix'] for r in rows]):.3f} "
                  f"oracle_prompt={np.mean([r['oracle_prompt'] for r in rows]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    m = lambda k: round(float(np.mean([r[k] for r in rows])), 4)
    keys = {r["idx"]: r for r in rows}
    by = lambda f: {i: f(r) for i, r in keys.items()}
    greedy, ofix, opr, ornd = m("greedy_correct"), m("oracle_fix"), m("oracle_prompt"), m("oracle_random")
    summary = {
        "problem": "CP-1 SLU arm — H_prompt - H_fix on MInDS-14 en-US intent (frozen Qwen3-Omni-30B)",
        "stage": "1-directional", "grade": "[directional | n=%d | single-touch | not significance-bearing]" % len(rows),
        "n_utts": len(rows), "n_skipped": len(skipped), "budget_per_arm": BUDGET, "temp": TEMP,
        "dataset": "minds14/en-US (14 balanced banking intents)", "slice_seed": SLICE_SEED,
        "acc": {"greedy": greedy, "majority": m("majority_correct"),
                "oracle_fix": ofix, "oracle_prompt": opr, "oracle_random_b1floor": ornd},
        "H_fix": round(ofix - greedy, 4), "H_prompt": round(opr - greedy, 4),
        "H_prompt_minus_H_fix": round(opr - ofix, 4),
        "Delta_BM_matched_budget": round(opr - ofix, 4),
        "b2_share_prompt_minus_random": round(opr - ornd, 4),
        "ci_prompt_minus_fix_descriptive": cboot(by(lambda r: r["oracle_prompt"] - r["oracle_fix"])),
        "ci_prompt_minus_random_descriptive": cboot(by(lambda r: r["oracle_prompt"] - r["oracle_random"])),
        "best_instruction_histogram": {str(i): sum(1 for r in rows if r["best_instr"] == i) for i in range(BUDGET)},
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "prereg": "module docstring, committed before generation (git history is the anchor)",
    }
    ci = summary["ci_prompt_minus_fix_descriptive"]; cib2 = summary["ci_prompt_minus_random_descriptive"]
    summary["verdict"] = (
        f"greedy intent-acc {greedy:.3f}; oracle-over-8 fix {ofix:.3f} vs prompt {opr:.3f} -> "
        f"H_prompt-H_fix = {opr-ofix:+.4f} (descriptive CI {ci}); b1-floor (random-label) oracle {ornd:.3f}, "
        f"so b2-share prompt-minus-random {opr-ornd:+.4f} (CI {cib2}). Directional read: instruction "
        f"diversity {'adds' if ci[0] > 0 else 'does not clearly add'} oracle headroom beyond sampling on "
        f"intent; {'and the gain survives the b1 format-floor' if cib2[0] > 0 else 'and it does NOT clearly survive the b1 format-floor (may be format/diversity, not task-definition)'}.")
    out = {"summary": summary, "per_utt": rows, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp1_slu_hprompt_minds14.py (llama-server resident)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("CP1_SLU_DONE", flush=True)


if __name__ == "__main__":
    main()

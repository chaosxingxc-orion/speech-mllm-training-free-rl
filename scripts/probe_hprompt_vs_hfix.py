"""STAGE-1 DIRECTIONAL PROBE — H_prompt vs H_fix at matched budget (mini-prereg, frozen on commit).

MINI-PREREG (committed before any generation; owner pre-authorized at K1, 2026-07-04):
  Directional question: on a frozen omni speech model, does INSTRUCTION DIVERSITY buy oracle
  headroom beyond what the same SAMPLE BUDGET buys under one fixed instruction?
  Design: n=50 fresh LibriSpeech test-other utterances (seed 20260704; excludes all previously
  used ids: 144 C1 dev + 36 M3 + 144 M5-confirmatory), snr5, single-touch. Two budget-matched
  arms of 32 samples each: ARM-FIX = 1 instruction (the C1 fixed prompt) x 32 temp-0.8 samples;
  ARM-PROMPT = 8 task-definition instructions x 4 samples. Readouts: oracle WER per arm
  (headroom vs the shared greedy reference), MBR WER per arm (deployable readout), and the
  per-instruction oracle contribution. Deltas reported with cluster-bootstrap 95% CIs marked
  DESCRIPTIVE. Statistical standing: [directional-only | n=50 | single-touch |
  not significance-bearing] — this probe feeds the problem-definition doc's viability column
  ONLY; per the Stage-1 methodology it cannot rank problems, kill hypotheses, or count as
  evidence of significance. Budget: 50 x (1 + 32 + 32) = 3,250 generations, ~2-2.5 h GPU.

Requires the resident llama-server (see repro_asr_best_of_n_llamacpp.py header).
reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/probe_hprompt_vs_hfix.py
"""
import base64, collections, io, json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/d/chao_workspace/exploring-l4-intelligence/common/src")
from speechrl_common.rl.reward import wer
from m5_selector_rescore_dev import mbr_utils, zscore

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
REPRO = Path(__file__).resolve().parent.parent / "_repro"
OUT = DATA / "_repro" / "probe_hprompt_vs_hfix.json"
WAV_DIR = DATA / "_repro" / "probe_hpf_wavs_snr5"
SLICE_SEED = 20260704
N_UTTS = 50
BUDGET = 32
TEMP, TOP_P, MAXTOK, SNR = 0.8, 0.95, 100, 5.0
NBOOT = 10000

# K=8 task-definition instructions. #0 is the C1 fixed prompt (the H_fix arm's single instruction).
INSTRUCTIONS = [
    "Transcribe the English speech into text. Output only the transcript, no explanation.",
    "You are a professional transcriptionist. Write down exactly the words you hear in the audio, preserving every word. Output the transcript only.",
    "Listen to the audio carefully and transcribe it verbatim. Do not paraphrase and do not correct grammar; transcribe exactly what is spoken. Transcript only.",
    "Task: automatic speech recognition. Input: one English utterance read aloud from a book. Output: the exact transcript in plain text, nothing else.",
    "Transcribe the noisy speech recording. Focus on the voice and ignore the background noise. Output only the words spoken.",
    "This is an audiobook excerpt, possibly older literary English with unusual proper names. Transcribe it word-for-word. Output the transcript only.",
    "Write the exact transcription of the utterance. If a word is unclear, choose the most plausible English word given the sentence context. Transcript only.",
    "ASR task. Provide a verbatim transcript of the audio as plain text without any commentary. Output only the transcript.",
]
K = len(INSTRUCTIONS)
PER_INSTR = BUDGET // K  # 4


def pick_slice():
    import pyarrow.parquet as pq
    t = pq.read_table(DATA / "datasets/librispeech/all/test.other/0000.parquet", columns=["text", "id"])
    ids = t.column("id").to_pylist()
    ref = dict(zip(ids, t.column("text").to_pylist()))
    used = set()
    for art in ("asr_bon_llamacpp_snr5.json", "m5_selector_confirmatory.json"):
        for r in json.load(open(REPRO / art)).get("per_utt", json.load(open(REPRO / art)).get("per_item", [])):
            used.add(r["id"])
    used |= set(r["id"] for r in json.load(open(REPRO / "m3_phase0_selection.json"))["selected"])
    elig = [i for i in ids if i not in used]
    rng = np.random.default_rng(SLICE_SEED)
    sel = sorted(rng.choice(elig, N_UTTS, replace=False).tolist())
    return sel, ref


def prepare_wavs(sel):
    import pyarrow.parquet as pq
    import soundfile as sf
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    need = [u for u in sel if not (WAV_DIR / f"{u}.wav").exists()]
    if need:
        t = pq.read_table(DATA / "datasets/librispeech/all/test.other/0000.parquet", columns=["audio", "id"])
        idx = {u: i for i, u in enumerate(t.column("id").to_pylist())}
        audio = t.column("audio").to_pylist()
        for j, u in enumerate(sel):
            if u not in need:
                continue
            wav, sr = sf.read(io.BytesIO(audio[idx[u]]["bytes"]), dtype="float32")
            rng = np.random.default_rng(SLICE_SEED * 100003 + j)
            sig_p = float(np.mean(wav ** 2)) + 1e-12
            noise = rng.standard_normal(len(wav)).astype("float32")
            noise *= np.sqrt(sig_p / (10 ** (SNR / 10.0)) / (float(np.mean(noise ** 2)) + 1e-12))
            sf.write(str(WAV_DIR / f"{u}.wav"), wav + noise, sr)


def gen(clip, instruction, greedy, seed):
    b64 = base64.b64encode(open(clip, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instruction}]}],
        "max_tokens": MAXTOK, "seed": seed, "temperature": 0.0 if greedy else TEMP}
    if not greedy:
        payload["top_p"] = TOP_P
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def cboot(vals_by_utt, seed=42):
    utts = sorted(vals_by_utt)
    arrs = [np.array(vals_by_utt[u]) for u in utts]
    rng = np.random.default_rng(seed)
    means = np.empty(NBOOT)
    n = len(arrs)
    for b in range(NBOOT):
        idx = rng.integers(0, n, n)
        means[b] = float(np.concatenate([arrs[i] for i in idx]).mean())
    return [round(float(np.quantile(means, 0.025)), 5), round(float(np.quantile(means, 0.975)), 5)]


def main():
    t0 = time.time()
    sel, ref = pick_slice()
    prepare_wavs(sel)
    print(f"server={BASE} n={len(sel)} budget={BUDGET}/arm K={K}x{PER_INSTR} snr={SNR}", flush=True)
    rows, skipped = [], []
    for j, u in enumerate(sel):
        clip = str(WAV_DIR / f"{u}.wav")
        try:
            greedy = gen(clip, INSTRUCTIONS[0], True, 42)
            fix_pool = [gen(clip, INSTRUCTIONS[0], False, 5000 + i) for i in range(BUDGET)]
            pr_pool, pr_tags = [], []
            for ki, ins in enumerate(INSTRUCTIONS):
                for i in range(PER_INSTR):
                    pr_pool.append(gen(clip, ins, False, 1000 * (ki + 1) + i))
                    pr_tags.append(ki)
        except Exception as e:
            print(f"  [{j+1}] {u} SKIP ({type(e).__name__}: {str(e)[:90]})", flush=True)
            skipped.append(u); continue
        gw = wer(ref[u], greedy)
        fix_wers = [wer(ref[u], c) for c in fix_pool]
        pr_wers = [wer(ref[u], c) for c in pr_pool]
        mbr_fix = fix_pool[int(np.argmax(zscore(mbr_utils(fix_pool))))]
        mbr_pr = pr_pool[int(np.argmax(zscore(mbr_utils(pr_pool))))]
        best_k = pr_tags[int(np.argmin(pr_wers))]
        rows.append({"id": u, "ref": ref[u], "greedy": greedy, "greedy_wer": gw,
                     "oracle_fix": min(fix_wers), "oracle_prompt": min(pr_wers),
                     "mbr_fix": wer(ref[u], mbr_fix), "mbr_prompt": wer(ref[u], mbr_pr),
                     "best_instruction_idx": best_k,
                     "per_instr_oracle": {str(ki): min(w for w, t in zip(pr_wers, pr_tags) if t == ki)
                                          for ki in range(K)},
                     "fix_pool": fix_pool, "prompt_pool": pr_pool, "prompt_tags": pr_tags})
        print(f"  [{j+1}/{len(sel)}] {u} greedy={gw:.3f} oracle_fix={min(fix_wers):.3f} "
              f"oracle_prompt={min(pr_wers):.3f} bestK={best_k} ({time.time()-t0:.0f}s)", flush=True)

    m = lambda k: round(float(np.mean([r[k] for r in rows])), 5)
    d_oracle = {r["id"]: [r["oracle_fix"] - r["oracle_prompt"]] for r in rows}
    d_mbr = {r["id"]: [r["mbr_fix"] - r["mbr_prompt"]] for r in rows}
    best_hist = collections.Counter(r["best_instruction_idx"] for r in rows)
    summary = {
        "stage": "1-directional",
        "label": "[directional-only | n=%d | single-touch | not significance-bearing]" % len(rows),
        "question": "does instruction diversity buy oracle headroom beyond sample count at matched budget",
        "design": {"n_utts": len(rows), "n_skipped": len(skipped), "budget_per_arm": BUDGET,
                   "K": K, "per_instr": PER_INSTR, "snr_db": SNR, "temp": TEMP,
                   "slice_seed": SLICE_SEED, "excluded_prior_ids": "144 C1 + 36 M3 + 144 M5-confirmatory",
                   "instructions": INSTRUCTIONS},
        "arms": {"greedy_wer": m("greedy_wer"),
                 "oracle_fix": m("oracle_fix"), "oracle_prompt": m("oracle_prompt"),
                 "mbr_fix": m("mbr_fix"), "mbr_prompt": m("mbr_prompt")},
        "H_fix": round(m("greedy_wer") - m("oracle_fix"), 5),
        "H_prompt": round(m("greedy_wer") - m("oracle_prompt"), 5),
        "delta_oracle_fix_minus_prompt": {"mean": round(float(np.mean([v[0] for v in d_oracle.values()])), 5),
                                           "ci95_descriptive": cboot(d_oracle)},
        "delta_mbr_fix_minus_prompt": {"mean": round(float(np.mean([v[0] for v in d_mbr.values()])), 5),
                                        "ci95_descriptive": cboot(d_mbr)},
        "best_instruction_histogram": {str(k): best_hist.get(k, 0) for k in range(K)},
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "prereg": "module docstring, committed before generation (git history is the anchor)",
    }
    out = {"summary": summary, "per_utt": rows, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/probe_hprompt_vs_hfix.py "
                        "(llama-server resident; slice deterministic from seed 20260704 + committed exclusions)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("PROBE_HPF_DONE", flush=True)


if __name__ == "__main__":
    main()

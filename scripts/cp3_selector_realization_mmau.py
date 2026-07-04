"""CP-3 (c) realization — can a MODERN label-free selector harvest the E3 oracle headroom? (GPU).

MINI-PREREG (committed before generation; Stage-1 directional; the pivotal (c) leg for Q1):
  E3 on MMAU showed oracle-over-8 = 0.773 vs greedy = majority = 0.640 -> a real +0.133 headroom that
  the deployable majority selector realizes 0% of (rho ~ 0). Question: does any MODERN label-free
  selector — self-certainty (token confidence), confidence-weighted vote, or an LLM-judge rerank —
  harvest a nontrivial fraction of that headroom? All selectors are label-free (zero leakage): they see
  only the audio, the sampled answers, and the model's own logprobs — never the reference.

  Slice: mmau-mini test_mini, SAME seed-frozen n=150 as E3 (paired). Fixed text instruction. Per item:
  8 temp samples WITH logprobs -> per-sample chosen option + confidence (mean generated-token logprob);
  1 greedy; 1 judge pass (re-ask, showing the sampled options + vote counts, pick one). Selectors:
    - greedy / majority (baselines) ; self_certainty (option of the max-confidence sample) ;
      conf_weighted_vote (sum confidence per option, argmax) ; judge (2nd-pass re-pick) ; oracle (ceiling).
  Readouts: accuracy per selector; **rho = (sel_acc - greedy) / (oracle - greedy)** = fraction of the
  headroom harvested; paired cluster-bootstrap 95% CIs (descriptive) on per-item (sel - majority).
  Grade: [directional | n=150 | single-touch | not significance-bearing].
  Budget: 150 x (1 greedy + 8 sample + 1 judge) = 1,500 short-output generations, ~30-45 min GPU.

Requires the resident llama-server.
reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp3_selector_realization_mmau.py
"""
import base64, collections, glob, io, json, math, os, sys, time, urllib.request
from pathlib import Path
import numpy as np

BASE = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8091")
DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
N_UTTS = int(os.environ.get("M3_N", "150"))
BUDGET = 8
TEMP = float(os.environ.get("CP1_TEMP", "0.7"))
MAXTOK = 48
SLICE_SEED = 20260704
OUT = DATA / "_repro" / "cp3_selector_realization_mmau.json"
BASE_WAV = DATA / "_repro" / "cp1_mmau_wavs"   # E3 originals
NBOOT = 10000
LETTERS = ["A", "B", "C", "D", "E", "F"]


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
            out.append({"wav": str(wp), "choices": list(r["choices"]), "gt": gt[0],
                        "question": r["question"], "task": r["task"]})
    return out


def gen(clip, instruction, greedy, seed, want_logprobs=False):
    b64 = base64.b64encode(open(clip, "rb").read()).decode()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": instruction}]}],
        "max_tokens": MAXTOK, "seed": seed, "temperature": 0.0 if greedy else TEMP}
    if not greedy:
        payload["top_p"] = 0.95
    if want_logprobs:
        payload["logprobs"] = True; payload["top_logprobs"] = 1
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        ch = json.loads(r.read().decode())["choices"][0]
    txt = ch["message"]["content"].strip()
    conf = 0.0
    lp = ch.get("logprobs")
    if lp and lp.get("content"):
        toks = [c["logprob"] for c in lp["content"] if c.get("logprob") is not None]
        conf = float(np.mean(toks)) if toks else 0.0   # mean generated-token logprob (higher = more confident)
    return txt, conf


def cboot(vals, seed=42):
    keys = sorted(vals); arr = [vals[k] for k in keys]
    rng = np.random.default_rng(seed)
    ms = np.array([float(np.mean([arr[i] for i in rng.integers(0, len(arr), len(arr))])) for _ in range(NBOOT)])
    return [round(float(np.quantile(ms, 0.025)), 4), round(float(np.quantile(ms, 0.975)), 4)]


def main():
    t0 = time.time()
    utts = load_slice()
    print(f"server={BASE} n={len(utts)} budget={BUDGET} temp={TEMP} (self-cert/conf-vote/judge selectors)", flush=True)
    rows, skipped = [], []
    for k, u in enumerate(utts):
        ch, gt, instr = u["choices"], u["gt"], fix_instruction(u["choices"])
        try:
            gtxt, _ = gen(u["wav"], instr, True, 42)
            g = parse_choice(gtxt, ch)
            samples = [gen(u["wav"], instr, False, 100 + i, want_logprobs=True) for i in range(BUDGET)]
            opts = [parse_choice(s[0], ch) for s in samples]
            confs = [s[1] for s in samples]
            # judge: show the distinct sampled options + vote counts, re-pick
            vc = collections.Counter(o for o in opts if o >= 0)
            cand = "\n".join(f"{LETTERS[o]}. {ch[o]}  (chosen {n}/{BUDGET} times)" for o, n in vc.most_common())
            jinstr = (f"You answered a multiple-choice question about the audio several times and produced "
                      f"these candidate answers:\n{cand}\nRe-examine the audio and output the single correct "
                      f"option (letter and text).")
            jtxt, _ = gen(u["wav"], jinstr, True, 42)
            judge = parse_choice(jtxt, ch)
        except Exception as e:
            print(f"  [{k+1}] SKIP ({type(e).__name__}: {str(e)[:80]})", flush=True); skipped.append(k); continue
        valid = [(o, c) for o, c in zip(opts, confs) if o >= 0]
        majority = collections.Counter(o for o, _ in valid).most_common(1)[0][0] if valid else g
        self_cert = max(valid, key=lambda oc: oc[1])[0] if valid else g
        cw = collections.defaultdict(float)
        for o, c in valid:
            cw[o] += math.exp(c)          # confidence-weighted (prob-space) vote
        conf_vote = max(cw, key=cw.get) if cw else g
        oracle = 1 if gt in opts else 0
        rows.append({"task": u["task"], "gt": gt,
                     "greedy": int(g == gt), "majority": int(majority == gt),
                     "self_certainty": int(self_cert == gt), "conf_weighted_vote": int(conf_vote == gt),
                     "judge": int(judge == gt), "oracle": oracle})
        if (k + 1) % 25 == 0:
            acc = lambda key: np.mean([r[key] for r in rows])
            print(f"  [{k+1}/{len(utts)}] greedy={acc('greedy'):.3f} maj={acc('majority'):.3f} "
                  f"selfcert={acc('self_certainty'):.3f} confvote={acc('conf_weighted_vote'):.3f} "
                  f"judge={acc('judge'):.3f} oracle={acc('oracle'):.3f} ({time.time()-t0:.0f}s)", flush=True)

    m = lambda key: round(float(np.mean([r[key] for r in rows])), 4)
    idxd = lambda f: {i: f(r) for i, r in enumerate(rows)}
    greedy, oracle = m("greedy"), m("oracle")
    headroom = oracle - greedy
    sels = ["majority", "self_certainty", "conf_weighted_vote", "judge"]
    selstats = {}
    for s in sels:
        acc = m(s)
        rho = (acc - greedy) / headroom if headroom > 0 else None
        selstats[s] = {"acc": acc, "realized_over_greedy": round(acc - greedy, 4),
                       "rho": round(rho, 4) if rho is not None else None,
                       "ci_vs_majority": cboot(idxd(lambda r, s=s: r[s] - r["majority"]))}
    best = max(sels, key=lambda s: selstats[s]["acc"])
    summary = {
        "problem": "CP-3 (c) realization — modern label-free selectors on MMAU-mini pools (frozen Qwen3-Omni-30B)",
        "stage": "1-directional", "grade": "[directional | n=%d | single-touch | not significance-bearing]" % len(rows),
        "n_utts": len(rows), "n_skipped": len(skipped), "budget": BUDGET, "temp": TEMP,
        "dataset": "mmau-mini test_mini (SAME n=150 slice as E3)", "slice_seed": SLICE_SEED,
        "greedy": greedy, "oracle": oracle, "oracle_headroom": round(headroom, 4),
        "selectors": selstats, "best_selector": best,
        "model": "qwen3-omni-30b-a3b-instruct Q8_0 GGUF (llama.cpp, -ngl 28; audio EXPERIMENTAL)",
        "prereg": "module docstring, committed before generation (git history is the anchor)",
    }
    bci = selstats[best]["ci_vs_majority"]
    summary["verdict"] = (
        f"greedy {greedy:.3f}, oracle {oracle:.3f} (headroom {headroom:+.4f}). Best label-free selector "
        f"= {best} acc {selstats[best]['acc']:.3f}, ρ={selstats[best]['rho']} (vs majority CI {bci}). "
        f"self-certainty ρ={selstats['self_certainty']['rho']}, conf-vote ρ={selstats['conf_weighted_vote']['rho']}, "
        f"judge ρ={selstats['judge']['rho']}. (c)-realization "
        f"{'is NONTRIVIAL: a modern selector harvests real headroom' if selstats[best]['rho'] and selstats[best]['rho'] > 0.15 and bci[0] > 0 else 'stays ~0: no label-free selector significantly beats majority — the +0.133 headroom is not deployably realizable'}.")
    out = {"summary": summary, "per_utt": rows, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp3_selector_realization_mmau.py (llama-server resident)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("CP3_REAL_DONE", flush=True)


if __name__ == "__main__":
    main()

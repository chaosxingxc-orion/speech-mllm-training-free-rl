"""CP-3 selector anatomy on the committed C1 pools — measure ρ(ASR) properly (CPU-only, no generation).

MINI-PREREG (committed before running; Stage-2 dev measurement of the K2-selected problem CP-3):
  Question: over the SAME frozen best-of-8 candidate pools, how much of the oracle-over-8 WER headroom
  does each LABEL-FREE selector realize (ρ = realized / oracle headroom)? Tests the house prior
  ρ(ASR) ≈ 0 [hypothesis-grade] with a wider selector family than the single MBR-overlap already run:
    - greedy (N=1 baseline)                          — the deployed default
    - MBR-overlap (mean −WER utility)                — the C1 deployable selector (re-computed)
    - consensus / plurality (exact-string majority)  — the cheapest label-free vote
    - frozen-LM rescoring (trigram LM over LibriSpeech-960h train transcripts, Kneser-Ney-free
      add-k interpolation) — the classical n-best LM-rescoring utility, the offline analog of the
      frozen-LM PLL-MBR positive (arXiv:2606.23306, ~9% rel on CTC pools)
    - MBR ⊕ LM (z-score sum)                          — combined fluency + consensus
  vs the pool oracle (reward = −WER vs reference) as the ceiling.
  Readouts per selector: mean WER; realized headroom (greedy − WER_sel); ρ = realized / (greedy −
  oracle); paired cluster-bootstrap 95% CI on the per-utterance (MBR_wer − sel_wer) and
  (greedy − sel_wer). Grade: [directional] — n=144 pooled dev pool (the C1 pool; CP-3 is a NEW
  selector-family question over it, so re-scoring is legitimate as a dev measurement); a fresh
  test-other slice is the Stage-2 confirmatory follow-up. self-certainty needs per-token logprobs
  absent from the committed artifact → deferred to a GPU regen arm.
  Budget: CPU only, < 5 min incl. trigram-LM build; zero generation.

reproduce:
  SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp3_selector_anatomy.py
"""
import collections, glob, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/d/chao_workspace/exploring-l4-intelligence/common/src")
from speechrl_common.rl.reward import wer, _normalize
from m5_selector_rescore_dev import mbr_utils, zscore

DATA = Path(os.environ["SPEECHRL_DATA_DIR"])
ART = Path(__file__).resolve().parent.parent / "_repro" / "asr_bon_llamacpp_snr5.json"
LM_CACHE = DATA / "_repro" / "librispeech_train960_trigram.json"
OUT = DATA / "_repro" / "cp3_selector_anatomy.json"
NBOOT = 10000
ADD_K = 0.1
# interpolation weights for the (uni, bi, tri) mixture
L1, L2, L3 = 0.1, 0.3, 0.6


def build_trigram_lm():
    if LM_CACHE.exists():
        return json.load(open(LM_CACHE))
    import pyarrow.parquet as pq
    uni = collections.Counter()
    bi = collections.Counter()
    tri = collections.Counter()
    n_tok = 0
    for split in ("train.clean.100", "train.clean.360", "train.other.500"):
        for f in sorted(glob.glob(str(DATA / "datasets/librispeech/all" / split / "*.parquet"))):
            for s in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                toks = ["<s>", "<s>"] + _normalize(s).split() + ["</s>"]
                n_tok += len(toks) - 2
                for i in range(2, len(toks)):
                    uni[toks[i]] += 1
                    bi[(toks[i-1], toks[i])] += 1
                    tri[(toks[i-2], toks[i-1], toks[i])] += 1
    lm = {"V": len(uni), "n_tok": n_tok,
          "uni": dict(uni),
          "bi": {" ".join(k): v for k, v in bi.items()},
          "tri": {" ".join(k): v for k, v in tri.items()}}
    LM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(lm, open(LM_CACHE, "w"))
    print(f"trigram LM built: V={lm['V']} tokens={n_tok}", flush=True)
    return lm


class LM:
    def __init__(self, lm):
        self.V = lm["V"]
        self.uni = lm["uni"]; self.bi = lm["bi"]; self.tri = lm["tri"]
        self.tot = sum(self.uni.values())

    def logprob(self, text):
        toks = ["<s>", "<s>"] + _normalize(text).split() + ["</s>"]
        lp = 0.0
        for i in range(2, len(toks)):
            w, w1, w2 = toks[i], toks[i-1], toks[i-2]
            p_uni = (self.uni.get(w, 0) + ADD_K) / (self.tot + ADD_K * self.V)
            bi_ctx = self.uni.get(w1, 0)
            p_bi = (self.bi.get(f"{w1} {w}", 0) + ADD_K) / (bi_ctx + ADD_K * self.V) if bi_ctx else p_uni
            tri_ctx = self.bi.get(f"{w2} {w1}", 0)
            p_tri = (self.tri.get(f"{w2} {w1} {w}", 0) + ADD_K) / (tri_ctx + ADD_K * self.V) if tri_ctx else p_bi
            lp += math.log(L1 * p_uni + L2 * p_bi + L3 * p_tri)
        return lp / max(1, len(toks) - 2)  # per-token, length-normalized


def consensus_pick(pool):
    norm = [_normalize(c) for c in pool]
    cnt = collections.Counter(norm)
    best = max(range(len(pool)), key=lambda j: (cnt[norm[j]], -j))
    return best


def main():
    t0 = time.time()
    lm = LM(build_trigram_lm())
    rows = json.load(open(ART))["per_utt"]
    key = lambda u: u["id"] + "|" + str(u["gen_seed"])

    per = {}   # selector -> {utt_key: sel_wer}
    for u in rows:
        pool, ref = u["pool"], u["ref"]
        util = mbr_utils(pool)
        lm_scores = [lm.logprob(c) for c in pool]
        z_mbr, z_lm = zscore(util), zscore(lm_scores)
        picks = {
            "greedy": None,  # handled via greedy_wer
            "mbr": int(np.argmax(util)),
            "consensus": consensus_pick(pool),
            "lm_rescore": int(np.argmax(lm_scores)),
            "mbr_plus_lm": int(np.argmax([z_mbr[j] + z_lm[j] for j in range(len(pool))])),
            "oracle": int(np.argmin([wer(ref, c) for c in pool])),
        }
        for name, j in picks.items():
            per.setdefault(name, {})[key(u)] = u["greedy_wer"] if name == "greedy" else wer(ref, pool[j])

    greedy = {key(u): u["greedy_wer"] for u in rows}
    oracle = per["oracle"]
    mean = lambda d: float(np.mean(list(d.values())))
    g_bar, o_bar = mean(greedy), mean(oracle)
    headroom = g_bar - o_bar

    def cboot(delta_by_utt):
        utts = sorted(delta_by_utt)
        rng = np.random.default_rng(42)
        ms = np.empty(NBOOT)
        arr = [delta_by_utt[u] for u in utts]
        n = len(arr)
        for b in range(NBOOT):
            ms[b] = float(np.mean([arr[i] for i in rng.integers(0, n, n)]))
        return [round(float(np.quantile(ms, 0.025)), 5), round(float(np.quantile(ms, 0.975)), 5)]

    selectors = {}
    for name, d in per.items():
        wer_bar = mean(d)
        realized = g_bar - wer_bar
        rho = realized / headroom if headroom > 0 else None
        vs_greedy = {u: greedy[u] - d[u] for u in d}
        vs_mbr = {u: per["mbr"][u] - d[u] for u in d}
        selectors[name] = {
            "wer": round(wer_bar, 5),
            "realized_headroom_vs_greedy": round(realized, 5),
            "rho_realized_fraction": round(rho, 4) if rho is not None else None,
            "delta_vs_greedy_mean": round(mean(vs_greedy), 5),
            "delta_vs_greedy_ci95": cboot(vs_greedy),
            "delta_vs_mbr_mean": round(mean(vs_mbr), 5),
            "delta_vs_mbr_ci95": cboot(vs_mbr),
        }

    summary = {
        "problem": "CP-3 selector/utility anatomy — rho(ASR) on frozen C1 pools",
        "stage": "2-dev", "grade": "[directional | n=144 pooled dev pool | not significance-bearing]",
        "pool_artifact": "_repro/asr_bon_llamacpp_snr5.json (144 utts x 8, 3 gen seeds, snr +5 dB)",
        "greedy_wer": round(g_bar, 5), "oracle_wer_8": round(o_bar, 5),
        "oracle_headroom": round(headroom, 5),
        "selectors": selectors,
        "lm": {"type": "trigram add-k interpolated over LibriSpeech-960h train", "add_k": ADD_K,
               "lambdas": [L1, L2, L3], "V": lm.V},
        "verdict": None,
        "prereg": "module docstring, committed before running (git history is the anchor)",
        "note": ("self-certainty deferred (needs per-token logprobs absent from the committed pool); "
                 "a fresh test-other confirmatory slice is the Stage-2 follow-up"),
    }
    # mechanical verdict line
    beats_mbr = [n for n, s in selectors.items()
                 if n not in ("greedy", "oracle", "mbr") and s["delta_vs_mbr_ci95"][0] > 0]
    best = max((n for n in selectors if n not in ("greedy", "oracle")),
               key=lambda n: selectors[n]["rho_realized_fraction"] or -1)
    summary["verdict"] = (
        f"oracle headroom {headroom:+.4f}; best deployable selector = {best} "
        f"(ρ={selectors[best]['rho_realized_fraction']}, WER {selectors[best]['wer']:.4f}); "
        f"selectors beating MBR at CI-LB>0: {beats_mbr or 'none'}. "
        f"House ρ≈0 prior {'CONTRADICTED (a selector realizes real headroom)' if beats_mbr else 'consistent (no selector significantly beats MBR)'}.")
    out = {"summary": summary, "elapsed_s": round(time.time() - t0, 1),
           "reproduce": "SPEECHRL_DATA_DIR=<repo>/speechrl-data python scripts/cp3_selector_anatomy.py (CPU-only)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    print("wrote", OUT, flush=True)
    print("CP3_DONE", flush=True)


if __name__ == "__main__":
    main()

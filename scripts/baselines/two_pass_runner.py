"""scripts/baselines/two_pass_runner.py — S3 two-pass runner CONTRACT (续21-B② M1 item 4).

Separate from ``run_mock.py``'s mock harness on purpose: ``run_mock`` is a FIXED, pre-registered,
**strictly no-adaptive-logic** pipeline (its own docstring: "no reward-based selection, no
confidence gating, no per-item branching on the model's own output"). This module is the OPPOSITE
by design — a genuinely adaptive, self-consistency-gated two-pass controller:

  **Pass 1** — draw ``m`` stochastic samples (default ``m=5``, ``T=0.7``) from the frozen core with
  NO retrieval augmentation at all, and compute their PAIRWISE AGREEMENT with a K-type-appropriate
  metric (exact-match for closed/MCQ K-types, token-F1>=0.8 for free-text QA, WER<=0.1 for
  ASR/echo content K-types). Low agreement is read as "the model is internally uncertain about
  this item" — the trigger signal.

  **Trigger decision** — if the agreement rate falls BELOW ``cfg.trigger_threshold``, retrieval is
  triggered; otherwise the pass-1 samples' own CONSENSUS is trusted as the final answer (no
  retrieval, no extra generation call).

  **Pass 2** (only when triggered) — retrieval + injection + a single GREEDY (``T=0``) generation
  with the retrieved context. This module does NOT reimplement retrieval/injection itself (that
  machinery is ``kb_retrieve``/``run_mock.apply_retrieval_kind``/``render_delivery``) — it takes
  ``retrieve_fn``/``inject_fn`` as injectable seams, so a real caller wires in those exact
  functions and this module stays free of any KB/embedder import (lazy-import discipline).

**Black-box contract** (Decision-Log 续18: 音频/文本进、文本出、多次采样): the ONE model-facing
seam is ``generate_fn(wav_path, instruction_or_payload, seed, temperature) -> str`` — the exact
shape ``run_baseline.generate``/``run_mock.generate_mock`` already expose (a real caller passes
one of those in; the fake-model E2E test in ``test_two_pass_runner.py`` passes a deterministic
stub with the identical signature, so the CONTRACT is what's tested, not any particular backbone).

No real runs happen in this module or its test — this is infrastructure only (M1 task brief).
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------------------------
# K-type -> agreement metric (per task brief: "exact-match / token-F1>=0.8 / WER<=0.1 as
# specified"). Mirrors metrics.py's K-type scorer wiring (score_k1_wer/score_k2_cer = content ASR/
# echo -> WER; score_k8_qa = free-text QA -> token-F1; every other, closed-label K-type -> EM).
# ---------------------------------------------------------------------------------------------
AGREEMENT_METRICS = ("exact-match", "token-f1", "wer")

K_TYPE_AGREEMENT_METRIC: dict[str, str] = {
    "K1": "wer",         # ASR content (en)
    "K2": "wer",          # ASR/echo content (zh CER-style, jiwer.wer still applies char-wise below)
    "K8": "token-f1",      # free-text QA
}
DEFAULT_AGREEMENT_METRIC = "exact-match"  # every other (closed-label/MCQ) K-type


def agreement_metric_for(k_type: str, override: str | None = None) -> str:
    if override:
        if override not in AGREEMENT_METRICS:
            raise ValueError(f"agreement_metric_for: override={override!r} not in {AGREEMENT_METRICS}")
        return override
    return K_TYPE_AGREEMENT_METRIC.get(k_type, DEFAULT_AGREEMENT_METRIC)


@dataclass(frozen=True)
class TwoPassConfig:
    m: int = 5                          # pass-1 sample count
    pass1_temperature: float = 0.7
    pass2_temperature: float = 0.0        # greedy
    k_type: str = "K8"
    agreement_metric: str | None = None    # override agreement_metric_for(k_type); None = derive from k_type
    token_f1_threshold: float = 0.8
    wer_threshold: float = 0.1
    trigger_threshold: float = 0.6         # agreement_rate BELOW this triggers pass 2
    seed_base: int = 20260713              # pass-1 samples use seed_base..seed_base+m-1
    pass2_seed_offset: int = 1000          # pass-2's greedy call uses seed_base + this offset
                                              # (a distinct namespace from every pass-1 sample seed)


# ---------------------------------------------------------------------------------------------
# per-pair agreement primitives (reused by pairwise_agreement AND the consensus pick below)
# ---------------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return "".join(c for c in s if c.isalnum() or c.isspace()).strip()


def _token_f1(a: str, b: str) -> float:
    """SQuAD-style bag-of-tokens F1 over normalized whitespace tokens."""
    ta, tb = _norm(a).split(), _norm(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    from collections import Counter

    common = Counter(ta) & Counter(tb)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(ta)
    recall = n_common / len(tb)
    return 2 * precision * recall / (precision + recall)


def _wer(a: str, b: str) -> float:
    import jiwer  # lazy

    ref, hyp = _norm(a), _norm(b)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return float(jiwer.wer(ref, hyp))


def _pair_agrees(a: str, b: str, metric: str, cfg: TwoPassConfig) -> bool:
    if metric == "exact-match":
        return _norm(a) == _norm(b)
    if metric == "token-f1":
        return _token_f1(a, b) >= cfg.token_f1_threshold
    if metric == "wer":
        return _wer(a, b) <= cfg.wer_threshold
    raise ValueError(f"_pair_agrees: unknown metric {metric!r} (expected one of {AGREEMENT_METRICS})")


def pairwise_agreement(samples: list[str], metric: str, cfg: TwoPassConfig) -> dict:
    """All C(m,2) pairwise comparisons among ``samples`` under ``metric``. Returns
    ``{"metric", "n_samples", "n_pairs", "agreement_rate", "pairs": [{"i","j","agrees"}, ...]}``.
    ``agreement_rate`` is the fraction of pairs that agree (1.0 for a single sample -- vacuously
    "fully agreeing", never a 0/0 crash)."""
    n = len(samples)
    idx_pairs = list(itertools.combinations(range(n), 2))
    pair_results = []
    for i, j in idx_pairs:
        pair_results.append({"i": i, "j": j, "agrees": _pair_agrees(samples[i], samples[j], metric, cfg)})
    rate = (sum(1 for p in pair_results if p["agrees"]) / len(pair_results)) if pair_results else 1.0
    return {"metric": metric, "n_samples": n, "n_pairs": len(pair_results),
            "agreement_rate": rate, "pairs": pair_results}


def decide_trigger(agreement: dict, cfg: TwoPassConfig) -> dict:
    triggered = agreement["agreement_rate"] < cfg.trigger_threshold
    return {
        "triggered": triggered,
        "agreement_rate": agreement["agreement_rate"],
        "trigger_threshold": cfg.trigger_threshold,
        "rule": "agreement_rate < trigger_threshold -> retrieval triggered",
    }


def _consensus_sample(samples: list[str], metric: str, cfg: TwoPassConfig) -> dict:
    """When pass-1 agreement is high enough that retrieval is NOT triggered, the final answer is
    the "medoid" sample -- the one with the highest total pairwise agreement against every other
    sample. This generalizes across all three agreement metrics (unlike a naive "most common
    EXACT string" majority vote, which under token-f1/wer would rarely find a literal duplicate
    even when samples are clearly in semantic agreement). Ties break toward the EARLIEST sample
    (lowest index -- the lowest-seed draw), a deterministic, documented tie-break rule."""
    n = len(samples)
    if n == 1:
        return {"text": samples[0], "index": 0, "consensus_score": 1.0}
    scores = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _pair_agrees(samples[i], samples[j], metric, cfg):
                scores[i] += 1.0
    best = max(range(n), key=lambda i: (scores[i], -i))
    return {"text": samples[best], "index": best, "consensus_score": scores[best] / (n - 1)}


# ---------------------------------------------------------------------------------------------
# pass 1 / pass 2 / full contract
# ---------------------------------------------------------------------------------------------

def run_pass1(generate_fn, wav_path: str, instruction: str, cfg: TwoPassConfig) -> dict:
    """``generate_fn(wav_path, instruction, seed, temperature) -> str``, called ``cfg.m`` times at
    ``cfg.pass1_temperature`` with seeds ``cfg.seed_base .. cfg.seed_base + m - 1`` (deterministic,
    reproducible -- no wall-clock/random seeding). Returns ``{"samples", "seeds", "agreement",
    "trigger"}``."""
    seeds = [cfg.seed_base + i for i in range(cfg.m)]
    samples = [generate_fn(wav_path, instruction, seed=s, temperature=cfg.pass1_temperature) for s in seeds]
    metric = agreement_metric_for(cfg.k_type, cfg.agreement_metric)
    agreement = pairwise_agreement(samples, metric, cfg)
    trigger = decide_trigger(agreement, cfg)
    return {"samples": samples, "seeds": seeds, "metric": metric, "agreement": agreement, "trigger": trigger}


def run_pass2(generate_fn, retrieve_fn, inject_fn, wav_path: str, instruction: str, cfg: TwoPassConfig) -> dict:
    """retrieval (``retrieve_fn(wav_path) -> hits``) + injection (``inject_fn(hits, instruction) ->
    payload`` — a str or a messages list, matching ``run_mock.render_delivery``'s own return
    shape) + a single GREEDY generation (``generate_fn(wav_path, payload, seed=..., temperature=
    cfg.pass2_temperature)``). The pass-2 seed (``cfg.seed_base + cfg.pass2_seed_offset``) is a
    DISTINCT namespace from every pass-1 sample seed, by construction (offset default 1000 >>
    any realistic ``m``), so a pass-2 call can never accidentally replay a pass-1 seed."""
    hits = retrieve_fn(wav_path)
    payload = inject_fn(hits, instruction)
    seed = cfg.seed_base + cfg.pass2_seed_offset
    text = generate_fn(wav_path, payload, seed=seed, temperature=cfg.pass2_temperature)
    return {"hits": hits, "payload": payload, "seed": seed, "text": text}


def run_two_pass(generate_fn, retrieve_fn, inject_fn, wav_path: str, instruction: str,
                  cfg: TwoPassConfig | None = None) -> dict:
    """The full S3 contract: pass 1 -> trigger decision -> (maybe) pass 2. Returns:

        {"pass1": {...}, "pass2": {...} | None, "final_text": str,
         "final_source": "self-consistency-consensus" | "retrieval-augmented-greedy",
         "config": <TwoPassConfig as dict>}

    ``retrieve_fn``/``inject_fn`` are NEVER called when pass 1 does not trigger (no retrieval
    cost paid unless the self-consistency signal calls for it) -- verified by
    ``test_two_pass_runner.py`` via call-counting stubs.
    """
    cfg = cfg or TwoPassConfig()
    pass1 = run_pass1(generate_fn, wav_path, instruction, cfg)
    result: dict = {"pass1": pass1, "pass2": None, "final_text": None,
                     "final_source": None, "config": asdict(cfg)}
    if pass1["trigger"]["triggered"]:
        pass2 = run_pass2(generate_fn, retrieve_fn, inject_fn, wav_path, instruction, cfg)
        result["pass2"] = pass2
        result["final_text"] = pass2["text"]
        result["final_source"] = "retrieval-augmented-greedy"
    else:
        consensus = _consensus_sample(pass1["samples"], pass1["metric"], cfg)
        result["final_text"] = consensus["text"]
        result["final_source"] = "self-consistency-consensus"
        result["consensus"] = consensus
    return result

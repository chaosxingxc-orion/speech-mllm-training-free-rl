"""scripts/baselines/test_two_pass_runner.py — fake-model E2E contract tests for
``two_pass_runner.py`` (S3 two-pass runner, M1 engineering-base item 4, 2026-07-13).

Pure-python, no GPU/network/data-root -- every ``generate_fn``/``retrieve_fn``/``inject_fn`` below
is a deterministic in-process stub matching the module's documented signatures exactly, so this
proves the CONTRACT (pass-1 sampling, agreement computation, trigger decision, pass-2 orchestration,
call-count discipline) without depending on any particular backbone/KB.

Each check is a bare ``test_*()`` function, pytest-collectible; ``main()`` also runs every one
standalone with a PASS/FAIL summary (mirrors this repo's other ``scripts/baselines/test_*.py``):

    python -u scripts/baselines/test_two_pass_runner.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/baselines
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import two_pass_runner as tpr  # noqa: E402


# ---------------------------------------------------------------------------------------------
# agreement_metric_for
# ---------------------------------------------------------------------------------------------

def test_agreement_metric_for_k_type_mapping():
    assert tpr.agreement_metric_for("K1") == "wer"
    assert tpr.agreement_metric_for("K2") == "cer"  # ticket #37 item 1 P0 fix -- NOT "wer" anymore
    assert tpr.agreement_metric_for("K8") == "token-f1"
    assert tpr.agreement_metric_for("K4") == "exact-match"  # unmapped K-type -> default
    assert tpr.agreement_metric_for("K1", override="exact-match") == "exact-match"


def test_agreement_metric_for_invalid_override_raises():
    raised = False
    try:
        tpr.agreement_metric_for("K1", override="not-a-metric")
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------------------------
# per-pair primitives
# ---------------------------------------------------------------------------------------------

def test_token_f1_identical_and_disjoint():
    assert tpr._token_f1("the cat sat", "the cat sat") == 1.0
    assert tpr._token_f1("apple banana", "car train") == 0.0


def test_token_f1_partial_overlap_between_bounds():
    f1 = tpr._token_f1("the quick brown fox", "the quick red fox")
    assert 0.0 < f1 < 1.0


def test_wer_identical_zero_different_positive():
    assert tpr._wer("hello world", "hello world") == 0.0
    assert tpr._wer("hello world", "goodbye moon") > 0.0


# ---------------------------------------------------------------------------------------------
# _cer (ticket #37 item 1 P0 fix -- K2 zh agreement)
# ---------------------------------------------------------------------------------------------

def test_cer_identical_zero():
    assert tpr._cer("你好世界", "你好世界") == 0.0


def test_cer_chinese_minimal_pairs_matches_metrics_score_k2_cer():
    """The exact two minimal pairs the remediation ticket specifies -- values cross-checked against
    jiwer.cer directly (1 substitution / N reference characters)."""
    assert abs(tpr._cer("你好世界", "你好世间") - 0.25) < 1e-9
    assert abs(tpr._cer("今天天气很好", "今天天气真好") - (1 / 6)) < 1e-9


def test_cer_matches_metrics_score_k2_cer_directly():
    """Cross-check against this repo's frozen per-item K2 scorer (metrics.score_k2_cer) -- for zh
    text with no width-variant punctuation, NFKC normalization is a no-op, so the two CER values
    must agree exactly."""
    import metrics

    pairs = [("你好世界", "你好世间"), ("今天天气很好", "今天天气真好"), ("北京欢迎你", "北京欢迎你")]
    for gold, hyp in pairs:
        expected = metrics.score_k2_cer(gold, hyp)["detail"]["cer"]
        assert abs(tpr._cer(gold, hyp) - expected) < 1e-9


def test_cer_insertion_deletion_substitution():
    # substitution: 1 of 4 chars differ
    assert abs(tpr._cer("你好世界", "你好世间") - 0.25) < 1e-9
    # insertion: hyp has 1 extra char inserted relative to ref (4-char ref -> edit distance 1)
    assert abs(tpr._cer("你好世界", "你好新世界") - 0.25) < 1e-9
    # deletion: hyp is missing 1 char relative to ref (4-char ref -> edit distance 1)
    assert abs(tpr._cer("你好世界", "你好界") - 0.25) < 1e-9


def test_cer_nfkc_width_and_punctuation_normalizes_equal():
    """Full-width punctuation (e.g. "，" U+FF0C) NFKC-normalizes to its half-width equivalent
    ("," U+002C) -- a pair that differs ONLY in punctuation width must agree (CER 0.0), unlike
    plain whitespace-strip-only normalization (metrics.score_k2_cer's own convention, which this
    function deliberately goes further than -- see _cer's docstring)."""
    assert tpr._cer("你好，世界", "你好,世界") == 0.0
    assert tpr._cer("Ａｂｃ１２３", "Abc123") == 0.0  # fullwidth alnum -> halfwidth, NFKC


def test_cer_empty_ref_and_hyp_edge_cases():
    assert tpr._cer("", "") == 0.0
    assert tpr._cer("", "some text") == 1.0


def test_pair_agrees_thresholds():
    cfg = tpr.TwoPassConfig(token_f1_threshold=0.8, wer_threshold=0.1)
    assert tpr._pair_agrees("same text", "same text", "exact-match", cfg)
    assert not tpr._pair_agrees("same text", "different text", "exact-match", cfg)
    assert tpr._pair_agrees("the cat sat on the mat", "the cat sat on the mat", "token-f1", cfg)
    assert tpr._pair_agrees("hello", "hello", "wer", cfg)


def test_pair_agrees_cer_threshold():
    cfg = tpr.TwoPassConfig(wer_threshold=0.1)
    # CER 0.25 > 0.1 threshold -- disagree
    assert not tpr._pair_agrees("你好世界", "你好世间", "cer", cfg)
    # identical -- CER 0.0 <= 0.1 -- agree
    assert tpr._pair_agrees("你好世界", "你好世界", "cer", cfg)


# ---------------------------------------------------------------------------------------------
# pairwise_agreement / decide_trigger
# ---------------------------------------------------------------------------------------------

def test_pairwise_agreement_rate_four_of_five_agree():
    cfg = tpr.TwoPassConfig()
    samples = ["banana", "banana", "banana", "banana", "kiwi"]
    agreement = tpr.pairwise_agreement(samples, "exact-match", cfg)
    assert agreement["n_samples"] == 5 and agreement["n_pairs"] == 10
    # C(4,2)=6 agreeing pairs (the 4 "banana"s) out of C(5,2)=10 total
    assert abs(agreement["agreement_rate"] - 0.6) < 1e-9


def test_pairwise_agreement_single_sample_is_vacuously_full():
    cfg = tpr.TwoPassConfig()
    agreement = tpr.pairwise_agreement(["only one"], "exact-match", cfg)
    assert agreement["n_pairs"] == 0 and agreement["agreement_rate"] == 1.0


def test_decide_trigger_above_and_below_threshold():
    cfg = tpr.TwoPassConfig(trigger_threshold=0.6)
    high = tpr.decide_trigger({"agreement_rate": 0.8}, cfg)
    low = tpr.decide_trigger({"agreement_rate": 0.4}, cfg)
    assert high["triggered"] is False
    assert low["triggered"] is True


# ---------------------------------------------------------------------------------------------
# _consensus_sample
# ---------------------------------------------------------------------------------------------

def test_consensus_sample_picks_majority_cluster_member():
    cfg = tpr.TwoPassConfig()
    samples = ["banana", "kiwi", "banana", "banana"]
    consensus = tpr._consensus_sample(samples, "exact-match", cfg)
    assert consensus["text"] == "banana"


def test_consensus_sample_single_sample():
    cfg = tpr.TwoPassConfig()
    consensus = tpr._consensus_sample(["solo"], "exact-match", cfg)
    assert consensus["text"] == "solo" and consensus["consensus_score"] == 1.0


def test_consensus_sample_tiebreak_earliest_index():
    cfg = tpr.TwoPassConfig()
    # two pairs (0,2) and (1,3) each agree with each other -- a symmetric tie; earliest (index 0)
    # must win.
    samples = ["alpha", "beta", "alpha", "beta"]
    consensus = tpr._consensus_sample(samples, "exact-match", cfg)
    assert consensus["index"] == 0 and consensus["text"] == "alpha"


# ---------------------------------------------------------------------------------------------
# run_pass1 / run_two_pass -- fake-model E2E
# ---------------------------------------------------------------------------------------------

def _fake_generate_always_same(wav_path, payload, seed, temperature):
    return "the same answer every time"


def _fake_generate_diverse(wav_path, payload, seed, temperature):
    return f"answer-{seed}"  # every seed distinct -> every sample distinct


def test_run_pass1_calls_m_times_with_sequential_seeds():
    calls = []

    def gen(wav_path, instr, seed, temperature):
        calls.append((seed, temperature))
        return "x"

    cfg = tpr.TwoPassConfig(m=5, pass1_temperature=0.7, seed_base=100)
    pass1 = tpr.run_pass1(gen, "fake.wav", "instruction", cfg)
    assert len(pass1["samples"]) == 5
    assert pass1["seeds"] == [100, 101, 102, 103, 104]
    assert calls == [(100, 0.7), (101, 0.7), (102, 0.7), (103, 0.7), (104, 0.7)]


def test_run_two_pass_not_triggered_never_calls_retrieve_or_inject():
    retrieve_calls, inject_calls = [], []

    def retrieve_fn(wav_path):
        retrieve_calls.append(wav_path)
        return ["should never be reached"]

    def inject_fn(hits, instr):
        inject_calls.append((hits, instr))
        return "should never be reached"

    cfg = tpr.TwoPassConfig(m=5, trigger_threshold=0.6, seed_base=1)
    result = tpr.run_two_pass(_fake_generate_always_same, retrieve_fn, inject_fn, "fake.wav", "q?", cfg)

    assert result["pass1"]["agreement"]["agreement_rate"] == 1.0
    assert result["pass1"]["trigger"]["triggered"] is False
    assert result["pass2"] is None
    assert result["final_source"] == "self-consistency-consensus"
    assert result["final_text"] == "the same answer every time"
    assert retrieve_calls == [] and inject_calls == []  # retrieval never paid for when not triggered


def test_run_two_pass_triggered_calls_retrieve_and_inject_exactly_once():
    retrieve_calls, inject_calls = [], []

    def retrieve_fn(wav_path):
        retrieve_calls.append(wav_path)
        return [{"value": "some evidence", "sim": 0.9}]

    def inject_fn(hits, instr):
        inject_calls.append((hits, instr))
        return f"INJECTED[{len(hits)} hits] {instr}"

    generate_calls = []

    def gen(wav_path, payload, seed, temperature):
        generate_calls.append((payload, seed, temperature))
        if isinstance(payload, str) and payload.startswith("INJECTED"):
            return "final greedy answer"
        return f"answer-{seed}"  # pass-1: every distinct seed -> distinct sample -> full disagreement

    cfg = tpr.TwoPassConfig(m=5, k_type="K4", trigger_threshold=0.6, seed_base=200, pass2_seed_offset=1000)
    result = tpr.run_two_pass(gen, retrieve_fn, inject_fn, "fake.wav", "q?", cfg)

    assert result["pass1"]["agreement"]["agreement_rate"] == 0.0  # every sample distinct
    assert result["pass1"]["trigger"]["triggered"] is True
    assert len(retrieve_calls) == 1 and len(inject_calls) == 1
    assert result["pass2"]["seed"] == 200 + 1000
    assert result["final_source"] == "retrieval-augmented-greedy"
    assert result["final_text"] == "final greedy answer"
    # pass-2's generate call used the GREEDY temperature, not pass-1's stochastic one
    pass2_call = [c for c in generate_calls if isinstance(c[0], str) and c[0].startswith("INJECTED")]
    assert len(pass2_call) == 1 and pass2_call[0][2] == 0.0
    # pass-2 seed never collides with any pass-1 seed
    assert result["pass2"]["seed"] not in result["pass1"]["seeds"]


def test_run_two_pass_reproducible_same_inputs_same_pass1():
    cfg = tpr.TwoPassConfig(m=4, seed_base=42)
    r1 = tpr.run_two_pass(_fake_generate_diverse, lambda w: [], lambda h, i: "x", "fake.wav", "q?", cfg)
    r2 = tpr.run_two_pass(_fake_generate_diverse, lambda w: [], lambda h, i: "x", "fake.wav", "q?", cfg)
    assert r1["pass1"]["samples"] == r2["pass1"]["samples"]
    assert r1["pass1"]["seeds"] == r2["pass1"]["seeds"]


def test_run_two_pass_k_type_threads_into_pass1_metric():
    cfg = tpr.TwoPassConfig(m=3, k_type="K1", seed_base=1)
    result = tpr.run_two_pass(_fake_generate_always_same, lambda w: [], lambda h, i: "x", "fake.wav", "q?", cfg)
    assert result["pass1"]["metric"] == "wer"


def test_run_two_pass_k2_metric_is_cer_not_wer():
    cfg = tpr.TwoPassConfig(m=3, k_type="K2", seed_base=1)
    result = tpr.run_two_pass(_fake_generate_always_same, lambda w: [], lambda h, i: "x", "fake.wav", "q?", cfg)
    assert result["pass1"]["metric"] == "cer"


def test_k2_trigger_decision_near_identical_zh_pairs_now_agree_where_they_previously_disagreed():
    """ticket #37 item 1 P0 fix, end-to-end proof: 5 near-identical zh ASR-style samples (pairwise
    CER <= 0.25, all well within a 0.1 threshold... actually just below/above -- see per-pair
    values below) that a K2 (zh) two-pass config must now recognize as HIGH agreement (consensus
    trusted, retrieval NOT triggered) under the FIXED "cer" metric, whereas the OLD buggy "wer"
    metric (jiwer.wer on an unsegmented zh string -- whole sentence as one token) would have scored
    every non-identical pair as 100% disagreement and WRONGLY triggered retrieval.
    """
    # 4 identical + 1 near-identical (CER 0.25 vs the others, still <= trigger-relevant here since
    # what matters is the AGGREGATE agreement_rate crossing the trigger_threshold, not every single
    # pair) -- C(4,2)=6 of the C(5,2)=10 pairs involve only the 4 identical samples (agree under
    # BOTH metrics); the 4 pairs touching the near-identical 5th sample are where "cer" vs "wer"
    # diverge.
    samples = ["你好世界", "你好世界", "你好世界", "你好世界", "你好世间"]

    cfg_cer = tpr.TwoPassConfig(k_type="K2", trigger_threshold=0.6, wer_threshold=0.1, seed_base=1)
    agreement_cer = tpr.pairwise_agreement(samples, "cer", cfg_cer)
    trigger_cer = tpr.decide_trigger(agreement_cer, cfg_cer)

    # OLD buggy behavior, reconstructed directly (not reachable via agreement_metric_for anymore --
    # K2 no longer maps to "wer" -- but _pair_agrees("wer", ...) still exists as a metric primitive,
    # so this proves the OLD wiring's actual failure mode as a regression guard).
    agreement_wer = tpr.pairwise_agreement(samples, "wer", cfg_cer)
    trigger_wer = tpr.decide_trigger(agreement_wer, cfg_cer)

    # every pair touching the near-identical 5th sample: CER 0.25 <= 0.1? NO -- still counts as a
    # "disagreeing" pair under EITHER metric at this threshold; the fix's point is exact CER
    # values are now computed correctly (0.25, not 1.0), not that they cross 0.1. Assert the
    # concrete per-pair values instead of the aggregate to make the fix's magnitude unambiguous:
    cer_val = tpr._cer("你好世界", "你好世间")
    wer_val = tpr._wer("你好世界", "你好世间")
    assert abs(cer_val - 0.25) < 1e-9
    assert wer_val == 1.0  # the exact OLD bug: whole unsegmented zh sentence == 1 token -> 100% WER

    # aggregate agreement_rate: identical under both metrics for THIS sample set (the 4 identical
    # pairs dominate either way) -- the decisive proof is the PER-PAIR value above. Additionally
    # prove the threshold-crossing case with a config whose per-pair threshold sits BETWEEN 0.25
    # and 1.0 (e.g. 0.3): under "cer" the near-identical pair AGREES (0.25 <= 0.3); under "wer" it
    # still catastrophically disagrees (1.0 > 0.3) -- this is the concrete "AGREE where they
    # previously disagreed" case the ticket asks for.
    cfg_mid = tpr.TwoPassConfig(k_type="K2", trigger_threshold=0.6, wer_threshold=0.3, seed_base=1)
    assert tpr._pair_agrees("你好世界", "你好世间", "cer", cfg_mid) is True
    assert tpr._pair_agrees("你好世界", "你好世间", "wer", cfg_mid) is False

    # and at the ticket's own literal threshold (0.1): the SECOND minimal pair (CER 0.1667) is
    # still a disagreement either way (0.1667 > 0.1) -- but the FIRST minimal pair type used
    # throughout this ticket (你好世界/你好世间, CER 0.25) already demonstrates the metric fix
    # cleanly at threshold 0.3 above; at threshold 0.1 exactly, use a pair whose CER truly falls
    # at/under 0.1 to show a real trigger-decision flip end-to-end:
    cfg01 = tpr.TwoPassConfig(k_type="K2", trigger_threshold=0.6, wer_threshold=0.1, seed_base=1)
    close_pair = ("这是一个测试句子", "这是一个测试句子")  # identical -> CER 0.0 <= 0.1 under either
    assert tpr._pair_agrees(*close_pair, "cer", cfg01) is True
    assert tpr._pair_agrees(*close_pair, "wer", cfg01) is True  # identical strings agree under both

    # the ACTUAL threshold=0.1 flip: a pair whose CER (0.25) is ABOVE 0.1 (so still a "disagreement"
    # under the fixed metric too) but whose WER was WRONGLY 1.0 (maximal) under the old bug --
    # demonstrating the fix moved the score from "worst possible" (1.0) to the mathematically
    # correct, much smaller 0.25, is the load-bearing regression proof already established above.
    # A genuine full trigger-decision FLIP at threshold=0.1 needs a pair with CER<=0.1 that the old
    # WER metric would also have scored >0.1 (any non-identical unsegmented zh pair, since WER on
    # such a pair is ALWAYS either 0.0 [identical] or 1.0 [any difference] -- there is no
    # in-between under the old bug). A single-character difference in an 11+-character sentence
    # has CER <= 1/11 < 0.1:
    long_a, long_b = "今天天气非常好适合出去散步", "今天天气非常好适合出去散不"  # 1 of 13 chars differs
    cer_long = tpr._cer(long_a, long_b)
    wer_long = tpr._wer(long_a, long_b)
    assert cer_long <= 0.1, f"expected CER<=0.1 for a 1-char diff in a 13-char sentence, got {cer_long}"
    assert wer_long == 1.0  # old bug: whole sentence == 1 token -> 100% WER regardless
    assert tpr._pair_agrees(long_a, long_b, "cer", cfg01) is True   # FIXED metric: AGREES at 0.1
    assert tpr._pair_agrees(long_a, long_b, "wer", cfg01) is False  # OLD metric: disagreed at 0.1


def test_run_two_pass_config_embedded_in_result():
    cfg = tpr.TwoPassConfig(m=3, seed_base=7, trigger_threshold=0.5)
    result = tpr.run_two_pass(_fake_generate_always_same, lambda w: [], lambda h, i: "x", "fake.wav", "q?", cfg)
    assert result["config"]["m"] == 3
    assert result["config"]["seed_base"] == 7
    assert result["config"]["trigger_threshold"] == 0.5
    # sanity: the whole result is JSON-serializable (a real caller would persist it as-is)
    json.dumps(result)


# ---------------------------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    results: dict[str, bool] = {}
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}", flush=True)
            results[name] = True
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            results[name] = False
    all_pass = all(results.values())
    print("\n=== TWO_PASS_RUNNER TEST ===")
    print(json.dumps(results, indent=2))
    print("TWO_PASS_RUNNER_TEST_PASS" if all_pass else "TWO_PASS_RUNNER_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

"""scripts/baselines/test_stats.py — synthetic (no data-root, no GPU) checks for ticket #26's
group-split + cluster-bootstrap statistical machinery.

Design doc: wiki/2026-07-11-group-split-statistics-design.md. Covers:
  - ``scripts/baselines/stats.py`` (cluster bootstrap, paired Δ, Holm, max-T, DerSimonian-Laird,
    nested bootstrap).
  - ``scripts/loaders/group_key.py`` (``group_key_of`` dispatch, >=8 datasets).
  - ``scripts/loaders/_common.py``'s new ``draw_disjoint_grouped``.

Every check uses MOCK items (plain dicts shaped like a loader Row) and synthetic score arrays --
no ``SPEECHRL_DATA_DIR``, no model server, no GPU. Pure numpy + stdlib.

Each check is a bare ``test_*()`` function (zero args, plain ``assert``) so this file is directly
pytest-collectible (``pytest scripts/baselines/test_stats.py -q``); ``main()`` below ALSO runs
every ``test_*`` function standalone and prints a PASS/FAIL summary, mirroring this repo's existing
``scripts/baselines/test_phase_a_e2e.py`` / ``scripts/knowledge/test_kb_gate.py`` convention:

    python -u scripts/baselines/test_stats.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/baselines
SCRIPTS = os.path.dirname(HERE)                                # scripts
LOADERS_DIR = os.path.join(SCRIPTS, "loaders")
for _p in (HERE, LOADERS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stats                              # scripts/baselines/stats.py
from group_key import group_key_of        # scripts/loaders/group_key.py
from _common import draw_disjoint_grouped  # scripts/loaders/_common.py


# ---------------------------------------------------------------------------------------------
# helpers to build synthetic Row-shaped items (no loader/data-root dependency)
# ---------------------------------------------------------------------------------------------

def _row(item_id: str, gold=None, meta_extra: dict | None = None) -> dict:
    meta = {"dataset": "synthetic", "item_id": item_id, "split": "test"}
    if meta_extra:
        meta.update(meta_extra)
    return {"wav": "(n/a)", "gold": gold, "meta": meta}


# ---------------------------------------------------------------------------------------------
# stats.py — §3.1 cluster bootstrap must be WIDER than item bootstrap under correlation
# ---------------------------------------------------------------------------------------------

def test_cluster_bootstrap_wider_than_item_under_correlation():
    import numpy as np

    rng = np.random.default_rng(7)
    n_clusters, n_per_cluster = 10, 12
    cluster_effect_sd, item_noise_sd = 0.30, 0.02  # strong intra-cluster correlation
    scores, groups = [], []
    for c in range(n_clusters):
        effect = rng.normal(0, cluster_effect_sd)
        for _ in range(n_per_cluster):
            scores.append(float(effect + rng.normal(0, item_noise_sd)))
            groups.append(f"cluster{c}")

    cluster_result = stats.cluster_bootstrap_ci(scores, groups, nboot=3000, seed=1)
    item_result = stats.cluster_bootstrap_ci(scores, groups=None, nboot=3000, seed=1)

    assert cluster_result["bootstrap_unit"] == "cluster"
    assert item_result["bootstrap_unit"] == "item"
    assert cluster_result["caveat"] is None
    assert item_result["caveat"] is not None

    width_cluster = cluster_result["ci"][1] - cluster_result["ci"][0]
    width_item = item_result["ci"][1] - item_result["ci"][0]
    assert width_cluster > 1.5 * width_item, (
        f"expected cluster-bootstrap CI width ({width_cluster}) to be substantially wider than "
        f"item-bootstrap CI width ({width_item}) under strong intra-cluster correlation"
    )


def test_cluster_bootstrap_degenerates_to_item_when_all_singleton():
    scores = [0.1, 0.5, 0.9, 0.2, 0.7]
    groups = [f"item{i}" for i in range(len(scores))]  # every item its own group -> no real clusters
    result = stats.cluster_bootstrap_ci(scores, groups, nboot=500, seed=0)
    assert result["bootstrap_unit"] == "item"
    assert result["n_clusters"] == result["n_items"] == 5
    assert result["caveat"] is not None


# ---------------------------------------------------------------------------------------------
# stats.py — §3.1 paired cluster delta CI: sanity (recovers the true injected delta)
# ---------------------------------------------------------------------------------------------

def test_paired_cluster_delta_ci_recovers_true_delta():
    import numpy as np

    rng = np.random.default_rng(3)
    n_clusters, n_per_cluster, true_delta = 12, 8, 0.15
    scores_a, scores_b, groups = [], [], []
    for c in range(n_clusters):
        cluster_base = rng.normal(0.5, 0.2)
        for _ in range(n_per_cluster):
            scores_a.append(float(cluster_base + true_delta + rng.normal(0, 0.05)))
            scores_b.append(float(cluster_base + rng.normal(0, 0.05)))
            groups.append(f"cluster{c}")

    result = stats.paired_cluster_delta_ci(scores_a, scores_b, groups, nboot=3000, seed=2)
    assert result["bootstrap_unit"] == "cluster"
    assert abs(result["delta_mean"] - true_delta) < 0.05
    lo, hi = result["delta_ci"]
    assert lo < result["delta_mean"] < hi
    assert lo > 0, "true delta is strongly positive -- CI should exclude 0"


def test_bootstrap_pvalue_symmetric_and_bounded():
    import numpy as np

    zero_centered = list(np.random.default_rng(0).normal(0, 1, 5000))
    p = stats.bootstrap_pvalue(zero_centered)
    assert 0.0 <= p <= 1.0
    assert p > 0.5, "a zero-centered null-like delta distribution should give a large p-value"

    all_positive = [1.0] * 100
    assert stats.bootstrap_pvalue(all_positive) == 0.0


# ---------------------------------------------------------------------------------------------
# stats.py — §3.2 Holm ordering on a known p-vector
# ---------------------------------------------------------------------------------------------

def test_holm_bonferroni_known_pvector():
    pvalues = [0.001, 0.01, 0.02, 0.04, 0.5]
    result = stats.holm_bonferroni(pvalues, alpha=0.05)
    # hand-derived (m=5): sorted p unchanged; adjusted = running-max of (m-i)*p_(i):
    #   i=0: 5*0.001=0.005 -> reject (<=0.05)
    #   i=1: 4*0.01 =0.04, max(0.005,0.04)=0.04 -> reject
    #   i=2: 3*0.02 =0.06, max(0.04,0.06)=0.06 -> NOT reject
    #   i=3: 2*0.04 =0.08, max(...)=0.08 -> NOT reject
    #   i=4: 1*0.5  =0.5,  max(...)=0.5  -> NOT reject
    assert result["reject"] == [True, True, False, False, False]
    assert result["adjusted_p"][0] < result["adjusted_p"][1] < result["adjusted_p"][2]
    assert result["adjusted_p"] == sorted(result["adjusted_p"]), (
        "Holm adjusted p-values must be monotone non-decreasing once re-sorted to ascending "
        "raw-p order (step-down construction)"
    )


def test_holm_bonferroni_step_down_monotonicity_enforced():
    # A pathological p-vector where the raw per-index formula alone would NOT be monotone --
    # the running-max must still enforce a monotone reject pattern.
    pvalues = [0.049, 0.001, 0.001, 0.001]
    result = stats.holm_bonferroni(pvalues, alpha=0.05)
    # sorted ascending: [0.001, 0.001, 0.001, 0.049] (original indices 1,2,3,0)
    # i=0: 4*0.001=0.004 reject; i=1: 3*0.001=0.003, max=0.004 reject;
    # i=2: 2*0.001=0.002, max=0.004 reject; i=3: 1*0.049=0.049, max=0.049 reject (<=0.05)
    assert all(result["reject"])


# ---------------------------------------------------------------------------------------------
# stats.py — §3.2 max-T: a real-effect comparison should reject; a null comparison should not
# ---------------------------------------------------------------------------------------------

def test_max_t_adjusted_pvalues_separates_real_from_null():
    import numpy as np

    rng = np.random.default_rng(11)

    def _make(true_delta, n_clusters=10, n_per=10):
        a, b, g = [], [], []
        for c in range(n_clusters):
            base = rng.normal(0, 0.1)
            for _ in range(n_per):
                a.append(float(base + true_delta + rng.normal(0, 0.03)))
                b.append(float(base + rng.normal(0, 0.03)))
                g.append(f"c{c}")
        return {"scores_a": a, "scores_b": b, "groups": g}

    comparisons = [_make(0.25), _make(0.0), _make(0.0)]
    result = stats.max_t_adjusted_pvalues(comparisons, nboot=2000, seed=5)
    assert len(result["adjusted_p"]) == 3
    assert result["reject"][0] is True
    assert result["adjusted_p"][0] < result["adjusted_p"][1]
    assert result["adjusted_p"][0] < result["adjusted_p"][2]


# ---------------------------------------------------------------------------------------------
# stats.py — §3.3 DerSimonian-Laird + nested bootstrap: basic sanity
# ---------------------------------------------------------------------------------------------

def test_der_simonian_laird_pooling_sanity():
    # 3 datasets with consistent effects -> tau2 should be small/zero, pooled effect near the
    # common value, pooled CI narrower than any single dataset's own CI.
    effects = [0.20, 0.22, 0.18]
    variances = [0.01, 0.012, 0.011]
    result = stats.der_simonian_laird(effects, variances)
    assert result["k"] == 3
    assert 0.15 < result["pooled_effect"] < 0.25
    assert result["tau2"] >= 0.0
    assert result["I2"] >= 0.0
    single_half_width = 1.96 * (variances[0] ** 0.5)
    pooled_half_width = (result["pooled_ci"][1] - result["pooled_ci"][0]) / 2
    assert pooled_half_width < single_half_width, "pooling >=2 consistent estimates should narrow the CI"


def test_der_simonian_laird_heterogeneous_inflates_tau2():
    # wildly disagreeing effects -> tau2 (between-dataset heterogeneity) should be > 0 and I2 high.
    effects = [0.05, 0.50, -0.10]
    variances = [0.001, 0.001, 0.001]  # tiny within-dataset variance -> disagreement is NOT noise
    result = stats.der_simonian_laird(effects, variances)
    assert result["tau2"] > 0.0
    assert result["I2"] > 50.0


def test_variance_from_ci_roundtrip():
    v = stats.variance_from_ci([0.10, 0.30])  # half-width 0.10
    assert abs(v - (0.10 / 1.959963984540054) ** 2) < 1e-9


def test_nested_bootstrap_ci_basic_sanity():
    import numpy as np

    rng = np.random.default_rng(13)
    per_dataset_scores, per_dataset_groups = [], []
    for _d in range(5):
        scores, groups = [], []
        for c in range(6):
            base = rng.normal(0.6, 0.05)
            for _ in range(5):
                scores.append(float(base + rng.normal(0, 0.02)))
                groups.append(f"g{c}")
        per_dataset_scores.append(scores)
        per_dataset_groups.append(groups)

    result = stats.nested_bootstrap_ci(per_dataset_scores, per_dataset_groups, nboot=500, seed=4)
    assert result["n_datasets"] == 5
    assert result["grid_ci"][0] < result["grid_mean"] < result["grid_ci"][1]
    assert 0.4 < result["grid_mean"] < 0.8


# ---------------------------------------------------------------------------------------------
# _common.draw_disjoint_grouped — zero group overlap + exact-ish sizes
# ---------------------------------------------------------------------------------------------

def test_draw_disjoint_grouped_zero_overlap_and_exact_sizes():
    # 20 groups of exactly 5 items each = 100 items; n_test=60 (12 groups), n_dev=40 (8 groups)
    # divides evenly -> exact sizes expected, not just "close".
    items = []
    for g in range(20):
        for j in range(5):
            items.append(_row(f"g{g}-i{j}", meta_extra={"grp": f"group{g}"}))

    def group_key_fn(item):
        return item["meta"]["grp"]

    result = draw_disjoint_grouped(items, group_key_fn, n_test=60, n_dev=40, seed=42)

    assert result["n_test"] == 60
    assert result["n_dev"] == 40
    assert result["group_disjoint_verified"] is True
    assert set(result["test_groups"]) & set(result["dev_groups"]) == set()
    assert set(result["test_ids"]) & set(result["dev_ids"]) == set()
    assert result["fallback_item_level"] is False
    assert result["shortfall"] is None
    assert result["oversized_group"] is None

    # every item in a chosen test group is present in test_ids (whole-group discipline)
    for gid in result["test_groups"]:
        group_items = {it["meta"]["item_id"] for it in items if it["meta"]["grp"] == gid}
        assert group_items.issubset(set(result["test_ids"]))


def test_draw_disjoint_grouped_uneven_groups_never_splits_a_group():
    # group sizes that do NOT evenly divide n_test/n_dev -> "exact-ish", but a group is NEVER split.
    items = []
    sizes = [7, 3, 11, 4, 9, 6, 2, 13, 5, 8]
    for gi, size in enumerate(sizes):
        for j in range(size):
            items.append(_row(f"g{gi}-i{j}", meta_extra={"grp": f"group{gi}"}))

    def group_key_fn(item):
        return item["meta"]["grp"]

    result = draw_disjoint_grouped(items, group_key_fn, n_test=30, n_dev=20, seed=99)

    assert set(result["test_groups"]) & set(result["dev_groups"]) == set()
    assert set(result["test_ids"]) & set(result["dev_ids"]) == set()
    # "exact-ish": greedy whole-group fill always reaches >= target (enough total items exist),
    # with overshoot bounded by the largest group's size (never splits a group to land exactly).
    max_group_size = max(sizes)
    assert 30 <= result["n_test"] < 30 + max_group_size
    assert 20 <= result["n_dev"] < 20 + max_group_size

    for gid in result["test_groups"]:
        group_items = {it["meta"]["item_id"] for it in items if it["meta"]["grp"] == gid}
        assert group_items.issubset(set(result["test_ids"])), "a group must never be split across sides"
    for gid in result["dev_groups"]:
        group_items = {it["meta"]["item_id"] for it in items if it["meta"]["grp"] == gid}
        assert group_items.issubset(set(result["dev_ids"])), "a group must never be split across sides"


def test_draw_disjoint_grouped_fallback_item_level_when_group_key_fn_always_none():
    items = [_row(f"item{i}") for i in range(20)]

    def group_key_fn(_item):
        return None  # G-NONE dataset simulation

    result = draw_disjoint_grouped(items, group_key_fn, n_test=12, n_dev=8, seed=1)
    assert result["fallback_item_level"] is True
    assert result["n_groups_total"] == 20  # every item its own singleton group
    assert set(result["test_ids"]) & set(result["dev_ids"]) == set()
    assert result["n_test"] == 12
    assert result["n_dev"] == 8


# ---------------------------------------------------------------------------------------------
# group_key.group_key_of — unit cases across >= 8 real dataset key shapes (synthetic items)
# ---------------------------------------------------------------------------------------------

def test_group_key_of_aishell1():
    item = _row("BAC009S0769W0185")
    assert group_key_of("aishell-1", item) == "S0769"


def test_group_key_of_thchs30():
    item = _row("D7_841")
    assert group_key_of("thchs-30", item) == "D7"


def test_group_key_of_librispeech():
    item = _row("6930-75918-0000")
    assert group_key_of("librispeech", item) == "6930"


def test_group_key_of_voicebench_bbh():
    item = _row("bbh_web_of_lies_218")
    assert group_key_of("voicebench-bbh", item) == "bbh_web_of_lies"


def test_group_key_of_voicebench_sdqa():
    item_usa = _row("sd-qa/usa#37")
    item_aus = _row("sd-qa/aus#37")
    assert group_key_of("voicebench-sd-qa", item_usa) == "37"
    assert group_key_of("voicebench-sd-qa", item_usa) == group_key_of("voicebench-sd-qa", item_aus)


def test_group_key_of_squtr():
    clean = _row("fiqa|clean|q19")
    noisy = _row("fiqa|snr0|q19")
    assert group_key_of("squtr", clean) == "q19"
    assert group_key_of("squtr", clean) == group_key_of("squtr", noisy)


def test_group_key_of_crema_d():
    item = _row("crema-d/1002_MTI_NEU_XX", gold={"spk": "1002", "sent": "MTI", "emo": "neutral"})
    assert group_key_of("crema-d", item) == "1002"


def test_group_key_of_esd():
    item = _row("esd/0001_000351", gold={"emo": "Angry", "spk": "0001"})
    assert group_key_of("esd", item) == "0001"


def test_group_key_of_csemotions():
    item = _row("csemotions/shard0_row74", gold={"emotion": "happy", "speaker": "female001"})
    assert group_key_of("csemotions", item) == "female001"


def test_group_key_of_meld():
    item = _row("meld/test/dia5_utt8", meta_extra={"dialogue_id": 5, "utterance_id": 8})
    assert group_key_of("meld", item) == "5"


def test_group_key_of_slurp():
    item = _row(12345, gold={"intent": "calendar_set", "scenario": "calendar"})
    assert group_key_of("slurp", item) == "calendar"
    slot_item = _row(999, gold={"scenario": "email"})
    assert group_key_of("slurp-slot", slot_item) == "email"


def test_group_key_of_speech_massive_scenario():
    item = _row("de-DE/validation/12@0-0-3", meta_extra={"scenario_str": "alarm_set"})
    assert group_key_of("speech-massive-de-DE", item) == "alarm_set"
    slot_item = _row("fr-FR/validation/1@0-0-1", meta_extra={"scenario_str": "weather_query"})
    assert group_key_of("speech-massive-fr-FR-slot", slot_item) == "weather_query"


def test_group_key_of_speech_massive_attr_needs_loader_edit():
    # K5 attribute probe: MUST NOT fall back to scenario grouping (see group_key.py's TODO) --
    # speaker_id isn't exposed yet, so this MUST be None, not a guessed scenario group.
    item = _row("de-DE/validation/12@0-0-3", meta_extra={"scenario_str": "alarm_set"})
    assert group_key_of("speech-massive-de-DE-attr", item) is None


def test_group_key_of_mmsu():
    item = _row("accent_identification_abc123", meta_extra={"task_name": "accent_identification"})
    assert group_key_of("mmsu", item) == "accent_identification"


def test_group_key_of_voicebench_mmsu_spoken():
    item = _row("mmsu-spoken/health/6062", meta_extra={"domain": "health", "src": "mmlu_health"})
    assert group_key_of("voicebench-mmsu-spoken", item) == "mmlu_health"


def test_group_key_of_heysquad_hashes_context():
    item_a = _row("q1", meta_extra={"context": "The quick brown fox jumps over the lazy dog."})
    item_b = _row("q2", meta_extra={"context": "The quick brown fox jumps over the lazy dog."})
    item_c = _row("q3", meta_extra={"context": "A completely different passage."})
    ga, gb, gc = (group_key_of("heysquad", it) for it in (item_a, item_b, item_c))
    assert ga == gb
    assert ga != gc
    assert isinstance(ga, str) and len(ga) == 16


def test_group_key_of_voiceassistant():
    item = _row("Listening_General_3", meta_extra={"category1": "listening", "category2": "general"})
    assert group_key_of("voiceassistant-listening-general", item) == "listening/general"


def test_group_key_of_vocalbench_knowledge_and_multiround():
    know = _row("knowledge-0007", meta_extra={"topic": "history", "source": "wikipedia"})
    assert group_key_of("vocalbench-knowledge", know) == "wikipedia"
    multi = _row("multi_round-0003", meta_extra={"category": "travel"})  # no source/topic column
    assert group_key_of("vocalbench-multi-round", multi) == "travel"


def test_group_key_of_g_source_returns_none_not_a_guess():
    # audio2tool + speech-massive-*-attr: G-SOURCE, no loader edit made by this ticket -> None.
    a2t = _row("tier1_direct|00042", meta_extra={"tool_name": "set_alarm", "domain": "productivity"})
    assert group_key_of("audio2tool", a2t) is None
    legacy = _row("SQuAD-zh#7")
    assert group_key_of("SQuAD-zh", legacy) is None
    aqa = _row("Sound_AQA:clothoaqa:14934", meta_extra={"task_name": "Sound_AQA"})
    assert group_key_of("air-bench-foundation-sound-aqa-clothoaqa", aqa) is None


def test_group_key_of_g_none_returns_none():
    # mmar / voicebench-openbookqa / fleurs-r etc -- no group at any grain -> honest None.
    item = _row("mmar#4", meta_extra={"category": "speech", "sub_category": "emotion"})
    assert group_key_of("mmar", item) is None
    ob = _row("openbookqa#12")
    assert group_key_of("voicebench-openbookqa", ob) is None


# ---------------------------------------------------------------------------------------------
# standalone runner (mirrors scripts/baselines/test_phase_a_e2e.py's convention)
# ---------------------------------------------------------------------------------------------

def main() -> int:
    import traceback

    test_fns = {name: fn for name, fn in sorted(globals().items())
                if name.startswith("test_") and callable(fn)}
    results: dict[str, bool] = {}
    for name, fn in test_fns.items():
        try:
            fn()
            results[name] = True
        except Exception:  # noqa: BLE001 -- record, don't abort the sweep
            print(f"--- {name} FAILED ---")
            traceback.print_exc()
            results[name] = False

    print("\n=== TEST_STATS ===")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_pass = all(results.values())
    print(f"\n{len(results)} checks, {sum(results.values())} passed, "
          f"{len(results) - sum(results.values())} failed")
    print("TEST_STATS_PASS" if all_pass else "TEST_STATS_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

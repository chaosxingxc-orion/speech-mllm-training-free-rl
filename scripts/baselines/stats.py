"""scripts/baselines/stats.py — statistical core for the group-split + cluster-bootstrap redesign
(ticket #26, QUESTION-INDEPENDENT machinery only).

Design doc: wiki/2026-07-11-group-split-statistics-design.md §3 (Task 3) — every function here
implements one piece of that section, cited in its own docstring. Status (2026-07-11 update,
ticket #26 implementation): ``run_baseline.run_one`` still calls its own item-level
``paired_bootstrap`` for the single-arm aggregate block (unchanged — that's the per-cell CI every
existing wave-1/2 consumer expects); this module is now wired into
``scripts/baselines/summarize_locked.py`` (a NEW v2 summarizer, alongside the untouched
``summarize_wave1.py``) for the paired qwen3-vs-meralion cluster-bootstrap Δ + Holm-within-family
view, and into ``scripts/baselines/locked_split.py`` (which calls ``_common.draw_disjoint_grouped``,
not this module directly, but shares its design-doc lineage). Every default seed below stays a
plain, caller-overridable ``0`` — ``locked_split.py`` owns the actual ``LOCKED_TEST_SEED`` constant
and passes its own seed explicitly to any of these functions it calls; this module itself still
does not hardcode or import that constant.

Pure numpy, no scipy/statsmodels (preserves the lazy-import + dependency-light discipline,
CLAUDE.md) — every heavy import (``numpy``) stays inside the function body, not at module top.

Reference implementation this generalizes (design doc §3, opening paragraph): the resample loop in
``run_baseline.paired_bootstrap`` (item-level, i.i.d.) and ``scripts/p6_perception_delta.py:boot``
— both resample INDIVIDUAL ITEMS; every function below resamples GROUPS/CLUSTERS instead (falling
back to item-level, explicitly flagged, when no group metadata exists — see ``group_key_of`` in
``scripts/loaders/group_key.py`` for how a group id is derived per dataset).
"""
from __future__ import annotations

from typing import Callable, Sequence

_Z_975 = 1.959963984540054  # standard-normal 97.5th-percentile quantile (95% CI half-width factor)


# ---------------------------------------------------------------------------------------------
# shared internals
# ---------------------------------------------------------------------------------------------

def _bucket_by_group(scores: Sequence, groups: Sequence) -> dict:
    """``[(score, group)] with score not None`` -> ``{group_id: np.ndarray(scores)}``.

    A ``None`` entry in ``groups`` (no group metadata for that ONE item — the G-NONE / fallback
    case, design doc §1.4) makes that item its OWN singleton cluster, keyed by its position so two
    different ungrouped items never accidentally collide into the same bucket.
    """
    import numpy as np

    buckets: dict = {}
    for i, (s, g) in enumerate(zip(scores, groups)):
        if s is None:
            continue
        key = g if g is not None else f"__singleton_{i}__"
        buckets.setdefault(key, []).append(float(s))
    return {k: np.array(v, dtype=float) for k, v in buckets.items()}


def _paired_cluster_bootstrap_deltas(scores_a, scores_b, groups, nboot: int, seed: int):
    """Raw paired-cluster bootstrap Δ replicates — the ONE resampling loop shared by
    ``paired_cluster_delta_ci`` and ``max_t_adjusted_pvalues`` (design doc §3.1/§3.2), so both
    callers' resampling semantics can never silently drift apart.

    Returns ``(deltas: np.ndarray[nboot], point: float | None, n_clusters: int, n_items: int,
    bootstrap_unit: "cluster" | "item")``. ``point`` is the OBSERVED (non-resampled) Δ = mean(a) -
    mean(b) over the paired items; ``None`` iff there are zero paired-scored items (both arrays
    then empty).

    Pairing contract: ``scores_a``/``scores_b``/``groups`` are aligned lists — same length, one
    entry per item, SAME item order across all three (e.g. one baseline arm and one training-free-
    RL arm scored over the identical eval item set). An item scored ``None`` in EITHER arm is
    dropped from BOTH (a paired comparison needs both arms' scores for the exact same item).
    """
    import numpy as np

    a, b, g = [], [], []
    for sa, sb, gi in zip(scores_a, scores_b, groups):
        if sa is None or sb is None:
            continue
        a.append(float(sa))
        b.append(float(sb))
        g.append(gi)

    if not a:
        return np.array([]), None, 0, 0, "cluster"

    idx_by_group: dict = {}
    for i, gi in enumerate(g):
        key = gi if gi is not None else f"__singleton_{i}__"
        idx_by_group.setdefault(key, []).append(i)
    clusters = list(idx_by_group.values())
    n_clusters = len(clusters)
    n_items = len(a)
    unit = "item" if n_clusters == n_items else "cluster"

    a_arr, b_arr = np.array(a), np.array(b)
    point = float(a_arr.mean() - b_arr.mean())

    rng = np.random.default_rng(seed)
    deltas = np.empty(nboot, dtype=float)
    for it in range(nboot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idxs = np.concatenate([clusters[j] for j in pick])
        deltas[it] = a_arr[idxs].mean() - b_arr[idxs].mean()

    return deltas, point, n_clusters, n_items, unit


_FALLBACK_CAVEAT = (
    "no grouping metadata (or every item is its own group) -- bootstrap resampled ITEMS, not "
    "clusters; intra-group correlation is not controlled (design doc §1.4 honest fallback)."
)


# ---------------------------------------------------------------------------------------------
# §3.1 — paired cluster bootstrap (replaces run_baseline.paired_bootstrap)
# ---------------------------------------------------------------------------------------------

def cluster_bootstrap_ci(scores: Sequence, groups: Sequence | None = None,
                          nboot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """95% (or ``1-alpha``) CI on the mean of ``scores``, resampling GROUPS (not items) with
    replacement (design doc §3.1).

    Resample unit = cluster: draw K clusters (K = number of clusters) WITH replacement, pool their
    member items, take the mean; repeat ``nboot`` times and report the empirical
    ``[alpha/2, 1-alpha/2]`` quantiles. This propagates intra-cluster correlation into the CI width
    — the whole point of audit finding #3 (i.i.d. item resampling underestimates variance under
    same-speaker/same-clip/same-passage correlation, giving falsely narrow "significant" CIs).

    ``groups``: list aligned to ``scores`` (item -> group id), or ``None``. When ``groups`` is
    ``None``, or every group id is effectively unique (no dataset has >1 scored item per group),
    this DEGENERATES to item-level bootstrap — and the returned dict flags
    ``bootstrap_unit: "item"`` (never silently reports a cluster-shaped result for a dataset that
    has no real clusters; design doc's "Fallback flagged" rule).

    Returns:
        {"ci": [lo, hi] | None, "mean": float | None, "n_clusters": int, "n_items": int,
         "bootstrap_unit": "cluster" | "item", "caveat": str | None}
    """
    import numpy as np

    if groups is None:
        groups = [None] * len(scores)
    buckets = _bucket_by_group(scores, groups)
    clusters = list(buckets.values())
    n_clusters = len(clusters)
    n_items = sum(len(c) for c in clusters)
    if n_clusters == 0:
        return {"ci": None, "mean": None, "n_clusters": 0, "n_items": 0,
                "bootstrap_unit": "cluster", "caveat": "no scored items"}

    unit = "item" if n_clusters == n_items else "cluster"
    rng = np.random.default_rng(seed)
    means = np.empty(nboot, dtype=float)
    for b in range(nboot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        pooled = np.concatenate([clusters[j] for j in pick])
        means[b] = pooled.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    all_scores = np.concatenate(clusters)
    caveat = None if unit == "cluster" else _FALLBACK_CAVEAT
    return {"ci": [round(float(lo), 4), round(float(hi), 4)],
            "mean": round(float(all_scores.mean()), 4),
            "n_clusters": n_clusters, "n_items": int(n_items),
            "bootstrap_unit": unit, "caveat": caveat}


def paired_cluster_delta_ci(scores_a: Sequence, scores_b: Sequence, groups: Sequence,
                             nboot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """95% CI on Δ = mean(scores_a) - mean(scores_b), using the SAME cluster resample indices for
    BOTH arms in each bootstrap iteration (design doc §3.1: "Paired ... by using the same cluster
    resample indices for both arms ... then recording the difference of means — a CI on Δ that
    cancels the shared cluster-draw noise (this is the actual 'paired' part the old name only
    gestured at)").

    See ``_paired_cluster_bootstrap_deltas`` for the pairing contract (items dropped if either arm
    scored ``None``).

    Returns:
        {"delta_ci": [lo, hi] | None, "delta_mean": float | None, "pvalue": float | None,
         "n_clusters": int, "n_items": int, "bootstrap_unit": "cluster" | "item",
         "caveat": str | None}

    ``pvalue`` (added 2026-07-11, ticket #26 summarizer wiring -- design doc §3.2's "compute a
    per-comparison bootstrap p-value" input) is ``bootstrap_pvalue(deltas)`` over this SAME call's
    resample replicates -- exposed here so a caller building a family of comparisons for
    ``holm_bonferroni`` doesn't need to reach into the private ``_paired_cluster_bootstrap_deltas``
    itself; purely additive, does not change any existing key's meaning.
    """
    import numpy as np

    deltas, point, n_clusters, n_items, unit = _paired_cluster_bootstrap_deltas(
        scores_a, scores_b, groups, nboot, seed)
    if point is None:
        return {"delta_ci": None, "delta_mean": None, "pvalue": None, "n_clusters": 0, "n_items": 0,
                "bootstrap_unit": "cluster", "caveat": "no paired-scored items"}
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    caveat = None if unit == "cluster" else _FALLBACK_CAVEAT
    return {"delta_ci": [round(float(lo), 4), round(float(hi), 4)],
            "delta_mean": round(point, 4), "pvalue": bootstrap_pvalue(deltas),
            "n_clusters": n_clusters, "n_items": n_items,
            "bootstrap_unit": unit, "caveat": caveat}


def bootstrap_pvalue(deltas: Sequence) -> float:
    """Two-sided bootstrap p-value for H0: Δ=0 from a Δ bootstrap-replicate array (design doc
    §3.2): ``p = 2 * min(fraction of replicates <= 0, fraction of replicates >= 0)``, capped at
    1.0. Feed it ``_paired_cluster_bootstrap_deltas(...)[0]`` / the ``deltas`` array underlying
    ``paired_cluster_delta_ci`` for one comparison.
    """
    import numpy as np

    d = np.asarray(deltas, dtype=float)
    if len(d) == 0:
        return 1.0
    frac_le = float(np.mean(d <= 0))
    frac_ge = float(np.mean(d >= 0))
    return float(min(1.0, 2 * min(frac_le, frac_ge)))


# ---------------------------------------------------------------------------------------------
# §3.2 — within-arm-family multiplicity: Holm / max-T
# ---------------------------------------------------------------------------------------------

def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down family-wise error control (design doc §3.2) — distribution-free,
    pure numpy, no statsmodels dependency.

    Standard step-down procedure: sort p-values ascending; reject ``p_(i)`` (1-indexed ``i``)
    while ``p_(i) <= alpha / (m - i + 1)``; once one hypothesis fails to reject, every LARGER
    p-value is also not rejected (step-down monotonicity, enforced explicitly here so a
    non-monotonic raw comparison can never yield a non-monotonic reject pattern). Reports the
    standard Holm ADJUSTED p-value too: ``adj_p_(i) = max_{j<=i} min(1, (m-j+1) * p_(j))``.

    Returns (all lists in the ORIGINAL, caller-given order):
        {"adjusted_p": [...], "reject": [...] (bool), "alpha": alpha}
    """
    import numpy as np

    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return {"adjusted_p": [], "reject": [], "alpha": alpha}

    order = np.argsort(p)  # ascending
    sorted_p = p[order]

    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        val = min(1.0, (m - i) * sorted_p[i])
        running_max = max(running_max, val)
        adj_sorted[i] = running_max

    reject_sorted = adj_sorted <= alpha
    for i in range(1, m):  # step-down monotonicity: one failed rejection stops all later ones
        if not reject_sorted[i - 1]:
            reject_sorted[i] = False

    adjusted_p = np.empty(m, dtype=float)
    reject = np.empty(m, dtype=bool)
    adjusted_p[order] = adj_sorted
    reject[order] = reject_sorted

    return {"adjusted_p": [round(float(x), 6) for x in adjusted_p],
            "reject": [bool(x) for x in reject], "alpha": alpha}


def max_t_adjusted_pvalues(comparisons: list, nboot: int = 10000, seed: int = 0,
                            alpha: float = 0.05) -> dict:
    """Bootstrap step-down max-T multiplicity control (design doc §3.2) for a family of paired
    comparisons — tighter than Holm when the family's tests are positively correlated (e.g. the
    same cluster recurring across several comparisons, "multi-accent sd-qa" per the design doc),
    because it uses the family's EMPIRICAL joint null distribution of the maximum studentized
    statistic rather than a fixed, independence-agnostic Bonferroni-style correction.

    For each comparison ``c`` (``{"scores_a", "scores_b", "groups"}``, same shape as
    ``paired_cluster_delta_ci``'s first three args), this computes ``nboot`` paired-cluster
    bootstrap Δ replicates (one resampling loop per comparison, design doc: "implemented as one
    extra reduction inside the existing cluster-bootstrap loop — negligible cost"), studentizes
    them against their OWN bootstrap standard deviation (``z_c_b = (Δ_c_b - Δ̂_c) / se_c``), and
    tracks the family MAXIMUM ``|z|`` at each iteration — the resulting empirical distribution of
    the max is the max-T null; a comparison's raw p-value is the fraction of that null exceeding
    its OWN observed ``|Δ̂_c| / se_c``. Step-down enforcement (Westfall-Young): comparisons are
    ranked by observed ``|z|`` descending, and each rank's adjusted p is forced to be
    non-decreasing as observed effect size shrinks (a smaller-effect comparison can never get a
    SMALLER adjusted p than a larger-effect one ahead of it).

    Passing comparisons whose ``groups`` are drawn from a SHARED cluster universe recovers the
    "uses the empirical correlation" benefit; comparisons over disjoint cluster universes are still
    valid (degrades gracefully to an independence-agnostic bound), just without the extra power.

    Returns:
        {"observed_delta": [...], "observed_z": [...], "adjusted_p": [...], "reject": [...],
         "alpha": alpha} — all lists in ``comparisons``' given order.
    """
    import numpy as np

    m = len(comparisons)
    if m == 0:
        return {"observed_delta": [], "observed_z": [], "adjusted_p": [], "reject": [], "alpha": alpha}

    deltas_by_c, point_by_c = [], []
    for c in comparisons:
        deltas, point, _n_clusters, _n_items, _unit = _paired_cluster_bootstrap_deltas(
            c["scores_a"], c["scores_b"], c["groups"], nboot, seed)
        deltas_by_c.append(deltas)
        point_by_c.append(point if point is not None else 0.0)

    se_by_c = []
    for d in deltas_by_c:
        se = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        se_by_c.append(se if se > 0 else 1e-12)

    z_matrix = np.stack(
        [(deltas_by_c[c] - point_by_c[c]) / se_by_c[c] for c in range(m)], axis=0)  # (m, nboot)
    max_abs_z = np.max(np.abs(z_matrix), axis=0)  # (nboot,) -- the family's max-T null distribution

    observed_z = np.array([point_by_c[c] / se_by_c[c] for c in range(m)])
    raw_adjusted_p = np.array([float(np.mean(max_abs_z >= abs(oz))) for oz in observed_z])

    order = np.argsort(-np.abs(observed_z))  # rank by observed effect size, descending
    running_max = 0.0
    adj_stepdown = np.empty(m)
    for ci in order:
        running_max = max(running_max, raw_adjusted_p[ci])
        adj_stepdown[ci] = running_max

    reject = adj_stepdown <= alpha
    return {
        "observed_delta": [round(float(x), 4) for x in point_by_c],
        "observed_z": [round(float(x), 4) for x in observed_z],
        "adjusted_p": [round(float(x), 6) for x in adj_stepdown],
        "reject": [bool(x) for x in reject],
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------------------------
# §3.3 — hierarchical (random-effects) cross-dataset aggregation
# ---------------------------------------------------------------------------------------------

def variance_from_ci(ci: Sequence[float], z: float = _Z_975) -> float:
    """``v = ((hi - lo) / (2*z))^2`` — converts a (95%, by default) CI's half-width back to an
    approximate normal variance (design doc §3.3's stated input contract for
    ``der_simonian_laird``: "its cluster-bootstrap variance v_k (square of half-CI-width / 1.96)").
    Pass a different ``z`` for a non-95% CI.
    """
    lo, hi = ci
    half_width = (hi - lo) / 2.0
    return (half_width / z) ** 2


def der_simonian_laird(effects: Sequence[float], variances: Sequence[float]) -> dict:
    """Two-level random-effects meta-analysis (DerSimonian-Laird), pure numpy (design doc §3.3).

    Inputs: per-dataset (or per-cell) effect estimates Δ̂_k and their (cluster-)bootstrap variances
    v_k (see ``variance_from_ci``). Estimates between-dataset heterogeneity τ² via the DL
    method-of-moments estimator, then reports the inverse-variance-weighted pooled effect with
    weights ``w_k = 1/(v_k + τ²)`` and its CI (``1/Σw_k`` gives the pooled variance). Also reports
    τ² and I² (percentage of total variation attributable to between-dataset heterogeneity, not
    sampling error) — design doc's explicit ask: "so a high-variance grid isn't summarized as a
    single deceptively-tight number".

    Returns:
        {"pooled_effect", "pooled_ci", "pooled_se", "tau2", "I2", "Q", "k"}
    """
    import numpy as np

    d = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(d)
    if k == 0:
        return {"pooled_effect": None, "pooled_ci": None, "pooled_se": None,
                "tau2": None, "I2": None, "Q": None, "k": 0}
    if k == 1:
        se = float(np.sqrt(max(v[0], 0.0)))
        return {"pooled_effect": round(float(d[0]), 4),
                "pooled_ci": [round(float(d[0] - _Z_975 * se), 4), round(float(d[0] + _Z_975 * se), 4)],
                "pooled_se": round(se, 4), "tau2": 0.0, "I2": 0.0, "Q": 0.0, "k": 1}

    v_safe = np.where(v <= 0, 1e-12, v)
    w_fixed = 1.0 / v_safe
    effect_fixed = float(np.sum(w_fixed * d) / np.sum(w_fixed))
    Q = float(np.sum(w_fixed * (d - effect_fixed) ** 2))
    df = k - 1
    C = float(np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0

    w_re = 1.0 / (v_safe + tau2)
    pooled = float(np.sum(w_re * d) / np.sum(w_re))
    pooled_var = 1.0 / np.sum(w_re)
    pooled_se = float(np.sqrt(pooled_var))
    return {
        "pooled_effect": round(pooled, 4),
        "pooled_ci": [round(pooled - _Z_975 * pooled_se, 4), round(pooled + _Z_975 * pooled_se, 4)],
        "pooled_se": round(pooled_se, 4),
        "tau2": round(tau2, 6), "I2": round(I2, 2), "Q": round(Q, 4), "k": k,
    }


def nested_bootstrap_ci(per_dataset_scores: list, per_dataset_groups: list,
                         nboot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Fully nonparametric grid-level CI via a nested (dataset-of-clusters) bootstrap — design doc
    §3.3's robustness-check alternative to ``der_simonian_laird`` ("more faithful for the coarse-
    group SER cells" where a per-dataset variance estimate from only a handful of clusters, e.g.
    esd/csemotions' 10 speakers, is unreliable for the DL normal-approximation).

    Outer loop resamples DATASETS with replacement (K = number of datasets); inner loop resamples
    CLUSTERS within each outer-drawn dataset (also with replacement) and takes that dataset's mean;
    one outer iteration's grid-level statistic is the mean of those per-dataset means. Repeating
    ``nboot`` times gives a CI needing no per-dataset normal/variance assumption (unlike DL).

    ``per_dataset_scores``/``per_dataset_groups``: one list of item scores / item group-ids PER
    dataset (``per_dataset_scores[k]`` aligned with ``per_dataset_groups[k]``). A dataset with zero
    scored items is dropped (never enters the outer draw).

    Returns:
        {"grid_ci": [lo, hi] | None, "grid_mean": float | None, "n_datasets": int}
    """
    import numpy as np

    per_dataset_clusters = []
    for scores, groups in zip(per_dataset_scores, per_dataset_groups):
        g = groups if groups is not None else [None] * len(scores)
        buckets = _bucket_by_group(scores, g)
        clusters = list(buckets.values())
        if clusters:
            per_dataset_clusters.append(clusters)

    n_datasets = len(per_dataset_clusters)
    if n_datasets == 0:
        return {"grid_ci": None, "grid_mean": None, "n_datasets": 0}

    rng = np.random.default_rng(seed)
    outer_means = np.empty(nboot, dtype=float)
    for o in range(nboot):
        ds_pick = rng.integers(0, n_datasets, size=n_datasets)
        ds_means = np.empty(n_datasets, dtype=float)
        for j, dsi in enumerate(ds_pick):
            clusters = per_dataset_clusters[dsi]
            kc = len(clusters)
            cl_pick = rng.integers(0, kc, size=kc)
            pooled = np.concatenate([clusters[c] for c in cl_pick])
            ds_means[j] = pooled.mean()
        outer_means[o] = ds_means.mean()

    lo, hi = np.quantile(outer_means, [alpha / 2, 1 - alpha / 2])
    grid_mean = float(np.mean([np.concatenate(cl).mean() for cl in per_dataset_clusters]))
    return {"grid_ci": [round(float(lo), 4), round(float(hi), 4)],
            "grid_mean": round(grid_mean, 4), "n_datasets": n_datasets}

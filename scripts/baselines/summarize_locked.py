"""scripts/baselines/summarize_locked.py — v2 summarizer: paired cluster-bootstrap Δ
(qwen3-omni-30b-gguf vs meralion-2-gguf) per (dataset, split) result cell, Holm-Bonferroni
corrected WITHIN each K-type arm family (ticket #26, design doc
wiki/2026-07-11-group-split-statistics-design.md §3.1/§3.2).

## Why a separate module, not an in-place edit of summarize_wave1.py

``summarize_wave1.py`` reports one number PER (dataset, backbone, split) cell (a single arm's
mean ± CI) -- it has no "compare two arms" concept at all. Wiring cluster-bootstrap Δ into it
would mean bolting an unrelated table shape onto an already-complete report; a new module keeps
that report untouched (still exactly what it was) and adds the delta-comparison view alongside it.
Reuses ``summarize_wave1``'s ``group_of``/``_group_sort_key`` (the K-type bucketing) rather than
re-deriving that logic.

## What "arm family" means here

The two backbones (``qwen3-omni-30b-gguf``, ``meralion-2-gguf``) are this repo's only two
generation arms today (owner ruling 2026-07-09, 双底座定稿) -- so "the Δ between the two arms" is
the ONE comparison per (dataset, split) cell. "Family" for the Holm correction (design doc §3.2:
"When an arm family compares many datasets/subsets at once") is therefore taken as (K-type group,
split): every dataset key sharing a K-type is one family of simultaneous comparisons, dev and test
kept as separate families (different sample draws, not repeated tests of the same hypothesis).

## group_id sourcing (per_item, when present -- ticket #26 task 4's explicit ask)

Per-item ``group_id`` is ONLY persisted by ``run_baseline._run_item`` for cells generated AFTER
ticket #26's core commit (33ff3c1) -- verified this session: every pre-existing
``_repro/baselines/*.json`` cell lacks it (e.g. ``crema-d``'s archived per_item entries are
``{item_id, instr, reply, score, detail}``, no ``group_id``). This module reads
``per_item[i]["group_id"]`` directly when present and falls back to item-level (``None``,
``stats.py`` flags it in ``bootstrap_unit``/``caveat``) when absent -- it does NOT attempt to
re-derive a group post-hoc from the archived JSON (that's only possible for G-ID datasets, which
parse item_id alone -- design doc §1.2 -- and would silently mix "real recovered group" with
"honest fallback" in a way the ticket's "when present" wording does not ask for).

Usage:
  python summarize_locked.py                          # stdout + _repro/locked_cluster_delta_report.md
  python summarize_locked.py --dir /other/baselines --out /tmp/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/baselines

import stats                  # noqa: E402
import summarize_wave1 as sw1  # noqa: E402  (reuse group_of / _group_sort_key, not re-derived)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINES_DIR = REPO_ROOT / "_repro" / "baselines"
DEFAULT_OUT = REPO_ROOT / "_repro" / "locked_cluster_delta_report.md"

BACKBONE_A = "qwen3-omni-30b-gguf"
BACKBONE_B = "meralion-2-gguf"
NBOOT = 10000
ALPHA = 0.05


# ---------------------------------------------------------------------------------------------
# loading + pairing
# ---------------------------------------------------------------------------------------------

def _load_cell(dataset: str, backbone: str, split: str, baselines_dir: Path) -> dict | None:
    p = baselines_dir / f"{dataset}__{backbone}__{split}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def paired_arrays(cell_a: dict, cell_b: dict) -> tuple[list, list, list, bool]:
    """Align two cells' per_item lists by item_id; return (scores_a, scores_b, groups,
    any_group_present). Only items with a NUMERIC score in BOTH arms are kept (mirrors
    ``stats._paired_cluster_bootstrap_deltas``'s own pairing contract -- this is upstream
    deduplication by item_id first, since the two arms' per_item lists aren't guaranteed to be in
    the same order). ``groups[i]`` is that item's ``per_item["group_id"]`` when EITHER arm's
    archived cell has it (both arms share the same item-id set by construction, so it should
    coincide when present at all -- see redraw.py's "双底座同 item-id 集零错配" note), else
    ``None`` -- the honest item-level-fallback signal ``stats.py`` already understands."""
    by_a = {it["item_id"]: it for it in cell_a.get("per_item", []) if it.get("item_id") is not None}
    by_b = {it["item_id"]: it for it in cell_b.get("per_item", []) if it.get("item_id") is not None}
    scores_a, scores_b, groups = [], [], []
    any_group = False
    for iid in sorted(set(by_a) & set(by_b)):
        ia, ib = by_a[iid], by_b[iid]
        sa, sb = ia.get("score"), ib.get("score")
        if not isinstance(sa, (int, float)) or not isinstance(sb, (int, float)):
            continue
        g = ia.get("group_id")
        if g is None:
            g = ib.get("group_id")
        if g is not None:
            any_group = True
        scores_a.append(sa)
        scores_b.append(sb)
        groups.append(g)
    return scores_a, scores_b, groups, any_group


# ---------------------------------------------------------------------------------------------
# comparisons + Holm
# ---------------------------------------------------------------------------------------------

def compute_comparisons(baselines_dir: Path) -> list[dict]:
    """One entry per (dataset, split) where BOTH backbones have a parseable, scoreable cell --
    computed via ``stats.paired_cluster_delta_ci`` (design doc §3.1). Never raises: an unreadable
    cell is silently skipped (mirrors ``summarize_wave1.load_cells``'s tolerate-a-live-sweep
    discipline -- this script must never crash mid-sweep either)."""
    comparisons: list[dict] = []
    if not baselines_dir.is_dir():
        return comparisons

    dataset_splits = set()
    for p in sorted(baselines_dir.glob("*.json")):
        parts = p.stem.split("__")
        if len(parts) != 3:
            continue
        dk, bb, sp = parts
        if bb in (BACKBONE_A, BACKBONE_B):
            dataset_splits.add((dk, sp))

    for dk, sp in sorted(dataset_splits):
        cell_a = _load_cell(dk, BACKBONE_A, sp, baselines_dir)
        cell_b = _load_cell(dk, BACKBONE_B, sp, baselines_dir)
        if cell_a is None or cell_b is None:
            continue
        scores_a, scores_b, groups, any_group = paired_arrays(cell_a, cell_b)
        if not scores_a:
            continue
        result = stats.paired_cluster_delta_ci(scores_a, scores_b, groups, nboot=NBOOT, seed=0)
        if result["delta_mean"] is None:
            continue
        comparisons.append({
            "dataset": dk, "split": sp, "group": sw1.group_of(dk),
            "n_paired": len(scores_a), "group_id_present": any_group,
            **result,
        })
    return comparisons


def apply_holm_within_families(comparisons: list[dict]) -> None:
    """In-place: Holm-Bonferroni family-wise error control (design doc §3.2), one family per
    (K-type group, split) -- see module docstring's "What 'arm family' means here". Adds
    "holm_adjusted_p"/"holm_reject" keys to every comparison (``None`` only if the whole list is
    empty, which can't happen per-comparison since every comparison here has a real p-value)."""
    families: dict[tuple, list[int]] = defaultdict(list)
    for i, c in enumerate(comparisons):
        families[(c["group"], c["split"])].append(i)

    for _key, idxs in families.items():
        pvalues = [comparisons[i]["pvalue"] for i in idxs]
        holm = stats.holm_bonferroni(pvalues, alpha=ALPHA)
        for j, i in enumerate(idxs):
            comparisons[i]["holm_adjusted_p"] = holm["adjusted_p"][j]
            comparisons[i]["holm_reject"] = holm["reject"][j]


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------

def render_report(comparisons: list[dict]) -> str:
    lines = [
        "# Locked cluster-bootstrap Δ report (qwen3-omni-30b-gguf vs meralion-2-gguf)",
        "",
        "Stage-1 directional numbers only (CLAUDE.md discipline) — purely factual, no "
        "interpretation. Δ = mean(qwen3) − mean(meralion), PAIRED cluster bootstrap (design doc "
        "`wiki/2026-07-11-group-split-statistics-design.md` §3.1) resampling GROUPS (`per_item` "
        "`\"group_id\"`, persisted only for cells generated after ticket #26's core commit) with "
        "replacement — falls back to item-level (flagged `bootstrap_unit=\"item\"` + a `caveat`) "
        "when neither arm's archived cell carries `group_id`. Holm-Bonferroni family-wise error "
        "control applied WITHIN each (K-type group, split) family (design doc §3.2) — see "
        "`holm_adjusted_p`/`holm_reject`.",
        "",
        "| dataset | split | group | n_paired | bootstrap_unit | delta_mean | delta_ci95 | "
        "pvalue | holm_adjusted_p | holm_reject | caveat |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(comparisons, key=lambda c: (sw1._group_sort_key(c["group"]), c["dataset"], c["split"])):
        ci = c["delta_ci"]
        ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"
        lines.append(
            f"| {c['dataset']} | {c['split']} | {c['group']} | {c['n_paired']} | "
            f"{c['bootstrap_unit']} | {c['delta_mean']:+.4f} | {ci_s} | {c['pvalue']:.4f} | "
            f"{c['holm_adjusted_p']:.4f} | {c['holm_reject']} | {c['caveat'] or '—'} |"
        )
    lines.append("")

    n_cluster = sum(1 for c in comparisons if c["bootstrap_unit"] == "cluster")
    n_item = sum(1 for c in comparisons if c["bootstrap_unit"] == "item")
    n_reject = sum(1 for c in comparisons if c["holm_reject"])
    lines.append(f"{len(comparisons)} paired (dataset, split) comparisons: {n_cluster} "
                 f"cluster-level bootstrap, {n_item} item-level fallback (flagged, no group_id on "
                 f"either arm's archived cell — see module docstring). {n_reject} reject "
                 f"H0:Δ=0 after Holm correction (α={ALPHA}).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_BASELINES_DIR,
                     help=f"directory of *.json result cells to scan (default: {DEFAULT_BASELINES_DIR})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                     help=f"markdown report output path (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    comparisons = compute_comparisons(args.dir)
    apply_holm_within_families(comparisons)
    report = render_report(comparisons)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    sys.stdout.write(report)
    print(f"\n[summarize_locked] wrote {args.out} ({len(comparisons)} comparisons)", file=sys.stderr)


if __name__ == "__main__":
    main()

"""scripts/baselines/summarize_wave1.py — wave-1 result aggregation + a GitHub-markdown report.

Scans ``_repro/baselines/*.json`` (one JSON per (dataset, backbone, split) cell, written by
``run_baseline.run_one``/``write_result`` via ``wave1_cells.py``'s wave-1 sweep) and emits a
purely factual GitHub-markdown report: one table per K-type group (K1, K2, ..., "legacy" for the
``scripts/p2_baselines.py``-native ``templates.LEGACY_DATASETS`` loaders), rows = dataset keys
sorted, columns = backbone x split, plus a coverage summary per backbone against the frozen
wave-1 grid (``wave1_cells.wave1_dataset_keys()`` x ``WAVE1_BACKBONES`` x ``WAVE1_SPLITS`` --
currently 56 datasets x 2 splits = 112 expected cells per backbone).

Tolerates a LIVE, still-writing sweep directory: a file that fails to parse (mid-``json.dump``
truncation) or is missing a required field is SKIPPED with a warning, never raises -- this script
is READ-ONLY on ``_repro/baselines/`` and must never crash mid-sweep.

Stage-1 discipline (CLAUDE.md): this report is purely factual (numbers + coverage only) -- no
"lowest/highest headroom" or other interpretive framing. Every number here is hypothesis-grade
until an owner review at the Stage-1 -> Stage-2 gate promotes it.

Usage:
  python scripts/baselines/summarize_wave1.py                          # stdout + _repro/wave1_results.md
  python scripts/baselines/summarize_wave1.py --out /tmp/w1.md         # stdout + custom path
  python scripts/baselines/summarize_wave1.py --dir /other/baselines   # scan a different dir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/baselines

import templates    # noqa: E402
import wave1_cells   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINES_DIR = REPO_ROOT / "_repro" / "baselines"
DEFAULT_OUT = REPO_ROOT / "_repro" / "wave1_results.md"

# ---- the frozen wave-1 grid (single source of truth: wave1_cells.py, not re-derived here) ----
EXPECTED_DATASETS = wave1_cells.wave1_dataset_keys()
EXPECTED_BACKBONES = list(wave1_cells.WAVE1_BACKBONES)
EXPECTED_SPLITS = list(wave1_cells.WAVE1_SPLITS)
EXPECTED_PER_BACKBONE = len(EXPECTED_DATASETS) * len(EXPECTED_SPLITS)


# ---------------------------------------------------------------------------------------------
# grouping + metric-name inference
# ---------------------------------------------------------------------------------------------

def group_of(dataset_key: str) -> str:
    """Table-bucket key: 'legacy' for scripts/p2_baselines.py-native loaders
    (templates.LEGACY_DATASETS), else the dataset's real K-type (templates.k_type_of). Mirrors
    the exact "legacy" if/else split run_baseline.py's dry_run() and wave1_cells.print_schedule()
    already use for their own [kt] column, so a dataset lands in the same bucket everywhere."""
    if dataset_key in templates.LEGACY_DATASETS:
        return "legacy"
    try:
        return templates.k_type_of(dataset_key)
    except KeyError:
        return "unknown"


def _group_sort_key(g: str):
    if g == "legacy":
        return (1, 0, g)
    m = re.match(r"^K(\d+)$", g)
    if m:
        return (0, int(m.group(1)), g)
    return (2, 0, g)                                              # any off-taxonomy group, last


def metric_name_of(detail: dict, score) -> str:
    """Infer a short, human-readable metric label from a per_item['detail'] dict's own shape.
    Schema-driven (reads what metrics.py actually recorded for that item) rather than a
    hand-curated per-dataset table that could silently drift from metrics.py's real dispatch."""
    if not isinstance(detail, dict):
        return "unknown"
    if score is None and "note" in detail and "refused" not in detail and "passes" not in detail:
        return "not verifiable (diagnostic/stub)"
    if "wer" in detail:
        return "1-WER"
    if "cer" in detail:
        return "1-CER"
    if "lang_em" in detail and "gender_em" in detail:
        return "LID exact-match"
    if "matched_by" in detail:
        return "intent EM"
    if "pred_idx" in detail and "gold_idx" in detail:
        return "MCQ EM"
    if set(detail.keys()) == {"pred", "gold"}:
        return "yes/no EM"
    if "gold_list" in detail:
        return "containment EM (weak-signal, multi-ref)"
    if "pred_label" in detail and "gold_label" in detail:
        return "closed-choice EM"
    if detail.get("reused") == "p2_baselines.reward":
        return f"legacy task-reward ({detail.get('task', '?')})"
    if "refused" in detail:
        return "refusal-rate"
    if "passes" in detail and "all_pass" in detail:
        return "IFEval pass-fraction"
    if "gold" in detail:
        return "EM (containment)"
    return "unknown"


def cell_metric(per_item: list) -> str:
    labels = []
    for it in per_item or []:
        if "detail" in it:
            labels.append(metric_name_of(it["detail"], it.get("score")))
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


# ---------------------------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------------------------

def load_cells(baselines_dir: Path) -> tuple[list[dict], list[str]]:
    """Scan *.json in baselines_dir. Returns (cells, warnings). Never raises: an unreadable,
    truncated, or field-incomplete file is skipped with a warning string, not an exception --
    the sweep this script reads from may still be mid-write."""
    cells: list[dict] = []
    warnings: list[str] = []
    if not baselines_dir.is_dir():
        warnings.append(f"SKIP: {baselines_dir} does not exist or is not a directory")
        return cells, warnings

    for p in sorted(baselines_dir.glob("*.json")):
        try:
            raw = p.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            warnings.append(f"SKIP {p.name}: unreadable/corrupt JSON ({type(e).__name__}: {e})")
            continue
        if not isinstance(data, dict):
            warnings.append(f"SKIP {p.name}: top-level JSON is not an object")
            continue
        try:
            dataset = data["dataset"]
            backbone = data["backbone"]
            split = data["split"]
            agg = data["aggregate"]
        except KeyError as e:
            warnings.append(f"SKIP {p.name}: missing required field {e}")
            continue
        if not isinstance(agg, dict):
            warnings.append(f"SKIP {p.name}: aggregate is not an object (partial write?)")
            continue

        cells.append({
            "dataset": dataset, "backbone": backbone, "split": split,
            "mean": agg.get("mean"), "ci95": agg.get("ci95"),
            "n_scored": agg.get("n_scored"), "n_total": agg.get("n_total"),
            "metric": cell_metric(data.get("per_item")),
            "group": group_of(dataset), "file": p.name,
        })
    return cells, warnings


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------

def format_cell(cell: dict | None) -> str:
    if cell is None:
        return "—"                                            # em dash: cell not present
    mean, ci95, n_scored, n_total = cell["mean"], cell["ci95"], cell["n_scored"], cell["n_total"]
    if mean is None:
        n_total_s = n_total if n_total is not None else "?"
        return f"unscored (0/{n_total_s})"
    if isinstance(ci95, (list, tuple)) and len(ci95) == 2 and all(v is not None for v in ci95):
        half_width = (float(ci95[1]) - float(ci95[0])) / 2.0
        return f"{mean:.4f} ±{half_width:.4f} ({n_scored})"
    return f"{mean:.4f} (n={n_scored})"


def render_group_table(group: str, dataset_keys: list[str],
                        cell_lookup: dict[tuple, dict]) -> list[str]:
    lines = [f"### {group}", ""]
    header = ["dataset", "metric"] + [f"{bb} / {sp}" for bb in EXPECTED_BACKBONES for sp in EXPECTED_SPLITS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for dk in sorted(dataset_keys):
        metrics_seen = [cell_lookup[(dk, bb, sp)]["metric"]
                         for bb in EXPECTED_BACKBONES for sp in EXPECTED_SPLITS
                         if (dk, bb, sp) in cell_lookup]
        metric = Counter(metrics_seen).most_common(1)[0][0] if metrics_seen else "—"
        row = [dk, metric]
        for bb in EXPECTED_BACKBONES:
            for sp in EXPECTED_SPLITS:
                row.append(format_cell(cell_lookup.get((dk, bb, sp))))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_coverage(cell_lookup: dict[tuple, dict], all_backbones: list[str]) -> list[str]:
    lines = ["## Coverage summary (wave-1 frozen grid)", ""]
    lines.append(f"Wave-1 grid: {len(EXPECTED_DATASETS)} datasets "
                 f"({', '.join(EXPECTED_BACKBONES)}) x {len(EXPECTED_SPLITS)} splits "
                 f"({', '.join(EXPECTED_SPLITS)}) = {EXPECTED_PER_BACKBONE} expected cells/backbone.")
    lines.append("")
    for bb in all_backbones:
        present = [(dk, sp) for dk in EXPECTED_DATASETS for sp in EXPECTED_SPLITS
                   if (dk, bb, sp) in cell_lookup]
        n_present = len(present)
        lines.append(f"- **{bb}**: {n_present}/{EXPECTED_PER_BACKBONE} cells present "
                     f"({EXPECTED_PER_BACKBONE - n_present} missing)"
                     + ("" if bb in EXPECTED_BACKBONES else "  (off-grid backbone, not in WAVE1_BACKBONES)"))
    lines.append("")

    lines.append("### Missing cells (wave-1 grid only)")
    lines.append("")
    any_missing = False
    for bb in EXPECTED_BACKBONES:
        missing_by_dataset: dict[str, list[str]] = {}
        for dk in EXPECTED_DATASETS:
            missing_splits = [sp for sp in EXPECTED_SPLITS if (dk, bb, sp) not in cell_lookup]
            if missing_splits:
                missing_by_dataset[dk] = missing_splits
        if not missing_by_dataset:
            lines.append(f"- **{bb}**: none (full wave-1 coverage).")
            continue
        any_missing = True
        lines.append(f"- **{bb}** ({len(missing_by_dataset)} dataset(s) with a missing split):")
        for dk in sorted(missing_by_dataset):
            lines.append(f"  - {dk}: {', '.join(missing_by_dataset[dk])}")
    if not any_missing:
        lines.append("(all backbones fully covered)")
    lines.append("")
    return lines


def render_report(cells: list[dict], warnings: list[str], baselines_dir: Path) -> str:
    cell_lookup = {(c["dataset"], c["backbone"], c["split"]): c for c in cells}
    observed_backbones = sorted({c["backbone"] for c in cells})
    all_backbones = sorted(set(EXPECTED_BACKBONES) | set(observed_backbones),
                            key=lambda b: (b not in EXPECTED_BACKBONES, b))

    # rows shown per group = union of the frozen wave-1 dataset keys and whatever was actually
    # observed on disk (defensive: a stray off-grid dataset key must still be visible, not dropped).
    groups: dict[str, set] = {}
    for dk in EXPECTED_DATASETS:
        groups.setdefault(group_of(dk), set()).add(dk)
    for c in cells:
        groups.setdefault(c["group"], set()).add(c["dataset"])

    lines = [
        "# Wave-1 baseline results (partial — sweep may still be running)",
        "",
        f"Source directory: `{baselines_dir}` (scanned read-only). "
        f"{len(cells)} cell JSON(s) parsed, {len(warnings)} skipped.",
        "",
        "Stage-1 directional numbers only (CLAUDE.md discipline): purely factual, no "
        "interpretation. `mean ±half-width (n_scored)` where half-width is half the width of "
        "the reported bootstrap 95% CI (`aggregate.ci95 = [lo, hi]`); see the underlying result "
        "JSON for the raw `[lo, hi]`. `unscored (0/n_total)` = every item's per-item score was "
        "`None` (e.g. a diagnostic-only K-type such as K9 squtr). `—` = cell not present on disk.",
        "",
    ]
    lines += render_coverage(cell_lookup, all_backbones)

    lines.append("## Results by K-type group")
    lines.append("")
    for g in sorted(groups, key=_group_sort_key):
        lines += render_group_table(g, sorted(groups[g]), cell_lookup)

    if warnings:
        lines.append("## Skipped files")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines) + "\n"


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

    cells, warnings = load_cells(args.dir)
    report = render_report(cells, warnings, args.dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    sys.stdout.write(report)
    print(f"\n[summarize_wave1] wrote {args.out} "
          f"({len(cells)} cells parsed, {len(warnings)} files skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()

"""scripts/baselines/rescore_cells.py — 2026-07-10 freeze-repair CPU-only rescore pass.

Context (see metrics.py's matching "2026-07-10 freeze-repair" comments): an adversarial audit of
the wave-1 baseline grid found 60 mechanically-invalid K8 cells (per-item ``score=0.0`` regardless
of the model's actual reply, because ``gold_idx`` never resolved):

  (A) 56 ``air-bench-foundation-*`` cells -- the K8 dispatcher's dict-gold branch only tried
      ``gold.get("answer", gold)``, but ``scripts/loaders/air_bench_foundation.py``'s own gold
      dict uses the key ``answer_gt`` (see that loader's Row-shape docstring), so ``gold`` stayed
      the whole dict and never matched an option.
  (B) 4 ``uro-bench-OpenbookQA-zh`` cells -- that subset's ``target_text`` gold is
      ``"<letter>. <option text>"`` (e.g. ``"D. 鲨鱼"``), unlike sibling HSK5-zh/GaokaoEval whose
      ``target_text`` is the bare letter alone -- ``score_k8_mcq`` had no branch for the longer
      format.

Both are now fixed in ``metrics.py`` (``score_k8_mcq`` + the K8 dispatcher's dict-gold branch).
This script does NOT regenerate anything -- every affected result JSON under ``_repro/baselines/``
already stores the model's raw ``reply`` text per item. It re-derives each item's ``gold`` by
re-invoking the SAME loader with the exact ``(split, n_requested, seed)`` recorded in that result
JSON (loaders are pure functions of those three inputs -- see their own
``(split, n, seed) -> list[Row]`` contract) and matching rows to stored ``per_item`` entries by
``item_id``, then recomputes ``metrics.score`` and the cell aggregate.

Reuses, does NOT reimplement:
  - ``metrics.score``                 -- the just-fixed K8 scorer (this is the whole point)
  - ``run_baseline._load_rows``       -- the SAME row-loading dispatcher run_baseline.py itself
                                          calls from ``run_one`` (so a rescore's rows are provably
                                          the exact rows the original generation pass used)
  - ``run_baseline.paired_bootstrap`` -- the SAME bootstrap-CI code path (identical NBOOT default)

Every REWRITTEN file gets a top-level ``"rescored"`` provenance block
(``{"date", "reason", "original_aggregate"}``) so the audit trail survives -- the old (broken)
aggregate is never silently lost. ``--check-only`` recomputes and prints without writing, used
both for the initial validation pass and the regression guard (confirming datasets OUTSIDE the
affected list score byte-identically before and after -- i.e. the fix does not touch them).

Usage:
  # validate ONE cell of each kind, no write:
  python scripts/baselines/rescore_cells.py --check-only \\
      --datasets air-bench-foundation-music-genre-fma --backbones qwen3-omni-30b-gguf --splits test
  python scripts/baselines/rescore_cells.py --check-only \\
      --datasets uro-bench-OpenbookQA-zh --backbones qwen3-omni-30b-gguf --splits test

  # regression guard on an UNAFFECTED cell (must show n_changed=0, mean unchanged):
  python scripts/baselines/rescore_cells.py --check-only \\
      --datasets uro-bench-HSK5-zh,mmar --backbones qwen3-omni-30b-gguf --splits test

  # rescore all 60 affected cells for real (rewrites the JSONs in place):
  python scripts/baselines/rescore_cells.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/baselines

import metrics                          # noqa: E402  (just-fixed K8 scorer)
import templates                        # noqa: E402
import run_baseline as rb               # noqa: E402  (reuse _load_rows + paired_bootstrap)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / "_repro" / "baselines"

REPAIR_DATE = "2026-07-10"
REPAIR_REASON = (
    "gold-key mismatch (wave-1 audit): K8's dict-gold branch only accepted key \"answer\", but "
    "air-bench-foundation's own gold dict uses \"answer_gt\" (56 cells, 14 datasets x 2 backbones "
    "x 2 splits); uro-bench-OpenbookQA-zh's target_text embeds \"<letter>. <text>\" rather than a "
    "bare letter, which score_k8_mcq's gold_idx resolution didn't parse (4 cells). Both fixed in "
    "metrics.py (score_k8_mcq + the K8 dispatcher's dict-gold branch) -- see that file's own "
    "2026-07-10 freeze-repair comments. This is a RESCORE ONLY: per-item scores + the cell "
    "aggregate were recomputed from the ALREADY-STORED model replies (no regeneration) via the "
    "fixed metrics.score and run_baseline.paired_bootstrap, unchanged."
)

# The 15 dataset keys the 2026-07-10 audit found mechanically-invalid (score=0.0 regardless of
# reply) -- 14 air-bench-foundation-* (gold-dict "answer_gt" key) + uro-bench-OpenbookQA-zh
# (gold-string "<letter>. <text>" format). See metrics.py's matching freeze-repair comments. This
# is the DEFAULT scope so a bare invocation never touches an unaffected cell.
AFFECTED_DATASETS = [
    "air-bench-foundation-acoustic-scene-cochlscene",
    "air-bench-foundation-acoustic-scene-tut2017",
    "air-bench-foundation-audio-grounding",
    "air-bench-foundation-music-aqa",
    "air-bench-foundation-music-genre-fma",
    "air-bench-foundation-music-genre-mtj-jamendo",
    "air-bench-foundation-music-instruments-mtj-jamendo",
    "air-bench-foundation-music-instruments-nsynth",
    "air-bench-foundation-music-midi-pitch-nsynth",
    "air-bench-foundation-music-midi-velocity-nsynth",
    "air-bench-foundation-music-mood-mtj-jamendo",
    "air-bench-foundation-sound-aqa-avqa",
    "air-bench-foundation-sound-aqa-clothoaqa",
    "air-bench-foundation-speech-grounding",
    "uro-bench-OpenbookQA-zh",
]

BACKBONES = ["qwen3-omni-30b-gguf", "meralion-2-gguf"]
SPLITS = ["dev", "test"]


def cell_path(dataset: str, backbone: str, split: str, baselines_dir: Path = BASELINES_DIR) -> Path:
    return baselines_dir / f"{dataset}__{backbone}__{split}.json"


def recompute_cell(path: Path) -> dict:
    """Recompute per-item scores + aggregate for one stored result JSON FROM ITS STORED REPLIES
    (no regeneration, no write). Returns a report dict; see ``rescore_cell`` for the write step.

    Idempotent: if ``path`` was already rescored by a prior run of this script, the ORIGINAL
    (pre-any-rescore) aggregate is carried forward from its existing ``"rescored"`` provenance
    block rather than re-captured from the file's current (already-fixed) aggregate -- so running
    this twice never overwrites the true baseline with an intermediate one.

    An item whose ``item_id`` can't be found in the freshly re-derived row set (observed for
    ``air-bench-foundation-sound-aqa-clothoaqa``, whose on-disk resolvable-clip pool is a genuine
    documented PARTIAL download -- see air_bench_foundation.py's module docstring -- so its
    resolved-row COUNT, and therefore ``rng.permutation(len(resolved))``, is not guaranteed stable
    across time if more/fewer clips appear on disk between the original run and this rescore) is
    marked ``score=None`` ("not verifiable this pass") rather than silently left at its old
    (already-known-broken) value -- honest-stub discipline, matching every other ``score=None``
    case in metrics.py.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    dataset_key, split = data["dataset"], data["split"]
    n, seed = data["n_requested"], data["seed"]

    rows = rb._load_rows(dataset_key, split, n, seed)          # reuse, not reimplement
    row_by_id = {r["meta"]["item_id"]: r for r in rows}

    new_per_item = []
    changed = 0
    missing_row_ids = []
    for it in data["per_item"]:
        if "error" in it or "item_id" not in it or "reply" not in it:
            new_per_item.append(it)          # nothing to rescore (generation itself failed)
            continue
        row = row_by_id.get(it["item_id"])
        if row is None:
            missing_row_ids.append(it["item_id"])
            new_it = dict(it)
            new_it["score"] = None
            new_it["detail"] = {
                "note": ("gold could not be re-derived: item_id not found when re-invoking the "
                         "loader with this cell's own (split, n_requested, seed) -- the on-disk "
                         "resolvable-row pool for this pair likely changed size since the original "
                         "generation run (see air_bench_foundation.py partial-download note), which "
                         "reshuffles the sampling permutation. Left unverifiable, not silently "
                         "kept at its prior (already-known-broken) score."),
                "prior_score": it.get("score"), "prior_detail": it.get("detail"),
            }
            if new_it.get("score") != it.get("score"):
                changed += 1
            new_per_item.append(new_it)
            continue
        result = metrics.score(dataset_key, row, it["reply"])
        new_it = dict(it)
        if new_it.get("score") != result["score"]:
            changed += 1
        new_it["score"] = result["score"]
        new_it["detail"] = result["detail"]
        new_per_item.append(new_it)

    numeric = [it["score"] for it in new_per_item if isinstance(it.get("score"), (int, float))]
    mean = round(sum(numeric) / len(numeric), 4) if numeric else None
    ci = rb.paired_bootstrap(numeric, seed=seed) if numeric else None    # reuse, not reimplement
    new_aggregate = {"mean": mean, "ci95": ci, "n_scored": len(numeric), "n_total": len(data["per_item"])}
    if missing_row_ids:
        new_aggregate["n_unresolved"] = len(missing_row_ids)

    kt = templates.k_type_of(dataset_key) if dataset_key not in templates.LEGACY_DATASETS else None
    if kt == "K4":  # mirrors run_baseline.run_one's own aggregate block (none of the 15 are K4)
        pairs = [(it["detail"]["gold_label"], it["detail"]["pred_label"]) for it in new_per_item
                 if "detail" in it and "gold_label" in it.get("detail", {})]
        if pairs:
            new_aggregate["macro_f1"] = metrics.aggregate_macro_f1(pairs)

    # idempotency: chain through the TRUE original aggregate if this file was already rescored by
    # a prior run of this script (see docstring) -- never re-capture an intermediate as "original".
    true_original_aggregate = data.get("rescored", {}).get("original_aggregate", data.get("aggregate"))

    return {
        "path": path, "data": data, "new_per_item": new_per_item,
        "old_aggregate": true_original_aggregate, "new_aggregate": new_aggregate,
        "n_changed": changed, "missing_row_ids": missing_row_ids,
    }


def rescore_cell(path: Path) -> dict:
    """``recompute_cell`` + write the result back to ``path`` with a ``"rescored"`` provenance
    block. Unaffected cells are simply never passed to this function (see ``main``'s scoping)."""
    report = recompute_cell(path)
    data = report["data"]
    data["per_item"] = report["new_per_item"]
    data["aggregate"] = report["new_aggregate"]
    data["rescored"] = {
        "date": REPAIR_DATE, "reason": REPAIR_REASON,
        "original_aggregate": report["old_aggregate"],
    }
    if report["missing_row_ids"]:
        data["rescored"]["unresolved_item_ids"] = report["missing_row_ids"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _print_examples(report: dict, k: int) -> None:
    for it in report["new_per_item"][:k]:
        if "detail" not in it:
            continue
        d = it["detail"]
        print(f"    item_id={it.get('item_id')!r} reply={it.get('reply')!r} "
              f"pred_idx={d.get('pred_idx')} gold_idx={d.get('gold_idx')} score={it.get('score')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(AFFECTED_DATASETS),
                     help="comma-separated dataset keys to process (default: the 15 known-affected keys)")
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--dir", type=Path, default=BASELINES_DIR)
    ap.add_argument("--check-only", action="store_true",
                     help="recompute + print before/after but do NOT write (validation / regression-guard mode)")
    ap.add_argument("--examples", type=int, default=3, help="how many per-item examples to print per cell")
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    backbones = [b for b in args.backbones.split(",") if b]
    splits = [s for s in args.splits.split(",") if s]

    n_ok, n_missing, n_err = 0, 0, 0
    for dk in datasets:
        for bb in backbones:
            for sp in splits:
                p = cell_path(dk, bb, sp, args.dir)
                if not p.exists():
                    print(f"SKIP (missing file): {p.name}")
                    n_missing += 1
                    continue
                try:
                    report = recompute_cell(p) if args.check_only else rescore_cell(p)
                except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole pass
                    print(f"ERROR {p.name}: {type(e).__name__}: {e}")
                    n_err += 1
                    continue
                old_agg = report["old_aggregate"] or {}
                new_agg = report["new_aggregate"]
                tag = "CHECK" if args.check_only else "WROTE"
                print(f"[{tag}] {p.name}: mean {old_agg.get('mean')} -> {new_agg.get('mean')}  "
                      f"(n_changed={report['n_changed']}/{new_agg['n_total']})")
                if args.examples:
                    _print_examples(report, args.examples)
                if report["missing_row_ids"]:
                    print(f"    WARNING: {len(report['missing_row_ids'])} item_id(s) not found in "
                          f"re-derived rows: {report['missing_row_ids'][:5]}")
                n_ok += 1

    print(f"\n=== {n_ok} cell(s) processed, {n_missing} missing, {n_err} error(s) ===")


if __name__ == "__main__":
    main()

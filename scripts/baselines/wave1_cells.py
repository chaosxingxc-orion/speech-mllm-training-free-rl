"""scripts/baselines/wave1_cells.py — Wave-1 frozen cell list + checkpointed per-cell runner.

Wave-1 scope (frozen, per the 2026-07-09 task brief): a FOCUSED subset of the full Step-1 grid
(``templates.DATASET_KTYPE``, 71 entries) rather than the whole thing --

    K8 (all)  +  K9 closed-book (``squtr``, diagnostic-only)  +  K1 (all)  +  K2 (all)

-- crossed with the two frozen backbones (``run_baseline.BACKBONES``: ``qwen3-omni-30b-gguf``,
``meralion-2-gguf``, owner ruling 2026-07-09 双底座定稿) x {dev, test}. "K8 all" already contains
BOTH the legacy (``p2_baselines``-native) and package (``uro-bench-*``) loader implementations of
SQuAD-zh/OpenbookQA-zh (see ``templates.DATASET_KTYPE`` and ``FREEZE_SHEET.md`` §2.6's "keep both,
cross-check they agree" scoping decision) -- ``wave1_dataset_keys()`` asserts that dual-run pair is
present rather than silently relying on it.

This module owns exactly the wave-1 CELL LIST + a checkpointed single-cell runner
(``run_cell``/``run_backbone``, thin wrappers around ``run_baseline.run_one``/``write_result`` --
NOT reimplemented). GPU-session lifecycle (acquire / resident-server up / iterate / down / release)
is ``run_wave1.sh``'s job, not this module's -- ``run_backbone``/``run_cell`` assume the resident
llama-server for the target backbone is already up and the GPU lock already held.

Usage:
  python wave1_cells.py --dry-run                                   # full wave-1 schedule + wall-clock estimate
  python wave1_cells.py --backbone meralion-2-gguf --dry-run         # one backbone's schedule only
  python wave1_cells.py --backbone qwen3-omni-30b-gguf --execute     # RUNS cells (needs GPU session + resident server up)
  python wave1_cells.py --backbone qwen3-omni-30b-gguf --split dev --dataset big-bench-audio --execute   # one cell
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # scripts/baselines

import run_baseline as rb             # noqa: E402  (reuses run_one/write_result/OUT_DIR/DEV_N/TEST_N)
import templates                      # noqa: E402

# ---- frozen wave-1 backbone/split roster ----------------------------------------------------
WAVE1_BACKBONES = ["qwen3-omni-30b-gguf", "meralion-2-gguf"]
WAVE1_SPLITS = ["dev", "test"]

# Per-generation wall-clock estimate (seconds/item), owner-supplied for the --dry-run schedule
# (2026-07-09 task brief): qwen3-omni-30b-gguf is a 30B-A3B MoE partially CPU-offloaded
# (-ngl 28) -> slower; meralion-2-gguf is a 3B fully-resident model (-ngl 99) -> faster. NOT
# independently measured on this box at the time this was written -- update from a real cell's
# recorded "elapsed_s" / n_scored once available (see run_one's result JSON).
GEN_TIME_S = {"qwen3-omni-30b-gguf": 2.8, "meralion-2-gguf": 1.0}


def wave1_dataset_keys() -> list[str]:
    """The frozen wave-1 dataset-key list: K1 (all) + K2 (all) + K8 (all) + K9 closed-book
    (``squtr``), sorted for a deterministic, reviewable schedule order.
    """
    keys = {k for k, v in templates.DATASET_KTYPE.items() if v in ("K1", "K2", "K8")}
    keys.add("squtr")  # K9, diagnostic closed-book pass only (templates.k9_squtr_closed_book)

    required_pair = {"SQuAD-zh", "OpenbookQA-zh", "uro-bench-SQuAD-zh", "uro-bench-OpenbookQA-zh"}
    missing = required_pair - keys
    assert not missing, (
        f"wave1_dataset_keys: missing required legacy/package dual-run pair member(s) {missing} "
        "-- see FREEZE_SHEET.md sec.2.6 (keep-both decision) / templates.DATASET_KTYPE"
    )
    return sorted(keys)


def wave1_cells(backbones: list[str] | None = None, splits: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(dataset, backbone, split) triples in a fixed, deterministic order: dataset-major, then
    backbone, then split -- so ``--dry-run``'s printed schedule and ``--execute``'s actual run
    order are the same thing, and a resumed run after a partial failure iterates identically."""
    backbones = backbones or WAVE1_BACKBONES
    splits = splits or WAVE1_SPLITS
    return [(dk, bb, sp) for dk in wave1_dataset_keys() for bb in backbones for sp in splits]


def result_path(dataset: str, backbone: str, split: str) -> Path:
    return rb.OUT_DIR / f"{dataset}__{backbone}__{split}.json"


def is_checkpointed(dataset: str, backbone: str, split: str) -> bool:
    """True iff a COMPLETE result JSON already exists for this cell (has a non-null "aggregate").
    A missing/unreadable/partial (e.g. crash mid-``json.dump``) file is treated as NOT
    checkpointed so a re-invocation safely retries it -- never silently skips real work."""
    p = result_path(dataset, backbone, split)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("aggregate"))


def n_for(split: str) -> int:
    return rb.DEV_N if split == "dev" else rb.TEST_N


def estimate_wallclock_s(cells: list[tuple[str, str, str]]) -> dict:
    per_backbone: dict[str, float] = {}
    total = 0.0
    for dk, bb, sp in cells:
        secs = n_for(sp) * GEN_TIME_S.get(bb, 2.0)
        per_backbone[bb] = per_backbone.get(bb, 0.0) + secs
        total += secs
    return {"total_s": total, "per_backbone_s": per_backbone}


def print_schedule(backbones: list[str] | None = None, splits: list[str] | None = None,
                    only_dataset: str | None = None) -> None:
    backbones = backbones or WAVE1_BACKBONES
    splits = splits or WAVE1_SPLITS
    datasets = wave1_dataset_keys()
    if only_dataset is not None:
        assert only_dataset in datasets, f"{only_dataset!r} is not a wave1 dataset key ({len(datasets)} total)"
        datasets = [only_dataset]
    cells = [(dk, bb, sp) for dk in datasets for bb in backbones for sp in splits]
    est = estimate_wallclock_s(cells)

    print(f"=== Wave-1 frozen cell list: {len(datasets)} datasets x {len(backbones)} backbones "
          f"x {len(splits)} splits = {len(cells)} cells ===")
    print(f"datasets ({len(datasets)}): {', '.join(datasets)}\n")
    n_pending = 0
    for dk, bb, sp in cells:
        n = n_for(sp)
        kt = "legacy" if dk in templates.LEGACY_DATASETS else templates.k_type_of(dk)
        done = is_checkpointed(dk, bb, sp)
        n_pending += 0 if done else 1
        status = "CHECKPOINTED (skip)" if done else "pending"
        print(f"  {dk:45s} [{kt:>4s}] {bb:22s} {sp:4s} n={n:<4d} {status}")
    print(f"\n{len(cells) - n_pending}/{len(cells)} cells already checkpointed; {n_pending} pending.")
    for bb in backbones:
        secs = est["per_backbone_s"].get(bb, 0.0)
        print(f"estimated wall-clock, {bb}: {secs / 60:.1f} min ({secs:.0f}s) "
              f"@ {GEN_TIME_S.get(bb, 2.0)}s/gen")
    print(f"estimated wall-clock, TOTAL (both backbones, full grid, ignoring checkpoints): "
          f"{est['total_s'] / 60:.1f} min ({est['total_s'] / 3600:.2f} h)")


def run_cell(dataset: str, backbone: str, split: str, skip_checkpoint: bool = True) -> dict | None:
    """Run exactly one wave-1 cell via ``run_baseline.run_one``/``write_result`` (not
    reimplemented), with checkpoint-skip. NEVER raises -- catches and logs, mirroring
    ``run_baseline.main()``'s own per-cell guard, so one bad/env-blocked cell (e.g. meld's
    NeedsExtraction, air-bench's NeedsAirBenchFoundationAudio) cannot sink the rest of the sweep.
    """
    if skip_checkpoint and is_checkpointed(dataset, backbone, split):
        print(f"  [skip] {dataset}/{backbone}/{split} already checkpointed at "
              f"{result_path(dataset, backbone, split)}", flush=True)
        return None
    print(f"=== wave1 cell: {dataset} / {backbone} / {split} ===", flush=True)
    try:
        result = rb.run_one(dataset, backbone, split, None)
    except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole wave-1 sweep
        print(f"  !! {dataset}/{backbone}/{split} FAILED: {type(e).__name__}: {e}", flush=True)
        return None
    p = rb.write_result(result)
    agg = result["aggregate"]
    print(f"  wrote {p} (mean={agg['mean']} ci={agg['ci95']} n_scored={agg['n_scored']}/{agg['n_total']} "
          f"elapsed_s={result['elapsed_s']})", flush=True)
    return result


def run_backbone(backbone: str, splits: list[str] | None = None, only_dataset: str | None = None) -> None:
    """Iterate every wave-1 (dataset, split) cell for ONE backbone. Assumes the GPU lock is
    already held and that backbone's resident llama-server is already up -- see run_wave1.sh."""
    datasets = [only_dataset] if only_dataset else wave1_dataset_keys()
    if only_dataset:
        assert only_dataset in wave1_dataset_keys(), f"{only_dataset!r} is not a wave1 dataset key"
    cells = [(dk, backbone, sp) for dk in datasets for sp in (splits or WAVE1_SPLITS)]
    t0 = time.time()
    n_ran = n_skipped = n_failed = 0
    for dk, bb, sp in cells:
        before = is_checkpointed(dk, bb, sp)
        result = run_cell(dk, bb, sp)
        if result is not None:
            n_ran += 1
        elif before:
            n_skipped += 1
        else:
            n_failed += 1
    print(f"=== backbone {backbone} done in {time.time() - t0:.0f}s "
          f"(ran={n_ran} skipped={n_skipped} failed={n_failed} of {len(cells)}) ===", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=WAVE1_BACKBONES, default=None,
                     help="restrict to one backbone (default: both, in sequence)")
    ap.add_argument("--split", choices=WAVE1_SPLITS, default=None,
                     help="restrict to one split (default: both)")
    ap.add_argument("--dataset", default=None,
                     help="restrict to one wave1 dataset key (default: the whole frozen list)")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule + wall-clock estimate; no model calls")
    ap.add_argument("--execute", action="store_true",
                     help="actually run the cells -- requires the GPU session held + the target "
                          "backbone's resident llama-server already up (see run_wave1.sh)")
    ap.add_argument("--no-checkpoint", action="store_true",
                     help="ignore existing result JSONs and rerun every selected cell (default: skip checkpointed cells)")
    args = ap.parse_args()

    backbones = [args.backbone] if args.backbone else WAVE1_BACKBONES
    splits = [args.split] if args.split else WAVE1_SPLITS

    if args.execute:
        for bb in backbones:
            datasets = [args.dataset] if args.dataset else None
            if args.no_checkpoint:
                cells = [(args.dataset, bb, sp) for sp in splits] if args.dataset else wave1_cells([bb], splits)
                for dk, b, sp in cells:
                    run_cell(dk, b, sp, skip_checkpoint=False)
            else:
                run_backbone(bb, splits, only_dataset=args.dataset)
        return

    # default (including --dry-run): print the schedule, never touch a model/GPU.
    print_schedule(backbones, splits, only_dataset=args.dataset)


if __name__ == "__main__":
    main()

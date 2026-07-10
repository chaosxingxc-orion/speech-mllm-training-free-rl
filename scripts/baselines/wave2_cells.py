"""scripts/baselines/wave2_cells.py — Wave-2 frozen cell list + checkpointed per-cell runner.

Wave-2 scope (mirrors wave-1's structure exactly, see ``wave1_cells.py``): the K4-K7 "structured
classification" family of the full Step-1 grid (``templates.DATASET_KTYPE``, 71+ entries) --

    K4 (SER, all)  +  K5 (speaker-attribute probes only, per FREEZE_SHEET.md sec.2 item 1 /
    templates.py's "K5 SCOPING DECISION" -- cn-celeb1/voxceleb1-test-split are NOT grid entries,
    see templates.K5_EXCLUDED_SID)  +  K6 (SLU intent, all)  +  K7 (SLU slot, all)

-- crossed with the two frozen backbones (``run_baseline.BACKBONES``: ``qwen3-omni-30b-gguf``,
``meralion-2-gguf``, owner ruling 2026-07-09 双底座定稿) x {dev, test}. 16 dataset keys total (7 K4
+ 2 K5 + 4 K6 + 3 K7) -- see ``wave2_dataset_keys()``, which asserts that count so a future
``templates.DATASET_KTYPE`` edit cannot silently change wave-2's scope out from under this frozen
list. One of the 16 (``minds14-zh``, K6) is a ``templates.LEGACY_DATASETS`` key (own
``p2_baselines``-native loader, passthrough-instruction, see ``run_baseline._legacy_rows``) --
handled exactly like wave-1's legacy SQuAD-zh/OpenbookQA-zh pair, not specially here.

Two structural quirks inherited from ``run_baseline.py`` that a wave-2 caller should know before
reading result JSONs:
  - K5/K6/K7 dataset keys are SYNTHETIC sub-keys of a shared base loader (e.g.
    ``speech-massive-de-DE-attr`` / ``-slot`` both re-load ``speech-massive-de-DE``, see
    ``run_baseline._load_rows``'s ``-attr``/``-slot`` branches) -- each is still its own
    (dataset, backbone, split) cell / result JSON, exactly like every other wave-2 key.
  - K7 (slot-filling)'s per-item ``score`` is ALWAYS ``None`` by design (corpus-level slot-F1 needs
    the whole cell's predictions at once via ``metrics.aggregate_slot_f1_*``, which is wired but
    NOT invoked by ``run_baseline.run_one``/this module -- see ``run_baseline.run_one``'s "K7" note
    and ``FREEZE_SHEET.md`` sec.1 K7 row). A K7 result JSON's ``aggregate.mean`` will be ``null``
    for that reason alone, not a cell failure.

**GPU EXECUTION GATE (2026-07-10, per task brief): wave-2 is LAUNCH-READY but NOT released.** This
module and ``run_wave2.sh`` are code-complete, import-clean, and ``--dry-run``-verified CPU-only --
but neither may touch the GPU / resident llama-server until the owner explicitly signs off on wave-2
(波2 放行, "wave-2 release"). Both this module's ``--execute`` path and ``run_wave2.sh``'s non-dry-run
path therefore hard-require ``WAVE2_RELEASE=1`` in the environment (see ``main()`` below) -- a
belt-and-suspenders check on top of "nobody actually runs it yet", not a substitute for the
GPU-session cooperative lock (``scripts/gpu_session.sh``), which still governs actual hardware access.

This module owns exactly the wave-2 CELL LIST + a checkpointed single-cell runner
(``run_cell``/``run_backbone``, thin wrappers around ``run_baseline.run_one``/``write_result`` --
NOT reimplemented, mirrors wave-1). GPU-session lifecycle (acquire / resident-server up / iterate /
down / release) is ``run_wave2.sh``'s job, not this module's -- ``run_backbone``/``run_cell`` assume
the resident llama-server for the target backbone is already up and the GPU lock already held.

Usage:
  python wave2_cells.py --dry-run                                   # full wave-2 schedule + wall-clock estimate
  python wave2_cells.py --backbone meralion-2-gguf --dry-run         # one backbone's schedule only
  WAVE2_RELEASE=1 python wave2_cells.py --backbone qwen3-omni-30b-gguf --execute   # RUNS cells (needs GPU session + resident server up + owner release)
  WAVE2_RELEASE=1 python wave2_cells.py --backbone qwen3-omni-30b-gguf --split dev --dataset crema-d --execute   # one cell
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # scripts/baselines

import run_baseline as rb             # noqa: E402  (reuses run_one/write_result/OUT_DIR/DEV_N/TEST_N)
import templates                      # noqa: E402

# ---- frozen wave-2 backbone/split roster (identical to wave-1's, owner ruling 2026-07-09) -------
WAVE2_BACKBONES = ["qwen3-omni-30b-gguf", "meralion-2-gguf"]
WAVE2_SPLITS = ["dev", "test"]

# Per-generation wall-clock estimate (seconds/item) -- reuses wave1_cells.GEN_TIME_S verbatim (same
# two backbones, same llama.cpp generation path via run_baseline.generate/gen_llamacpp; nothing
# wave-2-specific changes per-item latency). NOT independently re-measured for wave-2's own
# datasets -- update from a real wave-2 cell's recorded "elapsed_s"/n_scored once wave-2 actually
# runs (see run_one's result JSON), same discipline as wave1_cells.py's own note.
GEN_TIME_S = {"qwen3-omni-30b-gguf": 2.8, "meralion-2-gguf": 1.0}

# The frozen wave-2 K-type family + expected dataset-key count (see module docstring) -- an
# explicit constant so wave2_dataset_keys()'s assertion documents exactly what it is protecting.
WAVE2_KTYPES = ("K4", "K5", "K6", "K7")
WAVE2_EXPECTED_N = 16


def wave2_dataset_keys() -> list[str]:
    """The frozen wave-2 dataset-key list: every ``templates.DATASET_KTYPE`` entry whose K-type is
    K4, K5, K6, or K7 (see ``WAVE2_KTYPES``), sorted for a deterministic, reviewable schedule order.

    Asserts the count is exactly ``WAVE2_EXPECTED_N`` (16, verified 2026-07-10: 7 K4 + 2 K5 + 4 K6
    + 3 K7) so a future edit to ``templates.DATASET_KTYPE`` cannot silently change wave-2's frozen
    scope without this module noticing.
    """
    keys = {k for k, v in templates.DATASET_KTYPE.items() if v in WAVE2_KTYPES}
    assert len(keys) == WAVE2_EXPECTED_N, (
        f"wave2_dataset_keys: expected {WAVE2_EXPECTED_N} K4-K7 dataset keys, found {len(keys)}: "
        f"{sorted(keys)} -- templates.DATASET_KTYPE changed since wave-2 was frozen (2026-07-10); "
        "re-check FREEZE_SHEET.md sec.1 K4/K5/K6/K7 rows and update WAVE2_EXPECTED_N deliberately "
        "before proceeding."
    )
    return sorted(keys)


def wave2_cells(backbones: list[str] | None = None, splits: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(dataset, backbone, split) triples in a fixed, deterministic order: dataset-major, then
    backbone, then split -- so ``--dry-run``'s printed schedule and ``--execute``'s actual run
    order are the same thing, and a resumed run after a partial failure iterates identically."""
    backbones = backbones or WAVE2_BACKBONES
    splits = splits or WAVE2_SPLITS
    return [(dk, bb, sp) for dk in wave2_dataset_keys() for bb in backbones for sp in splits]


def result_path(dataset: str, backbone: str, split: str) -> Path:
    return rb.OUT_DIR / f"{dataset}__{backbone}__{split}.json"


def is_checkpointed(dataset: str, backbone: str, split: str) -> bool:
    """True iff a COMPLETE result JSON already exists for this cell (has a non-null "aggregate").
    A missing/unreadable/partial (e.g. crash mid-``json.dump``) file is treated as NOT
    checkpointed so a re-invocation safely retries it -- never silently skips real work.

    Note K7's own "aggregate.mean is always null by design" quirk (see module docstring) is
    orthogonal to this check -- ``is_checkpointed`` only looks at whether the "aggregate" KEY is
    present/truthy (a dict), not at whether "mean" inside it is non-null."""
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
    backbones = backbones or WAVE2_BACKBONES
    splits = splits or WAVE2_SPLITS
    datasets = wave2_dataset_keys()
    if only_dataset is not None:
        assert only_dataset in datasets, f"{only_dataset!r} is not a wave2 dataset key ({len(datasets)} total)"
        datasets = [only_dataset]
    cells = [(dk, bb, sp) for dk in datasets for bb in backbones for sp in splits]
    est = estimate_wallclock_s(cells)

    print(f"=== Wave-2 frozen cell list: {len(datasets)} datasets x {len(backbones)} backbones "
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
    print("\nNOTE: wave-2 GPU execution (--execute) is gated on owner release (波2 放行) -- requires "
          "WAVE2_RELEASE=1 in the environment, see module docstring.")


def run_cell(dataset: str, backbone: str, split: str, skip_checkpoint: bool = True) -> dict | None:
    """Run exactly one wave-2 cell via ``run_baseline.run_one``/``write_result`` (not
    reimplemented), with checkpoint-skip. NEVER raises -- catches and logs, mirroring
    ``run_baseline.main()``'s own per-cell guard, so one bad/env-blocked cell (e.g. a K5/K6/K7
    loader hitting an unexpected env issue) cannot sink the rest of the sweep.
    """
    if skip_checkpoint and is_checkpointed(dataset, backbone, split):
        print(f"  [skip] {dataset}/{backbone}/{split} already checkpointed at "
              f"{result_path(dataset, backbone, split)}", flush=True)
        return None
    print(f"=== wave2 cell: {dataset} / {backbone} / {split} ===", flush=True)
    try:
        result = rb.run_one(dataset, backbone, split, None)
    except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole wave-2 sweep
        print(f"  !! {dataset}/{backbone}/{split} FAILED: {type(e).__name__}: {e}", flush=True)
        return None
    p = rb.write_result(result)
    agg = result["aggregate"]
    print(f"  wrote {p} (mean={agg['mean']} ci={agg['ci95']} n_scored={agg['n_scored']}/{agg['n_total']} "
          f"elapsed_s={result['elapsed_s']})", flush=True)
    return result


def run_backbone(backbone: str, splits: list[str] | None = None, only_dataset: str | None = None) -> dict:
    """Iterate every wave-2 (dataset, split) cell for ONE backbone. Assumes the GPU lock is
    already held and that backbone's resident llama-server is already up -- see run_wave2.sh.

    Returns the {ran, skipped, failed, total} counts (also used by main() to decide the process
    exit code -- mirrors wave1_cells.run_backbone's fix for the same class of bug, see its
    docstring's 2026-07-09 postmortem note)."""
    datasets = [only_dataset] if only_dataset else wave2_dataset_keys()
    if only_dataset:
        assert only_dataset in wave2_dataset_keys(), f"{only_dataset!r} is not a wave2 dataset key"
    cells = [(dk, backbone, sp) for dk in datasets for sp in (splits or WAVE2_SPLITS)]
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
    # Machine-parseable twin of the line above -- run_wave2.sh greps this exact prefix out of the
    # tee'd log to print its own per-backbone summary and decide whether to exit non-zero. Keep the
    # key set (ran/skipped/failed/total) and "key=value" shape stable; run_wave2.sh's extract_field
    # parses it with plain shell parameter expansion (no PCRE dependency) -- mirrors WAVE1_SUMMARY.
    print(f"WAVE2_SUMMARY backbone={backbone} ran={n_ran} skipped={n_skipped} failed={n_failed} "
          f"total={len(cells)}", flush=True)
    return {"ran": n_ran, "skipped": n_skipped, "failed": n_failed, "total": len(cells)}


def _require_release() -> None:
    if os.environ.get("WAVE2_RELEASE") != "1":
        print(
            "ERROR: wave2_cells.py --execute requires owner release (波2 放行) -- set "
            "WAVE2_RELEASE=1 in the environment to confirm the owner has released wave-2 for GPU "
            "execution. See this module's docstring 'GPU EXECUTION GATE' note.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=WAVE2_BACKBONES, default=None,
                     help="restrict to one backbone (default: both, in sequence)")
    ap.add_argument("--split", choices=WAVE2_SPLITS, default=None,
                     help="restrict to one split (default: both)")
    ap.add_argument("--dataset", default=None,
                     help="restrict to one wave2 dataset key (default: the whole frozen list)")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule + wall-clock estimate; no model calls")
    ap.add_argument("--execute", action="store_true",
                     help="actually run the cells -- requires the GPU session held + the target "
                          "backbone's resident llama-server already up (see run_wave2.sh) + "
                          "WAVE2_RELEASE=1 (owner release gate, see module docstring)")
    ap.add_argument("--no-checkpoint", action="store_true",
                     help="ignore existing result JSONs and rerun every selected cell (default: skip checkpointed cells)")
    args = ap.parse_args()

    backbones = [args.backbone] if args.backbone else WAVE2_BACKBONES
    splits = [args.split] if args.split else WAVE2_SPLITS

    if args.execute:
        _require_release()
        total_failed = 0
        for bb in backbones:
            if args.no_checkpoint:
                cells = [(args.dataset, bb, sp) for sp in splits] if args.dataset else wave2_cells([bb], splits)
                n_ran = n_failed = 0
                for dk, b, sp in cells:
                    # skip_checkpoint=False means run_cell() never skips -- None here always means
                    # the cell raised (caught inside run_cell), never "already checkpointed".
                    result = run_cell(dk, b, sp, skip_checkpoint=False)
                    n_ran += 1 if result is not None else 0
                    n_failed += 0 if result is not None else 1
                print(f"WAVE2_SUMMARY backbone={bb} ran={n_ran} skipped=0 failed={n_failed} "
                      f"total={len(cells)}", flush=True)
                total_failed += n_failed
            else:
                summary = run_backbone(bb, splits, only_dataset=args.dataset)
                total_failed += summary["failed"]
        if total_failed > 0:
            print(f"ERROR: wave2_cells.py --execute: {total_failed} cell(s) failed across "
                  f"{len(backbones)} backbone(s) -- see FAILED lines above.", file=sys.stderr)
            sys.exit(1)
        return

    # default (including --dry-run): print the schedule, never touch a model/GPU.
    print_schedule(backbones, splits, only_dataset=args.dataset)


if __name__ == "__main__":
    main()

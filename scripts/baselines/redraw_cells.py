"""scripts/baselines/redraw_cells.py — disjoint-redraw RERUN driver (2026-07-10, owner ruling
Decision-Log 续10, "dev/test 全部重抽（最严谨选项）"; task scope = ALL 65 wave-1/wave-2 dataset keys
frozen in ``_repro/redraw_manifest.json``, qwen3-omni-30b-gguf ONLY).

## Precise rerun-list logic (this module's whole job)

``redraw.py`` already froze a genuinely-disjoint (dev, test) id pair for every one of the 65
overlapping dataset keys (``_repro/redraw_manifest.json``'s ``datasets[<key>]["new_dev_ids"]``/
``["new_test_ids"]``). Naively rerunning all 65 x 2 = 130 cells would be wasteful: ``draw_disjoint``
draws **test FIRST from the whole pool with the SAME seed (`TEST_SEED`) the ORIGINAL non-disjoint
draw already used** (see ``_common.draw_disjoint``'s docstring + ``redraw.py``'s module docstring) --
so for any dataset whose pool didn't shrink/reorder between the original run and the redraw, the new
disjoint TEST slice is BYTE-IDENTICAL to the old one; only DEV changes (drawn from the pool
*minus* test, hence a different sub-permutation). This module never assumes that though -- it
compares the ACTUAL id sets (frozen manifest vs. the existing result JSON's stored
``per_item[*].item_id`` list) for every (dataset, split) cell and reruns only where they differ,
counting + reporting every skip so "why didn't X rerun" is always answerable from the printed
table, not asserted from the general argument above.

Usage:
  python redraw_cells.py --census                                  # read-only rerun/skip table, no GPU/model calls
  python redraw_cells.py --dry-run                                  # alias for --census (mirrors wave3_cells.py's --dry-run naming)
  python redraw_cells.py --execute --dataset aishell-1 --split dev --parallel 4
                                                                       # ONE cell (task brief step 3 validation)
  python redraw_cells.py --execute --parallel 4                      # RUNS every cell whose id-set differs from
                                                                       # the frozen manifest (needs GPU session +
                                                                       # resident qwen3-omni-30b-gguf server up)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # scripts/baselines

import run_baseline as rb             # noqa: E402  (reuses run_one/write_result/OUT_DIR/DEV_N/TEST_N)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "_repro" / "redraw_manifest.json"

BACKBONE = "qwen3-omni-30b-gguf"          # the redraw task's ONLY backbone (task brief: "qwen3-omni-30b-gguf only")
SPLITS = ("dev", "test")
ARCHIVE_TAG = "pre-redraw-20260710"        # append-only archive suffix (task brief step 2)

# Per-generation wall-clock estimate (seconds/item) -- reused verbatim from wave1_cells.GEN_TIME_S /
# wave3_cells.GEN_TIME_S (same qwen3-omni-30b-gguf generation path via run_baseline.generate).
GEN_TIME_S = 2.8
BATCH_SPEEDUP = 1.75   # owner-approved smoke verdict (see wave3_cells.py) -- used only for --dry-run's estimate

# K7 (SLU slot-filling): per-item score is None by design; corpus-level slot-F1 needs a SEPARATE
# aggregation pass over the generated per-item parsed_slots (see k7_slot_repass.py). These 3
# dataset keys are exactly wave-2's K7 family (mirrors k7_slot_repass.K7_CELLS's dataset set).
K7_DATASET_KEYS = ("slurp-slot", "speech-massive-de-DE-slot", "speech-massive-fr-FR-slot")


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_dataset_keys(manifest: dict | None = None) -> list[str]:
    manifest = manifest or load_manifest()
    return sorted(manifest["datasets"].keys())


def manifest_ids(manifest: dict, dataset: str, split: str) -> list[str] | None:
    entry = manifest["datasets"].get(dataset)
    if entry is None or "error" in entry:
        return None
    key = "new_dev_ids" if split == "dev" else "new_test_ids"
    return entry.get(key)


def result_path(dataset: str, backbone: str, split: str) -> Path:
    return rb.OUT_DIR / f"{dataset}__{backbone}__{split}.json"


def archive_path(dataset: str, backbone: str, split: str) -> Path:
    return rb.OUT_DIR / f"{dataset}__{backbone}__{split}.{ARCHIVE_TAG}.json"


def _existing_ids_and_data(dataset: str, backbone: str, split: str) -> tuple[list[str] | None, dict | None]:
    p = result_path(dataset, backbone, split)
    if not p.exists():
        return None, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    ids = [it.get("item_id") for it in data.get("per_item", []) if it.get("item_id") is not None]
    return ids, data


def needs_rerun(dataset: str, split: str, manifest: dict | None = None) -> tuple[bool, str]:
    """(need_rerun, reason). ``reason`` is one of:
      "no-manifest-entry"      -- dataset/split has no frozen disjoint manifest (shouldn't happen
                                  for any of the 65 keys; defensive)
      "no-existing-result"     -- never generated before at all (rerun = first generation, not really
                                  a "redraw" but handled identically)
      "id-sets-already-match"  -- the existing result JSON's per_item ids ALREADY equal the frozen
                                  disjoint manifest's ids (this is what makes reruns naturally
                                  idempotent/checkpointed: after a cell is rerun once, a second
                                  invocation sees this and skips) -- covers BOTH "redraw agent
                                  verified permutation-loader test slices are identical by
                                  construction" cases AND a prior partial run of this very script.
      "id-sets-differ"         -- genuine rerun needed.
    """
    manifest = manifest or load_manifest()
    mids = manifest_ids(manifest, dataset, split)
    if mids is None:
        return False, "no-manifest-entry"
    old_ids, _ = _existing_ids_and_data(dataset, BACKBONE, split)
    if old_ids is None:
        return True, "no-existing-result"
    if set(old_ids) == set(mids):
        return False, "id-sets-already-match"
    return True, "id-sets-differ"


def all_cells(manifest: dict | None = None) -> list[tuple[str, str]]:
    manifest = manifest or load_manifest()
    return [(dk, sp) for dk in manifest_dataset_keys(manifest) for sp in SPLITS]


def census(manifest: dict | None = None) -> dict[tuple[str, str], dict]:
    """Read-only: (dataset, split) -> {"need_rerun": bool, "reason": str}. No loader/model calls --
    only the frozen manifest + existing result JSONs' per_item lists (mirrors redraw.py's own
    read-only ``census()`` discipline)."""
    manifest = manifest or load_manifest()
    return {(dk, sp): dict(zip(("need_rerun", "reason"), needs_rerun(dk, sp, manifest)))
            for dk, sp in all_cells(manifest)}


def print_census(manifest: dict | None = None, parallel: int = 4) -> dict:
    manifest = manifest or load_manifest()
    rep = census(manifest)
    cells = all_cells(manifest)
    n_rerun = sum(1 for v in rep.values() if v["need_rerun"])
    n_skip = len(cells) - n_rerun
    by_reason: dict[str, int] = {}
    for v in rep.values():
        by_reason[v["reason"]] = by_reason.get(v["reason"], 0) + 1

    print(f"=== redraw rerun census: {len(manifest['datasets'])} dataset keys x {len(SPLITS)} splits "
          f"= {len(cells)} cells (backbone={BACKBONE} only) ===\n")
    for dk in manifest_dataset_keys(manifest):
        parts = []
        for sp in SPLITS:
            v = rep[(dk, sp)]
            tag = "RERUN" if v["need_rerun"] else "skip"
            parts.append(f"{sp}={tag}({v['reason']})")
        k7 = " [K7 slot-F1 repass after generation]" if dk in K7_DATASET_KEYS else ""
        print(f"  {dk:45s} {'  '.join(parts)}{k7}")

    print(f"\nreason breakdown: {json.dumps(by_reason, indent=2)}")
    print(f"\nTOTAL: {n_rerun} cell(s) need rerun, {n_skip} cell(s) skip, of {len(cells)}.")
    est_seq = n_rerun * 0  # placeholder; real estimate below needs each cell's own n
    total_n = 0
    for dk, sp in cells:
        if rep[(dk, sp)]["need_rerun"]:
            mids = manifest_ids(manifest, dk, sp) or []
            total_n += len(mids)
    secs_seq = total_n * GEN_TIME_S
    secs_batch = secs_seq / BATCH_SPEEDUP if parallel > 1 else secs_seq
    print(f"estimated item count to generate: {total_n}; wall-clock sequential "
          f"{secs_seq / 60:.1f} min, batched --parallel {parallel}: {secs_batch / 60:.1f} min "
          f"({BATCH_SPEEDUP}x smoke-measured speedup)")
    return rep


def archive_old(dataset: str, backbone: str, split: str) -> Path | None:
    """Copy the CURRENT (pre-redraw) result JSON to ``<name>.pre-redraw-20260710.json`` -- append-
    only, never overwritten by a later invocation (idempotent: if the archive already exists, this
    is a no-op, so re-running this driver after a partial failure can never clobber the ORIGINAL
    pre-redraw content with an already-rewritten post-redraw file)."""
    p = result_path(dataset, backbone, split)
    if not p.exists():
        return None
    ap = archive_path(dataset, backbone, split)
    if not ap.exists():
        shutil.copy2(p, ap)
    return ap


def run_cell(dataset: str, split: str, parallel: int = 4, skip_checkpoint: bool = True) -> dict | None:
    """Run exactly one redraw cell via ``run_baseline.run_one(..., slice_mode="disjoint")`` /
    ``write_result`` (not reimplemented). Archives the pre-existing result JSON first (append-only,
    see ``archive_old``). NEVER raises -- catches and logs, mirroring wave1/wave2/wave3_cells.py's
    own per-cell guard."""
    manifest = load_manifest()
    need, reason = needs_rerun(dataset, split, manifest)
    if skip_checkpoint and not need:
        print(f"  [skip] {dataset}/{BACKBONE}/{split}: {reason}", flush=True)
        return None
    archived = archive_old(dataset, BACKBONE, split)
    if archived:
        print(f"  archived pre-redraw JSON -> {archived}", flush=True)
    print(f"=== redraw cell: {dataset} / {BACKBONE} / {split} (parallel={parallel}) ===", flush=True)
    try:
        result = rb.run_one(dataset, BACKBONE, split, None, parallel=parallel, slice_mode="disjoint")
    except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole redraw sweep
        print(f"  !! {dataset}/{split} FAILED: {type(e).__name__}: {e}", flush=True)
        return None
    p = rb.write_result(result)
    agg = result["aggregate"]
    print(f"  wrote {p} (mean={agg['mean']} ci={agg['ci95']} n_scored={agg['n_scored']}/{agg['n_total']} "
          f"elapsed_s={result['elapsed_s']} parallel={parallel})", flush=True)
    return result


def run_all(only_dataset: str | None = None, only_split: str | None = None,
            parallel: int = 4, skip_checkpoint: bool = True) -> dict:
    """Iterate every (dataset, split) cell needing rerun (or the one selected by
    ``only_dataset``/``only_split``). Assumes the GPU lock is already held and the resident
    qwen3-omni-30b-gguf llama-server is already up (see run_redraw.sh) -- mirrors wave1/2/3_cells.py's
    {ran, skipped, failed, total} exit-code contract exactly."""
    manifest = load_manifest()
    cells = all_cells(manifest)
    if only_dataset:
        assert only_dataset in manifest["datasets"], f"{only_dataset!r} is not in the redraw manifest"
        cells = [(dk, sp) for dk, sp in cells if dk == only_dataset]
    if only_split:
        cells = [(dk, sp) for dk, sp in cells if sp == only_split]

    t0 = time.time()
    n_ran = n_skipped = n_failed = 0
    for dk, sp in cells:
        before_need, _ = needs_rerun(dk, sp, manifest)
        result = run_cell(dk, sp, parallel=parallel, skip_checkpoint=skip_checkpoint)
        if result is not None:
            n_ran += 1
        elif not before_need:
            n_skipped += 1
        else:
            n_failed += 1
    print(f"=== redraw done in {time.time() - t0:.0f}s "
          f"(ran={n_ran} skipped={n_skipped} failed={n_failed} of {len(cells)}) ===", flush=True)
    # Machine-parseable twin -- run_redraw.sh greps this exact prefix, mirrors WAVE1/2/3_SUMMARY.
    print(f"REDRAW_SUMMARY ran={n_ran} skipped={n_skipped} failed={n_failed} total={len(cells)}", flush=True)
    return {"ran": n_ran, "skipped": n_skipped, "failed": n_failed, "total": len(cells)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset key (default: all 65)")
    ap.add_argument("--split", choices=SPLITS, default=None, help="restrict to one split (default: both)")
    ap.add_argument("--census", action="store_true", help="read-only rerun/skip table; no GPU/model calls")
    ap.add_argument("--dry-run", action="store_true", help="alias for --census")
    ap.add_argument("--execute", action="store_true",
                     help="actually run cells needing rerun -- requires the GPU session held + the "
                          "resident qwen3-omni-30b-gguf llama-server already up (see run_redraw.sh)")
    ap.add_argument("--no-checkpoint", action="store_true",
                     help="rerun the selected cell(s) even if id-sets already match (force)")
    ap.add_argument("--parallel", type=int, default=4,
                     help="client-side concurrent HTTP requests per cell (default 4, matches the "
                          "resident server's -np slot count)")
    args = ap.parse_args()

    if args.census or args.dry_run or not args.execute:
        print_census(parallel=args.parallel)
        return

    summary = run_all(only_dataset=args.dataset, only_split=args.split,
                       parallel=args.parallel, skip_checkpoint=not args.no_checkpoint)
    if summary["failed"] > 0:
        print(f"ERROR: redraw_cells.py --execute: {summary['failed']} cell(s) failed -- see FAILED "
              "lines above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

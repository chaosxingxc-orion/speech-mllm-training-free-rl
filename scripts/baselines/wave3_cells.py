"""scripts/baselines/wave3_cells.py — Wave-3 frozen cell list + checkpointed per-cell runner.

Wave-3 scope (mirrors wave-1/wave-2's structure exactly, see ``wave1_cells.py``/``wave2_cells.py``):
the LAST slice of the full Step-1 grid (``templates.DATASET_KTYPE``, 76 entries) not already covered
by wave-1 (K1+K2+K8+K9-squtr, 56 dataset keys) or wave-2 (K4+K5+K6+K7, 16 dataset keys) --

    K3 (multilingual LID+gender: fleurs-r)  +  K10 (spoken tool-calling: audio2tool)  +
    K11 (rule-verifiable: voicebench-advbench, voicebench-ifeval)

2026-07-10 reconciliation (task brief): 76 - 56 - 16 = 4 dataset keys remain, and
``wave3_dataset_keys()`` independently re-derives that SAME 4-key set by filtering
``templates.DATASET_KTYPE`` on K-type in {K3, K10, K11} -- the two counts agreeing is not a
coincidence, it is the closing check that wave-1 + wave-2 + wave-3 partition ALL 11 K-types
(K1..K11) with no gaps and no overlap. ``WAVE3_EXPECTED_N`` (4) documents/guards exactly that.

Unlike wave-1/wave-2 (dual-backbone: qwen3-omni-30b-gguf + meralion-2-gguf), wave-3's frozen cell
list is qwen3-omni-30b-gguf ONLY (task brief item 3: "4 datasets x qwen3-omni-30b-gguf x dev/test =
8 cells") -- a deliberate scope decision, not an oversight; ``WAVE3_BACKBONES`` documents it. A
future meralion-2-gguf wave-3b is not precluded by anything here, just out of THIS wave's scope.

**Batched inference (2026-07-10, owner-approved in the same breath as this wave's release -- see
task brief items 1-2): wave-3 cells run with ``--parallel WAVE3_PARALLEL`` (default 4) against a
resident qwen3-omni-30b-gguf server started with ``-np 4 -c 16384`` (see ``run_wave3.sh`` /
``gpu_session.sh``'s LLAMA_NP/LLAMA_CTX overrides). Smoke verdict backing this: mtmd-compatible,
0/24 greedy flips vs sequential, VRAM ~20.9GB, 1.75x wall-clock speedup at K=4 (no prompt-cache
reuse) -- see ``run_baseline.run_one``'s docstring for the same citation.** Unlike wave-2, this
wave has **NO WAVE3_RELEASE gate** -- the owner released the batching-run capability in the same
breath as approving the smoke test, so wave-3 is immediately executable (still requires the GPU
session lock + resident server, exactly like wave-1).

This module owns exactly the wave-3 CELL LIST + a checkpointed single-cell runner
(``run_cell``/``run_backbone``, thin wrappers around ``run_baseline.run_one``/``write_result`` --
NOT reimplemented, mirrors wave-1/wave-2). GPU-session lifecycle (acquire / resident-server up /
iterate / down / release) is ``run_wave3.sh``'s job, not this module's -- ``run_backbone``/
``run_cell`` assume the resident llama-server for the target backbone is already up (with the
-np/-c wave-3 config) and the GPU lock already held.

Usage:
  python wave3_cells.py --dry-run                                   # full wave-3 schedule + wall-clock estimate
  python wave3_cells.py --dataset fleurs-r --split dev --execute --parallel 4
                                                                       # validation cell (task brief step 4)
  python wave3_cells.py --execute                                    # RUNS every pending wave-3 cell (needs GPU
                                                                       # session + resident server up, -np 4 -c 16384)
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

# ---- frozen wave-3 backbone/split roster -- SINGLE backbone (see module docstring), unlike
# wave-1/wave-2's dual-backbone roster.
WAVE3_BACKBONES = ["qwen3-omni-30b-gguf"]
WAVE3_SPLITS = ["dev", "test"]

# Default client-side concurrency (concurrent.futures.ThreadPoolExecutor workers in
# run_baseline.run_one) -- matches the resident server's own -np 4 slot count (see run_wave3.sh),
# per the owner-approved smoke verdict cited in the module docstring. Overridable via --parallel.
WAVE3_PARALLEL = 4

# Per-generation wall-clock estimate (seconds/item), SEQUENTIAL baseline -- reuses wave1_cells.
# GEN_TIME_S verbatim (same qwen3-omni-30b-gguf generation path via run_baseline.generate/
# gen_llamacpp; wave-3-specific datasets do not change per-item latency). NOT independently
# re-measured for wave-3's own datasets -- update from a real wave-3 cell's recorded "elapsed_s"/
# n_scored once wave-3 actually runs, same discipline as wave1_cells.py's own note.
GEN_TIME_S = {"qwen3-omni-30b-gguf": 2.8, "meralion-2-gguf": 1.0}

# Empirically-measured wall-clock speedup at K=4 parallel (owner-approved smoke, 2026-07-10, no
# prompt-cache reuse) -- 1.75x, NOT a naive 4x: batching overlaps network+generation round-trips but
# generation itself is still compute-bound on one GPU, so the four requests do not run in
# fully-independent lockstep. Used ONLY for --dry-run's estimate; the real number comes from each
# cell's own recorded "elapsed_s" once wave-3 actually runs (see step 4 of the task brief).
WAVE3_BATCH_SPEEDUP = 1.75

# The frozen wave-3 K-type family + expected dataset-key count (see module docstring) -- an
# explicit constant so wave3_dataset_keys()'s assertion documents exactly what it is protecting.
WAVE3_KTYPES = ("K3", "K10", "K11")
WAVE3_EXPECTED_N = 4


def wave3_dataset_keys() -> list[str]:
    """The frozen wave-3 dataset-key list: every ``templates.DATASET_KTYPE`` entry whose K-type is
    K3, K10, or K11 (see ``WAVE3_KTYPES``), sorted for a deterministic, reviewable schedule order.

    Asserts the count is exactly ``WAVE3_EXPECTED_N`` (4, verified 2026-07-10: 1 K3 + 1 K10 + 2 K11
    -- fleurs-r, audio2tool, voicebench-advbench, voicebench-ifeval) so a future edit to
    ``templates.DATASET_KTYPE`` cannot silently change wave-3's frozen scope without this module
    noticing. This ALSO closes the wave-1/wave-2/wave-3 reconciliation (module docstring): asserting
    here that wave-1 (56) + wave-2 (16) + wave-3 (4) == the full grid (76) with no overlap.
    """
    keys = {k for k, v in templates.DATASET_KTYPE.items() if v in WAVE3_KTYPES}
    assert len(keys) == WAVE3_EXPECTED_N, (
        f"wave3_dataset_keys: expected {WAVE3_EXPECTED_N} K3/K10/K11 dataset keys, found {len(keys)}: "
        f"{sorted(keys)} -- templates.DATASET_KTYPE changed since wave-3 was frozen (2026-07-10); "
        "re-check the reconciliation in this module's docstring before proceeding."
    )
    return sorted(keys)


def wave3_cells(backbones: list[str] | None = None, splits: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(dataset, backbone, split) triples in a fixed, deterministic order: dataset-major, then
    backbone, then split -- so ``--dry-run``'s printed schedule and ``--execute``'s actual run
    order are the same thing, and a resumed run after a partial failure iterates identically."""
    backbones = backbones or WAVE3_BACKBONES
    splits = splits or WAVE3_SPLITS
    return [(dk, bb, sp) for dk in wave3_dataset_keys() for bb in backbones for sp in splits]


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


def estimate_wallclock_s(cells: list[tuple[str, str, str]], parallel: int = 1) -> dict:
    """Wall-clock estimate. ``parallel<=1`` is the plain sequential sum (matches wave1/wave2_cells'
    own estimator exactly). ``parallel>1`` divides by ``WAVE3_BATCH_SPEEDUP`` (1.75x, NOT a naive
    linear ``/parallel``) -- see that constant's comment for why."""
    per_backbone: dict[str, float] = {}
    total = 0.0
    for dk, bb, sp in cells:
        secs = n_for(sp) * GEN_TIME_S.get(bb, 2.0)
        per_backbone[bb] = per_backbone.get(bb, 0.0) + secs
        total += secs
    if parallel > 1:
        total /= WAVE3_BATCH_SPEEDUP
        per_backbone = {bb: s / WAVE3_BATCH_SPEEDUP for bb, s in per_backbone.items()}
    return {"total_s": total, "per_backbone_s": per_backbone}


def print_schedule(backbones: list[str] | None = None, splits: list[str] | None = None,
                    only_dataset: str | None = None, parallel: int = WAVE3_PARALLEL) -> None:
    backbones = backbones or WAVE3_BACKBONES
    splits = splits or WAVE3_SPLITS
    datasets = wave3_dataset_keys()
    if only_dataset is not None:
        assert only_dataset in datasets, f"{only_dataset!r} is not a wave3 dataset key ({len(datasets)} total)"
        datasets = [only_dataset]
    cells = [(dk, bb, sp) for dk in datasets for bb in backbones for sp in splits]
    est_seq = estimate_wallclock_s(cells, parallel=1)
    est_batch = estimate_wallclock_s(cells, parallel=parallel)

    print(f"=== Wave-3 frozen cell list: {len(datasets)} datasets x {len(backbones)} backbones "
          f"x {len(splits)} splits = {len(cells)} cells (--parallel {parallel}) ===")
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
        secs_seq = est_seq["per_backbone_s"].get(bb, 0.0)
        secs_batch = est_batch["per_backbone_s"].get(bb, 0.0)
        print(f"estimated wall-clock, {bb}: sequential {secs_seq / 60:.1f} min ({secs_seq:.0f}s) "
              f"@ {GEN_TIME_S.get(bb, 2.0)}s/gen; batched --parallel {parallel}: "
              f"{secs_batch / 60:.1f} min ({secs_batch:.0f}s) @ {WAVE3_BATCH_SPEEDUP}x smoke-measured speedup")
    print(f"estimated wall-clock, TOTAL (all backbones, full wave-3 grid, ignoring checkpoints): "
          f"sequential {est_seq['total_s'] / 60:.1f} min ({est_seq['total_s'] / 3600:.2f} h); "
          f"batched --parallel {parallel}: {est_batch['total_s'] / 60:.1f} min "
          f"({est_batch['total_s'] / 3600:.2f} h)")
    print("\nNOTE: unlike wave-2, wave-3 has NO release gate -- owner released batched-inference "
          "execution in the same breath as approving the smoke test (see module docstring).")


def run_cell(dataset: str, backbone: str, split: str, skip_checkpoint: bool = True,
             parallel: int = WAVE3_PARALLEL) -> dict | None:
    """Run exactly one wave-3 cell via ``run_baseline.run_one``/``write_result`` (not
    reimplemented), with checkpoint-skip. NEVER raises -- catches and logs, mirroring
    ``run_baseline.main()``'s own per-cell guard, so one bad/env-blocked cell cannot sink the rest
    of the sweep. ``parallel`` is threaded straight to ``run_one`` (2026-07-10 batched-inference
    task brief item 1) -- see that function's docstring for the scheduling semantics.
    """
    if skip_checkpoint and is_checkpointed(dataset, backbone, split):
        print(f"  [skip] {dataset}/{backbone}/{split} already checkpointed at "
              f"{result_path(dataset, backbone, split)}", flush=True)
        return None
    print(f"=== wave3 cell: {dataset} / {backbone} / {split} (parallel={parallel}) ===", flush=True)
    try:
        result = rb.run_one(dataset, backbone, split, None, parallel=parallel)
    except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole wave-3 sweep
        print(f"  !! {dataset}/{backbone}/{split} FAILED: {type(e).__name__}: {e}", flush=True)
        return None
    p = rb.write_result(result)
    agg = result["aggregate"]
    print(f"  wrote {p} (mean={agg['mean']} ci={agg['ci95']} n_scored={agg['n_scored']}/{agg['n_total']} "
          f"elapsed_s={result['elapsed_s']} parallel={parallel})", flush=True)
    return result


def run_backbone(backbone: str, splits: list[str] | None = None, only_dataset: str | None = None,
                  parallel: int = WAVE3_PARALLEL) -> dict:
    """Iterate every wave-3 (dataset, split) cell for ONE backbone. Assumes the GPU lock is
    already held and that backbone's resident llama-server is already up (with the wave-3
    ``-np``/``-c`` config, see run_wave3.sh) -- mirrors wave1_cells.run_backbone /
    wave2_cells.run_backbone's {ran, skipped, failed, total} exit-code contract exactly."""
    datasets = [only_dataset] if only_dataset else wave3_dataset_keys()
    if only_dataset:
        assert only_dataset in wave3_dataset_keys(), f"{only_dataset!r} is not a wave3 dataset key"
    cells = [(dk, backbone, sp) for dk in datasets for sp in (splits or WAVE3_SPLITS)]
    t0 = time.time()
    n_ran = n_skipped = n_failed = 0
    for dk, bb, sp in cells:
        before = is_checkpointed(dk, bb, sp)
        result = run_cell(dk, bb, sp, parallel=parallel)
        if result is not None:
            n_ran += 1
        elif before:
            n_skipped += 1
        else:
            n_failed += 1
    print(f"=== backbone {backbone} done in {time.time() - t0:.0f}s "
          f"(ran={n_ran} skipped={n_skipped} failed={n_failed} of {len(cells)}) ===", flush=True)
    # Machine-parseable twin of the line above -- run_wave3.sh greps this exact prefix out of the
    # tee'd log to print its own per-backbone summary and decide whether to exit non-zero. Keep the
    # key set (ran/skipped/failed/total) and "key=value" shape stable; run_wave3.sh's extract_field
    # parses it with plain shell parameter expansion (no PCRE dependency) -- mirrors
    # WAVE1_SUMMARY/WAVE2_SUMMARY.
    print(f"WAVE3_SUMMARY backbone={backbone} ran={n_ran} skipped={n_skipped} failed={n_failed} "
          f"total={len(cells)}", flush=True)
    return {"ran": n_ran, "skipped": n_skipped, "failed": n_failed, "total": len(cells)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=WAVE3_BACKBONES, default=None,
                     help="restrict to one backbone (default: the single wave-3 backbone)")
    ap.add_argument("--split", choices=WAVE3_SPLITS, default=None,
                     help="restrict to one split (default: both)")
    ap.add_argument("--dataset", default=None,
                     help="restrict to one wave3 dataset key (default: the whole frozen list)")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule + wall-clock estimate; no model calls")
    ap.add_argument("--execute", action="store_true",
                     help="actually run the cells -- requires the GPU session held + the resident "
                          "qwen3-omni-30b-gguf llama-server already up with the wave-3 -np/-c "
                          "config (see run_wave3.sh). NO owner-release env-var gate (unlike wave-2)"
                          " -- see module docstring.")
    ap.add_argument("--no-checkpoint", action="store_true",
                     help="ignore existing result JSONs and rerun every selected cell (default: skip checkpointed cells)")
    ap.add_argument("--parallel", type=int, default=WAVE3_PARALLEL,
                     help=f"client-side concurrent HTTP requests per cell (default {WAVE3_PARALLEL}, "
                          "matching the resident server's -np slot count -- see run_one's docstring)")
    args = ap.parse_args()

    backbones = [args.backbone] if args.backbone else WAVE3_BACKBONES
    splits = [args.split] if args.split else WAVE3_SPLITS

    if args.execute:
        total_failed = 0
        for bb in backbones:
            if args.no_checkpoint:
                cells = [(args.dataset, bb, sp) for sp in splits] if args.dataset else wave3_cells([bb], splits)
                n_ran = n_failed = 0
                for dk, b, sp in cells:
                    # skip_checkpoint=False means run_cell() never skips -- None here always means
                    # the cell raised (caught inside run_cell), never "already checkpointed".
                    result = run_cell(dk, b, sp, skip_checkpoint=False, parallel=args.parallel)
                    n_ran += 1 if result is not None else 0
                    n_failed += 0 if result is not None else 1
                print(f"WAVE3_SUMMARY backbone={bb} ran={n_ran} skipped=0 failed={n_failed} "
                      f"total={len(cells)}", flush=True)
                total_failed += n_failed
            else:
                summary = run_backbone(bb, splits, only_dataset=args.dataset, parallel=args.parallel)
                total_failed += summary["failed"]
        if total_failed > 0:
            print(f"ERROR: wave3_cells.py --execute: {total_failed} cell(s) failed across "
                  f"{len(backbones)} backbone(s) -- see FAILED lines above.", file=sys.stderr)
            sys.exit(1)
        return

    # default (including --dry-run): print the schedule, never touch a model/GPU.
    print_schedule(backbones, splits, only_dataset=args.dataset, parallel=args.parallel)


if __name__ == "__main__":
    main()

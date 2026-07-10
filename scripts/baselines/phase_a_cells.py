"""scripts/baselines/phase_a_cells.py — Step-2 Phase-A marginal-scan schedule (freeze-independent).

Enumerates ``wiki/2026-07-10-step2-grid-draft.md`` §3's Phase-A schedule: "each dimension swept
one arm at a time, the other five pinned at ref-config" (`run_mock.REF_CONFIG`), crossed with the
four main-decision-field datasets (`wiki/2026-07-09-stage1-three-step-experiment-design.md` line
48: squtr / heysquad(-scrubbed) / SQuAD-zh / vocalbench-knowledge) and the dev split only, qwen3
backbone only (owner ruling, grid draft §6 sign-off item 6: "Phase A 先行基座=qwen3 单基座（省一半，
MERaLiON 在 Phase B 补入）").

This module owns exactly the CELL LIST + a checkpointed single-cell runner shape (mirrors
``wave1_cells.py``'s own split of responsibility: schedule/estimate here, GPU-session
acquire/serve/release lifecycle is the CALLER's job, not this module's). **Execution is
deliberately NOT wired here** -- the grid draft's own §6 sign-off items (ref-config, dimension
enumeration/contrast-arm choices, the ~450-cell budget, the H-a/H-b adjudication protocol) are
still open as of this task, so a Phase-A "--execute" path would have nothing frozen to run
against yet. Once the freeze meeting signs off, wiring one is a thin ``run_mock.run_one_mock`` /
``run_mock.write_result_mock`` wrapper, exactly like ``wave1_cells.run_cell`` wraps
``run_baseline.run_one``/``write_result``.

Usage:
  python phase_a_cells.py --dry-run                      # full Phase-A schedule + wall-clock estimate
  python phase_a_cells.py --dataset squtr --dry-run       # one dataset's arm list only
  python phase_a_cells.py --dim embedder --dry-run        # one dimension's arm list only
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/baselines

import run_mock as rm  # noqa: E402  (reuses MockConfig/RetrievalConfig/REF_CONFIG/enums/result_path_mock)

# ---- frozen Phase-A dataset/backbone/split roster --------------------------------------------
# Main-decision-field datasets, per wiki/2026-07-09-stage1-three-step-experiment-design.md line 48
# ("主裁决场（squtr+heysquad-scrubbed+SQuAD-zh+vocalbench-knowledge）"). "heysquad-scrubbed" is a
# KB-BUILD-time property (kb_build.build_source(scrub=True), per heysquad.py's own LEAKAGE WARNING
# and templates.DATASET_KTYPE's "B4-recovery" note) -- NOT a MockConfig field, so it never appears
# in run_mock.source_name_for; whoever builds heysquad's KB source(s) for real MUST pass
# scrub=True unconditionally, or every heysquad Phase-A cell here is inadmissible by construction
# (kb_retrieve.load_source's own leakage gate would refuse an unclean source anyway).
PHASE_A_DATASETS = ["squtr", "heysquad", "SQuAD-zh", "vocalbench-knowledge"]
PHASE_A_BACKBONE = "qwen3-omni-30b-gguf"  # owner ruling, grid draft §6.6 -- MERaLiON added at Phase B
PHASE_A_SPLIT = "dev"                     # grid draft §3: "主裁决场，dev n=40"
PHASE_A_N = 40                            # == run_baseline.DEV_N, restated here for readability

# Owner-supplied wall-clock estimate (this task's brief): "server-resident assumption, ~3s/gen +
# retrieval overhead ~0.2s" -- NOT independently measured on this box (no cell has been executed
# yet, see module docstring); update from a real cell's recorded elapsed_s once Phase-A actually
# runs, mirroring wave1_cells.GEN_TIME_S's own "unverified, update once available" caveat.
GEN_TIME_S = 3.0
RETRIEVAL_OVERHEAD_S = 0.2
PER_ITEM_S = GEN_TIME_S + RETRIEVAL_OVERHEAD_S


def build_phase_a_arms() -> list[tuple[str, str, "rm.MockConfig"]]:
    """One-factor-at-a-time marginal-scan arms: for each of the 6 grid dimensions, hold the other
    5 at REF_CONFIG and vary this ONE across its full named-arm list (``run_mock.EMBEDDERS`` /
    ``KEY_ORGS`` / ``VALUE_ORGS`` / ``RETRIEVAL_KINDS`` / ``QUERY_CONSTRUCTIONS`` /
    ``DELIVERY_MODES``). Returns (dim_name, arm_label, MockConfig) triples in a fixed order
    (dimension-major, in the §2 enumeration's own order) so ``--dry-run``'s printed schedule is
    always the same list.

    **Counted totals**: 8 embedder + 4 key_org + 5 value_org + 7 retrieval + 4 query_construction
    + 6 delivery = **34** here, ONE OFF the wiki's own stated "8+4+6+7+4+6 = 35". Traced to
    value_org: the grid draft's §2 prose only NAMES 5 distinct value-org tokens under a
    "(4+2 对照)" header that implies 6 (one contrast arm -- RAPTOR-lite vs. HippoRAG-lite -- is an
    explicit 二选一 per §6.2, i.e. a single chosen arm here, not two). This is an OPEN freeze
    sign-off item in the source doc itself (a draft, §6 not yet signed), not silently resolved to
    force a match here -- see ``run_mock.VALUE_ORGS``'s own comment and ``print_schedule``'s
    printed reconciliation line below.
    """
    arms: list[tuple[str, str, rm.MockConfig]] = []
    for val in rm.EMBEDDERS:
        arms.append(("embedder", val, dataclasses.replace(rm.REF_CONFIG, embedder=val)))
    for val in rm.KEY_ORGS:
        arms.append(("key_org", val, dataclasses.replace(rm.REF_CONFIG, key_org=val)))
    for val in rm.VALUE_ORGS:
        arms.append(("value_org", val, dataclasses.replace(rm.REF_CONFIG, value_org=val)))
    for val in rm.RETRIEVAL_KINDS:
        rc = dataclasses.replace(rm.REF_CONFIG.retrieval, kind=val)
        arms.append(("retrieval", val, dataclasses.replace(rm.REF_CONFIG, retrieval=rc)))
    for val in rm.QUERY_CONSTRUCTIONS:
        arms.append(("query_construction", val, dataclasses.replace(rm.REF_CONFIG, query_construction=val)))
    for val in rm.DELIVERY_MODES:
        arms.append(("delivery", val, dataclasses.replace(rm.REF_CONFIG, delivery=val)))
    return arms


# Per-dimension (computed arm count, wiki-stated arm count) -- printed by print_schedule so any
# mismatch is visible every time the schedule renders, never silently absorbed.
_DOC_STATED_COUNTS = {
    "embedder": 8, "key_org": 4, "value_org": 6, "retrieval": 7,
    "query_construction": 4, "delivery": 6,
}


def build_phase_a_cells(datasets: list[str] | None = None) -> list[dict]:
    """Cross the Phase-A arms with the main-field datasets (dev split, qwen3 backbone only)."""
    datasets = datasets or PHASE_A_DATASETS
    arms = build_phase_a_arms()
    cells = []
    for dataset in datasets:
        for dim_name, arm_label, cfg in arms:
            cells.append({
                "dataset": dataset, "backbone": PHASE_A_BACKBONE, "split": PHASE_A_SPLIT,
                "dim": dim_name, "arm": arm_label, "config": cfg,
                "result_path": rm.result_path_mock(dataset, PHASE_A_BACKBONE, PHASE_A_SPLIT, cfg),
            })
    return cells


def is_checkpointed(cell: dict) -> bool:
    """True iff a COMPLETE result JSON already exists for this cell (has a non-null "aggregate"),
    mirroring wave1_cells.is_checkpointed's exact contract (missing/unreadable/partial -> NOT
    checkpointed, so a re-invocation safely retries it)."""
    import json

    p = cell["result_path"]
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("aggregate"))


def estimate_wallclock_s(cells: list[dict]) -> float:
    return len(cells) * PHASE_A_N * PER_ITEM_S


def print_schedule(dataset_filter: str | None = None, dim_filter: str | None = None) -> None:
    datasets = [dataset_filter] if dataset_filter else PHASE_A_DATASETS
    if dataset_filter:
        assert dataset_filter in PHASE_A_DATASETS, (
            f"{dataset_filter!r} is not a Phase-A main-field dataset ({PHASE_A_DATASETS})"
        )

    arms = build_phase_a_arms()
    if dim_filter:
        assert dim_filter in _DOC_STATED_COUNTS, f"{dim_filter!r} is not a grid dimension ({sorted(_DOC_STATED_COUNTS)})"
        arms = [a for a in arms if a[0] == dim_filter]

    # per-dimension computed-vs-doc-stated arm counts (printed every time, see build_phase_a_arms docstring)
    computed_counts: dict[str, int] = {}
    for dim_name, _arm, _cfg in build_phase_a_arms():
        computed_counts[dim_name] = computed_counts.get(dim_name, 0) + 1
    computed_total = sum(computed_counts.values())
    doc_total = sum(_DOC_STATED_COUNTS.values())

    cells = [c for c in build_phase_a_cells(datasets) if (dim_filter is None or c["dim"] == dim_filter)]
    est_s = estimate_wallclock_s(cells)
    n_pending = sum(0 if is_checkpointed(c) else 1 for c in cells)

    print("=== Step-2 Phase-A marginal-scan schedule (freeze-independent, NOT yet run) ===")
    print(f"ref-config: {rm.REF_CONFIG}")
    print(f"datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"backbone: {PHASE_A_BACKBONE}  split: {PHASE_A_SPLIT}  n/cell: {PHASE_A_N}")
    print()
    print("per-dimension arm counts (computed here vs. wiki §3's stated 8+4+6+7+4+6=35):")
    for dim_name in ("embedder", "key_org", "value_org", "retrieval", "query_construction", "delivery"):
        computed = computed_counts.get(dim_name, 0)
        stated = _DOC_STATED_COUNTS[dim_name]
        flag = "" if computed == stated else "  <-- MISMATCH, see build_phase_a_arms docstring"
        print(f"  {dim_name:20s} computed={computed}  doc-stated={stated}{flag}")
    print(f"  {'TOTAL':20s} computed={computed_total}  doc-stated={doc_total}"
          f"{'' if computed_total == doc_total else '  <-- see value_org mismatch above'}")
    print()
    for dim_name, arm_label, cfg in arms:
        print(f"  [{dim_name:20s}] {arm_label}")
    print()
    print(f"cells = {len(arms)} arms x {len(datasets)} dataset(s) "
          f"= {len(cells)} cells" + (f" (dim={dim_filter!r} only)" if dim_filter else ""))
    print(f"{len(cells) - n_pending}/{len(cells)} already checkpointed; {n_pending} pending.")
    print(f"estimated wall-clock: {est_s / 60:.1f} min ({est_s / 3600:.2f} h) "
          f"@ {PER_ITEM_S:.1f}s/item ({GEN_TIME_S}s gen + {RETRIEVAL_OVERHEAD_S}s retrieval, "
          "server-resident assumption -- unverified on this box, see module docstring)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, choices=PHASE_A_DATASETS,
                     help="restrict the schedule to one main-field dataset (default: all 4)")
    ap.add_argument("--dim", default=None,
                     choices=["embedder", "key_org", "value_org", "retrieval", "query_construction", "delivery"],
                     help="restrict the schedule to one grid dimension's arms (default: all 6)")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the schedule + wall-clock estimate (the only action this module "
                          "performs -- see module docstring: execution is not wired pre-freeze)")
    args = ap.parse_args()

    # --dry-run is the default AND only behavior right now (see module docstring) -- accepting the
    # flag either way keeps this CLI's shape parallel to wave1_cells.py's, without pretending an
    # --execute path exists before the freeze meeting has actually signed off anything.
    print_schedule(dataset_filter=args.dataset, dim_filter=args.dim)


if __name__ == "__main__":
    main()

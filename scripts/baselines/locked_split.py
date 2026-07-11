"""scripts/baselines/locked_split.py — group-aware LOCKED test/dev manifest generator (ticket
#26, design doc wiki/2026-07-11-group-split-statistics-design.md §2).

## Why this exists (finding #2, re-stated)

``scripts/baselines/redraw.py``'s disjoint-redraw draws ``test`` with ``TEST_SEED`` (20261705) --
the SAME seed the already-scored, already-decision-informing wave-1/2 grid used. A holdout is only
a real holdout if its membership has NEVER informed a decision (Dwork et al. reusable-holdout
problem); ``TEST_SEED`` fails that bar regardless of how disjoint dev/test are from each other. So
does every id ever exposed in ``_repro/baselines/*.json`` or ``_repro/redraw_manifest.json`` -- ALL
of that is **permanently non-locked** (see ``_repro/LOCKED_HOLDOUT/README.md``, written by this
module's ``--write-readme``).

This module draws a genuinely fresh (never-before-used, owner-ratified Decision-Log 续13)
``LOCKED_TEST_SEED``, group-aware (``group_key.group_key_of`` + ``_common.draw_disjoint_grouped``,
both already committed -- ticket #25/earlier-#26 prep, NOT written by this module), and writes ONE
manifest file per dataset key under ``_repro/LOCKED_HOLDOUT/<dataset_key>.json``.

## Access-control convention (process, not filesystem ACL -- owner-ratified 续13: "file convention +
append-only access log, no encryption")

The files under ``_repro/LOCKED_HOLDOUT/`` may be read by **exactly one** consumer: the final
confirmatory scoring pass. No arm/prompt/threshold selection may read ``test_ids`` from here --
see ``_repro/LOCKED_HOLDOUT/README.md`` (``--write-readme``) for the full rule, and
``_repro/LOCKED_HOLDOUT/ACCESS_LOG.md`` (``--write-access-log-header``) for the append-only log
every read of this directory should add a line to (by convention -- this module does not enforce
it mechanically, per the owner's chosen "no encryption" tradeoff).

## Pool reconstruction (reused verbatim, not reimplemented)

Every dataset's full natural pool is fetched via ``redraw.full_pool_rows(dataset_key)`` -- the
SAME audio-decode-stubbed, snapshot-write-suppressed pool reconstruction ``redraw.py`` already
uses (``rb._load_rows(dataset_key, "test", n=None, seed=POOL_RECONSTRUCTION_SEED)`` under
``redraw._stub_audio_materialization()``), so this module never re-derives that logic and stays
CPU-only / zero-model-call by construction (no ``gen()``/``generate()`` call anywhere here).

Usage:
  python locked_split.py --dry-run                              # every ~65 redraw-scope key, no writes
  python locked_split.py --dataset aishell-1                     # ONE dataset's manifest
  python locked_split.py                                          # generate all ~65 (writes files)
  python locked_split.py --write-readme --write-access-log-header # (re)write the two convention docs
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # scripts/baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # scripts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "loaders"))

import redraw                          # noqa: E402  (scripts/baselines/redraw.py -- reuse full_pool_rows)
import run_baseline as rb              # noqa: E402
from group_key import group_key_of     # noqa: E402  (scripts/loaders/group_key.py)
from _common import draw_disjoint_grouped  # noqa: E402  (scripts/loaders/_common.py)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = REPO_ROOT / "_repro" / "LOCKED_HOLDOUT"
REDRAW_MANIFEST_PATH = REPO_ROOT / "_repro" / "redraw_manifest.json"
DESIGN_DOC = "wiki/2026-07-11-group-split-statistics-design.md"

# ---------------------------------------------------------------------------------------------
# Owner-ratified constants (Decision-Log 续13, 2026-07-11): LOCKED_TEST_SEED is a brand-new,
# never-burned seed (distinct from SLICE_SEED=20260705, TEST_SEED=20261705, COHORT_SEED=20260711,
# NOISE_SEED=20260712 -- see design doc §2.1's "burned seeds" audit). draw_disjoint_grouped derives
# the DEV draw internally as `seed + 1` (see that function's docstring) -- the owner explicitly
# adopted THIS "(dev seed +1)" convention over the design doc's originally sketched separate
# LOCKED_DEV_SEED=611_741_213 constant (§2.1's sketch is superseded here; only ONE seed constant
# exists in this module).
# ---------------------------------------------------------------------------------------------
LOCKED_TEST_SEED = 611_741_209

TEST_N = 60   # matches the existing draw_disjoint/redraw.py 60/40 convention
DEV_N = 40

SPLIT_TAG = "grouplock-v2-20260711"

# ---------------------------------------------------------------------------------------------
# human-readable group-unit labels (design doc §1.3's "group unit" column) -- cosmetic provenance
# ONLY, never consulted by the split algorithm itself (group_key_of is the single source of truth
# for the actual group id). Keys not listed here fall back to a generic label (see
# group_unit_label below); G-NONE / item-fallback datasets get their own generic label regardless
# of whether they appear here.
# ---------------------------------------------------------------------------------------------
GROUP_UNIT_LABELS: dict[str, str] = {
    "librispeech": "speaker",
    "aishell-1": "speaker",
    "thchs-30": "speaker (reader)",
    "voicebench-sd-qa": "source question (same Q, multiple dialect renderings)",
    "squtr": "query (clean + SNR-augmented renderings share one group)",
    "crema-d": "speaker",
    "esd": "speaker",
    "csemotions": "speaker",
    "meld": "dialogue",
    "voicebench-mmsu-spoken": "source doc / domain",
    "mmsu": "task_name",
    "vocalbench-knowledge": "source / topic",
    "vocalbench-reasoning": "source / category",
    "vocalbench-multi-round": "category",
    "voiceassistant-listening-general": "category1/category2",
    "voiceassistant-listening-music": "category1/category2",
    "voiceassistant-listening-sound": "category1/category2",
    "voiceassistant-listening-speech": "category1/category2",
    "voiceassistant-speaking-reasoning": "category1/category2",
    "slurp": "scenario",
    "slurp-slot": "scenario",
    "heysquad": "SQuAD passage (hash of context)",
    "voicebench-bbh": "BBH subtask",
    "voicebench-ifeval": "instruction-type set (moot -- score always None at Step-1)",
    "audio2tool": "query_idx (one query rendered by many speakers)",
    "mmau-mini": "audio clip (audio_id) -- 1:1 with items on the current test_mini mirror, see "
                 "group_key.py's comment",
}
for _loc in ("de-DE", "fr-FR"):
    GROUP_UNIT_LABELS[f"speech-massive-{_loc}-attr"] = "speaker (speaker_id)"
    GROUP_UNIT_LABELS[f"speech-massive-{_loc}"] = "scenario"
    GROUP_UNIT_LABELS[f"speech-massive-{_loc}-slot"] = "scenario"
for _aqa in ("air-bench-foundation-sound-aqa-avqa", "air-bench-foundation-sound-aqa-clothoaqa",
             "air-bench-foundation-music-aqa"):
    GROUP_UNIT_LABELS[_aqa] = "audio clip (clip_id)"

ITEM_FALLBACK_LABEL = "item (no group metadata at any finer grain -- design doc §1.4 honest fallback)"
DEGENERATE_FALLBACK_LABEL = (
    "item (grouping degenerate: the dataset's ONE real group spans the whole pool -- e.g. a "
    "per-category voiceassistant-* grid key whose category1/category2 is constant -- so a "
    "group-disjoint split is undefined; forced item-level fallback, see "
    "_common.draw_disjoint_grouped's degenerate_single_group handling, 2026-07-11)"
)


def group_unit_label(dataset_key: str, fallback_item_level: bool,
                      degenerate_single_group: "str | None" = None) -> str:
    if degenerate_single_group is not None:
        return DEGENERATE_FALLBACK_LABEL
    if fallback_item_level:
        return ITEM_FALLBACK_LABEL
    return GROUP_UNIT_LABELS.get(dataset_key, "group (see scripts/loaders/group_key.py:group_key_of)")


# ---------------------------------------------------------------------------------------------
# "old exposed test ids" -- the honesty-metric comparator (Task 3's verify step)
# ---------------------------------------------------------------------------------------------

_BACKBONES_FOR_EXPOSURE_CHECK = ("qwen3-omni-30b-gguf", "meralion-2-gguf")


def old_exposed_test_ids(dataset_key: str) -> set[str]:
    """Union of every test-split item id this dataset key has EVER exposed BEFORE this locked
    manifest -- design doc finding #2: the original (still-live) ``_repro/baselines/<key>__
    <backbone>__test.json`` cells, their pre-redraw archived sidecars
    (``redraw_cells.ARCHIVE_TAG``), AND ``_repro/redraw_manifest.json``'s own ``new_test_ids`` (the
    wave-1/2 disjoint redraw's test draw -- also exposed, since ``TEST_SEED`` never changed). ALL
    of these are permanently non-locked; this function's ONLY purpose is to compute an honest
    overlap-fraction against the fresh ``LOCKED_TEST_SEED`` draw below -- read-only, no loader call.
    """
    ids: set[str] = set()
    for backbone in _BACKBONES_FOR_EXPOSURE_CHECK:
        for suffix in ("test.json", "test.pre-redraw-20260710.json"):
            p = rb.OUT_DIR / f"{dataset_key}__{backbone}__{suffix}"
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            ids.update(it.get("item_id") for it in data.get("per_item", []) if it.get("item_id") is not None)
    if REDRAW_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(REDRAW_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
        entry = manifest.get("datasets", {}).get(dataset_key, {})
        ids.update(entry.get("new_test_ids") or [])
    return ids


# ---------------------------------------------------------------------------------------------
# redraw-scope dataset keys (the ~65 grid keys this ticket's rerun scope targets)
# ---------------------------------------------------------------------------------------------

def redraw_scope_dataset_keys() -> list[str]:
    """The ~65 dataset keys ``_repro/redraw_manifest.json`` already covers (52 wave-1 + 13 wave-2,
    design doc §0 finding #2 / §4.2) -- reused verbatim as the default generation scope (ticket #26
    task 3: "run locked_split.py for the design's rerun scope (the ~65 grid dataset keys)"), not
    re-derived from ``wave1_cells``/``wave2_cells`` here (this file's dataset list IS the
    authoritative "already non-locked, needs a locked replacement" set)."""
    manifest = json.loads(REDRAW_MANIFEST_PATH.read_text(encoding="utf-8"))
    return sorted(manifest["datasets"].keys())


# ---------------------------------------------------------------------------------------------
# per-dataset manifest generation
# ---------------------------------------------------------------------------------------------

def generate_one(dataset_key: str, timestamp: str, n_test: int = TEST_N, n_dev: int = DEV_N,
                  seed: int = LOCKED_TEST_SEED) -> dict:
    """Build (but do not write) the locked manifest dict for one dataset key. Pure CPU: pool
    reconstruction is audio-decode-stubbed (``redraw.full_pool_rows``), the draw itself is pure
    numpy (``draw_disjoint_grouped``) -- no model/GPU call anywhere in this function."""
    rows = redraw.full_pool_rows(dataset_key)
    group_fn = functools.partial(group_key_of, dataset_key)
    drawn = draw_disjoint_grouped(rows, group_fn, n_test=n_test, n_dev=n_dev, seed=seed)

    old_exposed = old_exposed_test_ids(dataset_key)
    overlap = sorted(set(drawn["test_ids"]) & old_exposed)
    overlap_fraction = (len(overlap) / len(drawn["test_ids"])) if drawn["test_ids"] else None

    return {
        "dataset": dataset_key,
        # ---- the ticket's required minimal field set ----
        "test_ids": drawn["test_ids"],
        "dev_ids": drawn["dev_ids"],
        "group_unit": group_unit_label(dataset_key, drawn["fallback_item_level"],
                                        drawn["degenerate_single_group"]),
        "group_disjoint": not drawn["fallback_item_level"],
        "seed": seed,
        "created_at": timestamp,
        # ---- additive honest provenance (design doc §2.2/§2.4) ----
        "n_test": drawn["n_test"], "n_dev": drawn["n_dev"],
        "n_groups_total": drawn["n_groups_total"],
        "group_disjoint_verified": drawn["group_disjoint_verified"],
        "shortfall": drawn["shortfall"],
        "oversized_group": drawn["oversized_group"],
        "degenerate_single_group": drawn["degenerate_single_group"],
        "pool_size": len(rows),
        "old_exposed_test_ids": {
            "source": ("union of _repro/baselines/<key>__{qwen3-omni-30b-gguf,meralion-2-gguf}__"
                       "test[.pre-redraw-20260710].json per_item ids + "
                       "_repro/redraw_manifest.json's new_test_ids (design doc finding #2: ALL "
                       "permanently non-locked)"),
            "n_old_exposed": len(old_exposed),
            "overlap_count": len(overlap),
            "overlap_fraction": round(overlap_fraction, 4) if overlap_fraction is not None else None,
            "note": ("nonzero overlap for a small pool is EXPECTED and honest -- the point of this "
                     "manifest is a fresh never-used seed + group-awareness + access discipline, "
                     "NOT zero overlap by construction (ticket #26 task 3 verify step)."),
        },
        "locked_test_seed": seed,
        "split_tag": SPLIT_TAG,
        "source": "scripts/baselines/locked_split.py",
        "design_doc": f"{DESIGN_DOC} §2",
    }


def write_manifest(manifest: dict) -> Path:
    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    p = LOCKED_DIR / f"{manifest['dataset']}.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_locked_test(dataset_key: str) -> list[str]:
    """Read back a locked manifest's ``test_ids`` -- the ONE function a confirmatory scoring pass
    should ever call to touch ``test_ids`` (design doc §2.3's enforcement sketch (b): "a pre-commit
    / CI check greps that no tuning script imports locked_split.load_locked_test"). Not called
    anywhere in this repo yet -- wiring a real confirmatory-scoring consumer is a separate,
    owner-gated step."""
    p = LOCKED_DIR / f"{dataset_key}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["test_ids"]


def load_locked_dev(dataset_key: str) -> list[str]:
    """Dev-side counterpart of ``load_locked_test`` -- freely readable (tuning happens on dev),
    kept alongside for symmetry; see module docstring / README's access-control convention."""
    p = LOCKED_DIR / f"{dataset_key}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["dev_ids"]


# ---------------------------------------------------------------------------------------------
# access-control convention docs (README.md / ACCESS_LOG.md) -- written on request, not implicitly
# on every run (so a --dry-run or a --dataset-scoped invocation never clobbers an existing,
# possibly hand-annotated ACCESS_LOG.md).
# ---------------------------------------------------------------------------------------------

README_TEXT = """# `_repro/LOCKED_HOLDOUT/` — the locked test/dev manifests (ticket #26)

Design doc: `wiki/2026-07-11-group-split-statistics-design.md` §2. Generated by
`scripts/baselines/locked_split.py` — one JSON file per dataset key
(`<dataset_key>.json`), each holding `{test_ids, dev_ids, group_unit, group_disjoint, seed,
created_at, ...}` (see that module's `generate_one` for the full field list).

## The rule

- **The locked `test_ids` may be read by exactly ONE consumer: the final confirmatory scoring
  pass that produces the reported number.** No arm selection, prompt search, threshold tuning,
  N*-budget choice, or reward-model calibration may read `test_ids` from any file in this
  directory, directly or indirectly.
- **`dev_ids` is freely readable** — tuning happens on dev, never on the locked test slice.
- **Access control is a process convention + an append-only log, not filesystem
  encryption** (owner ruling, Decision-Log 续13: "file convention + append-only access log, no
  encryption"). Every read of a `test_ids` field for any purpose should add a line to
  `ACCESS_LOG.md` in this directory (who / when / why) — see that file's header.
- Enforcement is currently by convention only (design doc §2.3 sketches a future CI grep guard
  that would check no tuning script imports `locked_split.load_locked_test` — not implemented by
  this ticket).

## Permanently non-locked (design doc finding #2 — NOT this directory's content)

The following are **permanently exposed** and can never be re-designated as a locked holdout,
because their membership has already informed at least one arm/prompt/threshold decision (the
`TEST_SEED`/`DEV_SEED` pair, and every existing per_item id list, are burned):

- `_repro/redraw_manifest.json` (the wave-1/2 disjoint-redraw manifest — see its own sidecar note,
  `_repro/redraw_manifest.NONLOCKED.md`).
- Every existing `_repro/baselines/*.json` result cell (all 264 clean cells + every
  `*.pre-redraw-20260710.json` archived sidecar).

The **only** locked holdout is the fresh, `LOCKED_TEST_SEED`-drawn, group-aware split in THIS
directory.

## Seeds

`LOCKED_TEST_SEED = 611_741_209` (owner-ratified, Decision-Log 续13) — a brand-new seed, never
used anywhere else in this repo before this ticket. The DEV draw derives from `seed + 1`
internally (`scripts/loaders/_common.py:draw_disjoint_grouped`), not a separately-chosen constant.
"""

ACCESS_LOG_HEADER = """# ACCESS_LOG.md — append-only access log for `_repro/LOCKED_HOLDOUT/`

Process convention (owner ruling, Decision-Log 续13: "file convention + append-only access log,
no encryption" — see `README.md` in this directory for the full rule). Every read of a `test_ids`
field from any manifest in this directory should append ONE line below, in this format, and
should NEVER edit or remove an existing line:

    - YYYY-MM-DDTHH:MM:SSZ | <dataset_key(s)> | <who/what> | <why — which confirmatory pass>

Do not read `test_ids` for arm selection, prompt search, threshold tuning, N*-budget choice, or
reward-model calibration — see `README.md`.

<!-- append new entries below this line -->
"""


def write_readme() -> Path:
    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    p = LOCKED_DIR / "README.md"
    p.write_text(README_TEXT, encoding="utf-8")
    return p


def write_access_log_header() -> Path:
    """Writes ONLY if the file doesn't already exist -- append-only convention, never overwritten
    by a re-run of this module (that would destroy any access lines already appended)."""
    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    p = LOCKED_DIR / "ACCESS_LOG.md"
    if p.exists():
        return p
    p.write_text(ACCESS_LOG_HEADER, encoding="utf-8")
    return p


REDRAW_MANIFEST_NONLOCKED_SIDECAR = REPO_ROOT / "_repro" / "redraw_manifest.NONLOCKED.md"

NONLOCKED_SIDECAR_TEXT = """# `_repro/redraw_manifest.json` — PERMANENTLY NON-LOCKED

Sidecar note (ticket #26, design doc `wiki/2026-07-11-group-split-statistics-design.md` finding
#2) — added BESIDE the JSON, which is left byte-for-byte unedited by this ticket.

`redraw_manifest.json`'s `new_test_ids` (drawn with `TEST_SEED=20261705`, the SAME seed the
already-scored wave-1/2 grid used) are **permanently non-locked**: a holdout is only a real
holdout if its membership has never informed a decision, and `TEST_SEED` had already selected
arms/prompts/thresholds before this manifest was even drawn. Disjointness from `dev` does not fix
that — reusable-holdout contamination is about REUSE of a seed that already informed a decision,
not about dev/test overlap.

**The only locked holdout is `_repro/LOCKED_HOLDOUT/` (fresh `LOCKED_TEST_SEED=611_741_209`,
group-aware, access-controlled by convention — see that directory's `README.md`).**

This note does not change `redraw_manifest.json` itself (still the accurate historical record of
the wave-1/2 disjoint redraw) — it only stamps the "do not treat this as a locked holdout" banner
next to it, per the ticket's explicit instruction not to edit the JSON.
"""


def write_nonlocked_sidecar() -> Path:
    REDRAW_MANIFEST_NONLOCKED_SIDECAR.write_text(NONLOCKED_SIDECAR_TEXT, encoding="utf-8")
    return REDRAW_MANIFEST_NONLOCKED_SIDECAR


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                     help="restrict to one dataset key (default: every ~65 redraw-scope key)")
    ap.add_argument("--timestamp", default=None,
                     help="ISO-8601 UTC timestamp stamped as every generated manifest's "
                          "'created_at' (default: now, computed once for the whole invocation so "
                          "a multi-dataset run stamps one consistent batch timestamp)")
    ap.add_argument("--n-test", type=int, default=TEST_N)
    ap.add_argument("--n-dev", type=int, default=DEV_N)
    ap.add_argument("--seed", type=int, default=LOCKED_TEST_SEED)
    ap.add_argument("--dry-run", action="store_true", help="compute + print; write nothing")
    ap.add_argument("--write-readme", action="store_true",
                     help="(re)write _repro/LOCKED_HOLDOUT/README.md and exit")
    ap.add_argument("--write-access-log-header", action="store_true",
                     help="write _repro/LOCKED_HOLDOUT/ACCESS_LOG.md's header IF it does not "
                          "already exist (append-only -- never overwrites), and exit")
    ap.add_argument("--write-nonlocked-sidecar", action="store_true",
                     help="write _repro/redraw_manifest.NONLOCKED.md (does not touch the JSON) "
                          "and exit")
    args = ap.parse_args()

    if args.write_readme:
        p = write_readme()
        print(f"wrote {p}")
        return
    if args.write_access_log_header:
        p = write_access_log_header()
        print(f"wrote {p} (or already existed -- append-only)")
        return
    if args.write_nonlocked_sidecar:
        p = write_nonlocked_sidecar()
        print(f"wrote {p}")
        return

    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    keys = [args.dataset] if args.dataset else redraw_scope_dataset_keys()

    t0 = time.time()
    n_ok = n_failed = 0
    for dk in keys:
        try:
            manifest = generate_one(dk, timestamp, n_test=args.n_test, n_dev=args.n_dev, seed=args.seed)
        except Exception as e:  # noqa: BLE001 -- one bad dataset must not sink the whole generation sweep
            print(f"!! {dk} FAILED: {type(e).__name__}: {e}", flush=True)
            n_failed += 1
            continue
        oe = manifest["old_exposed_test_ids"]
        if args.dry_run:
            print(f"[dry-run] {dk}: test={manifest['n_test']} dev={manifest['n_dev']} "
                  f"group_unit={manifest['group_unit']!r} group_disjoint={manifest['group_disjoint']} "
                  f"overlap_fraction={oe['overlap_fraction']}", flush=True)
        else:
            p = write_manifest(manifest)
            print(f"wrote {p} (test={manifest['n_test']} dev={manifest['n_dev']} "
                  f"group_disjoint={manifest['group_disjoint']} overlap_fraction={oe['overlap_fraction']})",
                  flush=True)
        n_ok += 1
    print(f"\nLOCKED_SPLIT_SUMMARY ok={n_ok} failed={n_failed} total={len(keys)} "
          f"elapsed_s={time.time() - t0:.0f}", flush=True)
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

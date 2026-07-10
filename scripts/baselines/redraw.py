"""scripts/baselines/redraw.py -- dev/test DISJOINT redraw driver (owner ruling, Decision-Log
续10, 2026-07-10: "dev/test 全部重抽（最严谨选项）").

## Why this exists

Every dev/test slice this repo has drawn so far (``run_baseline.py``'s ``DEV_SEED``/``TEST_SEED``
pattern, inherited by every ``scripts/loaders/*.py`` loader and the 7 legacy
``scripts/p2_baselines.py`` loaders) is two INDEPENDENT samples of the SAME underlying pool: dev =
``permutation(pool, DEV_SEED)[:40]``, test = ``permutation(pool, TEST_SEED)[:60]``. Two different
seeds make dev not a literal prefix of test, but nothing stops the two draws from sharing items --
and the wave-1/wave-2 audit (Decision-Log 续6/续9, using each result JSON's ``per_item.item_id``
list as the authoritative sampled-id record) found dev/test overlap in the vast majority of
dataset keys. This module (a) re-audits that overlap directly from the result JSONs (never
trusting a remembered headline number), (b) re-derives each overlapping dataset's FULL natural
pool via the *real* loader code (not reimplemented), (c) draws a genuinely disjoint (dev, test)
pair from that pool via ``_common.draw_disjoint`` (test first, from the whole pool; dev from the
remainder), and (d) freezes the result as NEW ``kb_snapshot`` manifests
(``baselines-<dataset>-disjoint-{dev,test}``) that never touch the old (still-live,
still-referenced-by-existing-result-JSONs) manifest names.

## IMPORTANT — the owner's "52" is a wave-1-only figure, not the true total

Decision-Log 续6 quantified "52/56 集有重叠" from the WAVE-1 grid alone (K1+K2+K8+K9, 56 dataset
keys); 续10's ruling ("52 个重叠数据集全部重新冻结... 受影响基线批量重跑~104 格") carries that
number forward and its own cost estimate (104 = 52 x 2 splits, qwen3-only) confirms it is wave-1
-only -- wave-2 (K4-K7, 16 dataset keys, completed in 续9 AFTER the 续6 audit) was never
re-audited for its own dev/test overlap. Running this module's ``census()`` over BOTH waves
(exactly what the task brief asks: "for every wave-1/2 dataset key") finds wave-2 has its OWN
13/16 overlapping keys (csemotions/esd/meld are the only 3 already disjoint) -- so the TRUE total
is **65** overlapping dataset keys (52 wave-1 + 13 wave-2), not 52. This module redraws all 65 (the
ruling's own stated PRINCIPLE, "dev/test 全部重抽" = redraw every overlapping key, not literally
"exactly 52"), and flags the wave-2 gap prominently in ``_repro/redraw_manifest.json`` and the
printed affected-cell list so the owner can decide whether wave-2's rerun was already implicitly
authorized by 续10 or needs its own sign-off. See this module's ``main()`` output and the "NOTE"
block it prints.

## Pool reconstruction without paying for full-corpus audio decode

A dataset's full pool is fetched via ``run_baseline._load_rows(dataset_key, split, n=None,
seed=_common.POOL_RECONSTRUCTION_SEED)`` -- reusing run_baseline's own dispatch (legacy /
registry / -attr / -slot / speech-massive / slurp / uro-UnderEmotion / vocalbench-emotion / seed-
tts-lang branches), never reimplemented here. ``n=None`` makes every loader return its WHOLE
natural split (no permutation-truncation for registry loaders; a full, seed-ordered permutation
for the 7 legacy p2_baselines loaders) -- but doing that for real would decode+write a wav file
for every row in datasets with thousands-row pools (aishell-1 test alone is 7176 rows), which is
pure waste when this module only wants the resulting ``item_id`` strings, never the audio itself.
``_stub_audio_materialization()`` below neutralizes exactly that cost by patching the ACTUAL
``soundfile.read``/``soundfile.write`` (and, for ``big-bench-audio``'s direct-librosa path,
``librosa.load``) functions for the duration of one pool fetch -- a single library-level patch
covers every loader's decode path (whether via the shared ``_common.wav_from_bytes`` helper or
``p2_baselines``'s own inline ``_wav_from_bytes``/``load_bba``) without editing any of the ~30
individual loader files. The same guard also stubs ``kb_snapshot.freeze_snapshot`` (a SINGLE
module-attribute patch, since ``_common.freeze_ids`` resolves it via a fresh attribute lookup on
every call -- see that function's body) so a read-only pool fetch can never clobber a loader's OWN
live snapshot manifest under its usual (non-``-disjoint``) experiment name.

Usage:
  python redraw.py --census                 # census-only: overlap per wave-1/2 dataset key, no pool
                                             # reconstruction, no writes (fast, read-only)
  python redraw.py --execute                # full redraw: census + pool reconstruction + draw_disjoint
                                             # + freeze new manifests + write _repro/redraw_manifest.json
  python redraw.py --execute --dataset aishell-1   # one dataset only (spot-check / debugging)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "loaders"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge"))

import run_baseline as rb             # noqa: E402  (scripts/baselines/run_baseline.py)
import templates                      # noqa: E402
import wave1_cells                    # noqa: E402
import wave2_cells                    # noqa: E402
from _common import (                 # noqa: E402  (scripts/loaders/_common.py)
    POOL_RECONSTRUCTION_SEED, draw_disjoint, freeze_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "_repro" / "redraw_manifest.json"

# Primary backbone the census reads ids from -- present for every wave-1/2 dataset key (wave-2 ran
# qwen3 only, 续9: "K4–K7 × qwen3 单底座"), whereas meralion-2-gguf result files exist for wave-1
# keys only. Using ONE backbone's ids is correct because both backbones share the exact same
# item-id set per (dataset, split) by construction (same seed, same loader, backbone-independent --
# wave-1 report Opus audit line: "双底座同 item-id 集零错配").
CENSUS_BACKBONE = "qwen3-omni-30b-gguf"


# ---------------------------------------------------------------------------------------------
# audio-decode / snapshot-write stubbing (pool reconstruction must be metadata-only + read-only)
# ---------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _stub_audio_materialization():
    """Patch ``soundfile.read``/``soundfile.write``/``librosa.load`` (dummy fast returns) and
    ``kb_snapshot.freeze_snapshot`` (no-op) for the duration of the ``with`` block.

    Library-attribute patches, not per-loader-module patches -- see module docstring's "Pool
    reconstruction without paying for full-corpus audio decode" section for why this single choke
    point covers every loader (registry or legacy) without editing any of them. Restores the
    original functions in ``finally`` unconditionally.
    """
    import soundfile as sf

    import kb_snapshot  # scripts/knowledge/kb_snapshot.py (already on sys.path via freeze_ids' own
    # lazy sys.path.insert, executed at least once by the time this module imports run_baseline /
    # _common -- imported directly here too so this stands alone if that ordering ever changes)

    try:
        import librosa
    except ImportError:
        librosa = None

    import numpy as np

    def _stub_read(*_a, **_kw):
        return np.zeros(1, dtype="float32"), 16000

    def _stub_write(*_a, **_kw):
        return None

    def _stub_load(*_a, **_kw):
        return np.zeros(1, dtype="float32"), 16000

    def _stub_freeze(*_a, **_kw):
        return "<stubbed-during-pool-reconstruction, see redraw.py:_stub_audio_materialization>"

    orig_read, orig_write = sf.read, sf.write
    orig_freeze = kb_snapshot.freeze_snapshot
    orig_load = librosa.load if librosa is not None else None

    sf.read = _stub_read
    sf.write = _stub_write
    kb_snapshot.freeze_snapshot = _stub_freeze
    if librosa is not None:
        librosa.load = _stub_load
    try:
        yield
    finally:
        sf.read, sf.write = orig_read, orig_write
        kb_snapshot.freeze_snapshot = orig_freeze
        if librosa is not None:
            librosa.load = orig_load


def full_pool_rows(dataset_key: str) -> list[dict]:
    """The FULL natural pool for ``dataset_key`` (every row ``run_baseline._load_rows`` would ever
    draw dev/test from), via the real dispatch, audio decode stubbed out (see above). ``split``
    is passed as ``"test"`` but is a NO-OP for every ``_load_rows`` branch when ``n=None`` -- see
    ``_common.POOL_RECONSTRUCTION_SEED``'s docstring and run_baseline.py's own branches (none of
    them consult the outer ``split`` label to choose which underlying corpus split to read; only
    the legacy branch's row ORDER depends on the seed passed here, which is why this always uses
    the one fixed ``POOL_RECONSTRUCTION_SEED``, never ``DEV_SEED``/``TEST_SEED``)."""
    with _stub_audio_materialization():
        return rb._load_rows(dataset_key, "test", n=None, seed=POOL_RECONSTRUCTION_SEED)


def full_pool_ids(dataset_key: str) -> list[str]:
    rows = full_pool_rows(dataset_key)
    ids = [r["meta"]["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise AssertionError(
            f"full_pool_ids({dataset_key!r}): pool has {len(dupes)} duplicate item_id(s) "
            f"(first few: {dupes[:5]}) -- draw_disjoint requires unique pool ids"
        )
    return ids


# ---------------------------------------------------------------------------------------------
# census (read-only: existing result JSONs' per_item.item_id lists ONLY, no loader calls)
# ---------------------------------------------------------------------------------------------

def _result_ids(dataset_key: str, split: str, backbone: str = CENSUS_BACKBONE) -> list[str] | None:
    p = rb.OUT_DIR / f"{dataset_key}__{backbone}__{split}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return [it.get("item_id") for it in data.get("per_item", []) if it.get("item_id") is not None]


def all_wave_dataset_keys() -> dict[str, str]:
    """dataset_key -> "wave1" | "wave2", the exact union the task brief asks for ("for every
    wave-1/2 dataset key"). Asserts the two waves don't share a key (they shouldn't -- disjoint
    K-type families, see wave1_cells.py / wave2_cells.py module docstrings)."""
    w1 = set(wave1_cells.wave1_dataset_keys())
    w2 = set(wave2_cells.wave2_dataset_keys())
    assert not (w1 & w2), f"wave1/wave2 dataset-key overlap: {w1 & w2}"
    return {**{k: "wave1" for k in w1}, **{k: "wave2" for k in w2}}


def census() -> dict[str, dict]:
    """Per dataset key: dev/test id sets (from result JSONs, backbone-independent by
    construction), their overlap, and which wave it belongs to. Read-only -- no loader calls."""
    out: dict[str, dict] = {}
    for dk, wave in sorted(all_wave_dataset_keys().items()):
        dev_ids = _result_ids(dk, "dev")
        test_ids = _result_ids(dk, "test")
        if dev_ids is None or test_ids is None:
            out[dk] = {"wave": wave, "error": "missing result JSON(s) for dev and/or test"}
            continue
        overlap = sorted(set(dev_ids) & set(test_ids))
        out[dk] = {
            "wave": wave,
            "n_dev": len(dev_ids), "n_test": len(test_ids),
            "overlap_count": len(overlap),
            "overlap_ids_sample": overlap[:10],
        }
    return out


# ---------------------------------------------------------------------------------------------
# redraw (per dataset: pool reconstruction + draw_disjoint + freeze new manifests)
# ---------------------------------------------------------------------------------------------

def redraw_one(dataset_key: str, old_overlap: int) -> dict:
    """Redraw ONE dataset key: reconstruct its full pool, draw a disjoint (dev, test) pair, freeze
    both as new ``baselines-<dataset_key>-disjoint-{dev,test}`` kb_snapshot manifests (the OLD
    ``<dataset_key>`` / ``baselines-<dataset_key>`` manifest name is never written here -- new
    experiment name, old untouched, per the task brief). Returns the manifest-entry dict."""
    pool_ids = full_pool_ids(dataset_key)
    drawn = draw_disjoint(
        pool_ids,
        n_test=rb.TEST_N, n_dev=rb.DEV_N,
        seed_test=rb.TEST_SEED, seed_dev=rb.DEV_SEED,
    )
    dev_ids, test_ids = drawn["dev_ids"], drawn["test_ids"]

    verified_disjoint = not (set(dev_ids) & set(test_ids))
    verified_subset = set(dev_ids) <= set(pool_ids) and set(test_ids) <= set(pool_ids)

    common_extra = {
        "source": "scripts/baselines/redraw.py",
        "old_overlap": old_overlap,
        "pool_size": drawn["pool_size"],
        "shortfall": drawn["shortfall"],
    }
    manifest_test = freeze_ids(
        f"baselines-{dataset_key}-disjoint-test", test_ids, dataset=dataset_key,
        seed=rb.TEST_SEED, extra={**common_extra, "kind": "disjoint-test"},
    )
    manifest_dev = freeze_ids(
        f"baselines-{dataset_key}-disjoint-dev", dev_ids, dataset=dataset_key,
        seed=rb.DEV_SEED, extra={**common_extra, "kind": "disjoint-dev"},
    )

    return {
        "old_overlap": old_overlap,
        "pool_size": drawn["pool_size"],
        "n_dev": drawn["n_dev"], "n_test": drawn["n_test"],
        "new_dev_ids": dev_ids, "new_test_ids": test_ids,
        "disjoint_verified": verified_disjoint,
        "ids_subset_of_pool_verified": verified_subset,
        "shortfall": drawn["shortfall"],
        "manifest_dev": manifest_dev, "manifest_test": manifest_test,
    }


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def print_census_table(rep: dict) -> None:
    by_wave: dict[str, list[str]] = {"wave1": [], "wave2": []}
    for dk, v in rep.items():
        by_wave[v["wave"]].append(dk)
    for wave in ("wave1", "wave2"):
        keys = by_wave[wave]
        n_ov = sum(1 for dk in keys if rep[dk].get("overlap_count", 0) > 0)
        print(f"=== {wave}: {n_ov}/{len(keys)} dataset keys have dev∩test overlap ===")
        for dk in keys:
            v = rep[dk]
            if "error" in v:
                print(f"  {dk:45s} ERROR: {v['error']}")
            elif v["overlap_count"] > 0:
                print(f"  {dk:45s} overlap={v['overlap_count']:3d}  "
                      f"(dev n={v['n_dev']} test n={v['n_test']})")
    total_overlap = sum(1 for v in rep.values() if v.get("overlap_count", 0) > 0)
    print(f"\nTOTAL overlapping dataset keys (wave1+wave2): {total_overlap}/{len(rep)}")
    print("NOTE: owner ruling (Decision-Log 续10) quotes '52' -- that is the WAVE-1-ONLY figure "
          "from 续6's audit (56 keys). Wave-2 (16 keys, completed after that audit) has its own "
          "overlapping keys not covered by the '52' count or its ~104-cell rerun estimate -- see "
          "this module's docstring.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", action="store_true", help="read-only overlap census; no writes")
    ap.add_argument("--execute", action="store_true",
                     help="full redraw: census + pool reconstruction + freeze new -disjoint manifests "
                          "+ write _repro/redraw_manifest.json")
    ap.add_argument("--dataset", default=None, help="restrict --execute to one dataset key (debugging)")
    args = ap.parse_args()

    if not args.census and not args.execute:
        ap.print_help()
        return

    rep = census()
    if args.census:
        print_census_table(rep)
        return

    # --execute
    overlapping = {dk: v for dk, v in rep.items()
                   if "error" not in v and v["overlap_count"] > 0}
    if args.dataset:
        assert args.dataset in overlapping, (
            f"{args.dataset!r} has no dev/test overlap (or no result JSONs) -- nothing to redraw; "
            f"overlapping keys: {sorted(overlapping)}"
        )
        overlapping = {args.dataset: overlapping[args.dataset]}

    print(f"=== redrawing {len(overlapping)} overlapping dataset key(s) ===")
    manifest: dict[str, dict] = {}
    n_ok = n_failed = 0
    for dk, v in sorted(overlapping.items()):
        print(f"  [{v['wave']}] {dk} (old_overlap={v['overlap_count']}) ...", end=" ", flush=True)
        try:
            entry = redraw_one(dk, v["overlap_count"])
            entry["wave"] = v["wave"]
            manifest[dk] = entry
            n_ok += 1
            print(f"pool_size={entry['pool_size']} n_dev={entry['n_dev']} n_test={entry['n_test']} "
                  f"disjoint_verified={entry['disjoint_verified']}"
                  + (f" SHORTFALL={entry['shortfall']}" if entry["shortfall"] else ""))
        except Exception as e:  # noqa: BLE001 -- one bad dataset must not sink the whole redraw
            n_failed += 1
            manifest[dk] = {"wave": v["wave"], "old_overlap": v["overlap_count"],
                             "error": f"{type(e).__name__}: {e}"}
            print(f"FAILED: {type(e).__name__}: {e}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({
        "generated_by": "scripts/baselines/redraw.py",
        "owner_ruling": "Decision-Log 续10 (2026-07-10): dev/test 全部重抽",
        "note": ("owner's '52' is the wave-1-only figure from 续6's audit; this manifest covers "
                 "the TRUE union (wave-1 + wave-2) discovered by re-auditing both waves -- see "
                 "redraw.py module docstring."),
        "n_overlapping_total": len(overlapping),
        "n_overlapping_wave1": sum(1 for v in overlapping.values() if v["wave"] == "wave1"),
        "n_overlapping_wave2": sum(1 for v in overlapping.values() if v["wave"] == "wave2"),
        "n_redrawn_ok": n_ok, "n_redrawn_failed": n_failed,
        "datasets": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {MANIFEST_PATH} ({n_ok} ok, {n_failed} failed)")

    print("\n=== affected-cell rerun list (dataset x split, qwen3-omni-30b-gguf only) ===")
    wave1_cells_list = [(dk, sp) for dk in sorted(manifest) if manifest[dk].get("wave") == "wave1"
                        and "error" not in manifest[dk] for sp in ("dev", "test")]
    wave2_cells_list = [(dk, sp) for dk in sorted(manifest) if manifest[dk].get("wave") == "wave2"
                        and "error" not in manifest[dk] for sp in ("dev", "test")]
    print(f"wave-1 (owner-authorized, 续10): {len(wave1_cells_list)} cells "
          f"({len(wave1_cells_list) // 2} datasets x 2 splits)")
    for dk, sp in wave1_cells_list:
        print(f"  python scripts/baselines/run_baseline.py --dataset {dk} --backbone qwen3-omni-30b-gguf "
              f"--split {sp} --slice disjoint")
    print(f"wave-2 (NEWLY DISCOVERED, needs owner sign-off -- not in 续10's '52'/'~104'): "
          f"{len(wave2_cells_list)} cells ({len(wave2_cells_list) // 2} datasets x 2 splits)")
    for dk, sp in wave2_cells_list:
        print(f"  python scripts/baselines/run_baseline.py --dataset {dk} --backbone qwen3-omni-30b-gguf "
              f"--split {sp} --slice disjoint")


if __name__ == "__main__":
    main()

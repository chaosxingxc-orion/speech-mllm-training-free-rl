"""kb_snapshot — freeze a reproducible eval slice so any run is replayable item-for-item.

The Stage-1 reproducibility floor (owner, 2026-07-07): "所有的实验和验证必须要求可追溯,
你采样的数据集,进行的实验必须要可复现". Before this module, reproducibility relied only on a
hardcoded SEED + a deterministic ``np.random.default_rng(SEED).permutation`` (e.g.
``p2_baselines.py:215``, ``t7_rag_gate_probe.py:114``) — replayable ONLY if the underlying
parquet file and row order never change. This module records the EXPLICIT item ids sampled,
so a slice is provable independent of file/row drift, and pins the dataset revision + KB hash.

A ``sample_manifest.json`` fully determines a slice:
    { experiment, dataset, revision, slice_seed, n, item_ids[], kb_build_hash, extra }
"""
from __future__ import annotations

import json

from kb_schema import snapshot_dir


def freeze_snapshot(
    experiment: str,
    dataset: str,
    revision: str | None,
    slice_seed: int,
    item_ids: list,
    kb_build_hash: str | None = None,
    extra: dict | None = None,
) -> str:
    """Write snapshots/<experiment>/sample_manifest.json; return its path."""
    d = snapshot_dir(experiment)
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": experiment,
        "dataset": dataset,
        "revision": revision,
        "slice_seed": slice_seed,
        "n": len(item_ids),
        "item_ids": list(item_ids),
        "kb_build_hash": kb_build_hash,
        "extra": extra or {},
    }
    p = d / "sample_manifest.json"
    json.dump(manifest, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return str(p)


def load_snapshot(experiment: str) -> dict:
    p = snapshot_dir(experiment) / "sample_manifest.json"
    return json.load(open(p, encoding="utf-8"))


def replay_matches(experiment: str, replayed_item_ids: list) -> dict:
    """Verify a fresh sampling reproduces the frozen slice exactly (traceability proof).

    Returns {ok, n_frozen, n_replayed, missing, extra} — ok iff the two id sets are identical.
    """
    frozen = load_snapshot(experiment)["item_ids"]
    fs, rs = set(map(str, frozen)), set(map(str, replayed_item_ids))
    missing = sorted(fs - rs)
    extra = sorted(rs - fs)
    return {
        "ok": not missing and not extra and len(frozen) == len(replayed_item_ids),
        "n_frozen": len(frozen),
        "n_replayed": len(replayed_item_ids),
        "missing": missing,
        "extra": extra,
    }

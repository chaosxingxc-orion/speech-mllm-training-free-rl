"""loaders/audio2tool.py — Audio2Tool spoken-tool-calling loader (B6-retrieval-tools).

LICENSE: cc-by-nc-4.0 (dataset card frontmatter) — non-commercial, EVAL-ONLY.

On-disk layout (confirmed 2026-07-09): 8 tier JSONs under
``datasets/audio2tool/public/<tier>_data/<tier>.json``, each a flat JSON list of query dicts.
Every item's ``"path"`` field is relative to ``datasets/audio2tool/public/`` (verified —
``tier1_direct_audios/query_00001/speaker_02_....wav`` resolves under ``public/`` directly, NOT
under the ``..._data/`` dir). ``tools_registry.csv`` (152 tools, columns: tool_id, domain,
category, tool_name, signature, description, argument_defaults, argument_constraints) is a
flat knowledge source, not per-item — exposed via ``load_tools_registry()`` for callers that
want it (e.g. few-shot tool-spec delivery), not embedded in every Row.

Confirmed tier sizes (item counts, field-verified — larger than the survey recipe's rough
estimate of "tier1: 2146"; document the discrepancy here per the loader-contract instructions):
tier1_direct=4292, tier2_parametric=6320, tier3_multi_intent=4292, tier4_implicit=4278
(tiers 5-8 not enumerated here; same JSON-list shape).

Item schema (tier1 sample): {id, tier, query_idx, query, domain, category, tool_id, tool_name,
expected_tool_call, extracted_params, additional_tool_calls, functions, path, duration,
instruction, speaker_idx, speaker_id, speaker_source, source_endpoint}. ``functions`` is the
list of tool specs available to the model for that query (a dict subset of the tools_registry
row shape) — carried into ``meta`` as task-shaping context, not scoring.

Only a single (test-only, no train/val) pool per tier — ``split`` accepts only ``"test"``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders

from _common import SLICE_SEED, datasets_dir, freeze_ids  # noqa: E402

TIERS = [
    "tier1_direct", "tier2_parametric", "tier3_multi_intent", "tier4_implicit",
    "tier5_needle", "tier6_correction", "tier7_multiturn", "tier8_intent_blending",
]


def _public_dir() -> Path:
    return datasets_dir() / "audio2tool" / "public"


def _tier_json_path(tier: str) -> Path:
    if tier not in TIERS:
        raise ValueError(f"unknown audio2tool tier {tier!r}; choose one of {TIERS}")
    return _public_dir() / f"{tier}_data" / f"{tier}.json"


def _select_indices(n_total: int, n: int | None, seed: int) -> list[int]:
    import numpy as np

    if n is None or n >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n, replace=False)
    return sorted(int(i) for i in idx)


def load_audio2tool(
    split: str = "test",
    n: int | None = None,
    seed: int = SLICE_SEED,
    tier: str = "tier1_direct",
) -> list[dict]:
    """Load `n` items from one Audio2Tool tier.

    Row shape: ``wav`` = path under ``public/<tier>_.../query_XXXXX/`` (resolved absolute);
    ``gold`` = ``{"expected_tool_call", "extracted_params", "additional_tool_calls"}``;
    ``meta`` = ``{"dataset", "item_id", "split", "tier", "tool_name", "domain", "functions",
    "query_idx"}`` (the last added 2026-07-11, ticket #26 -- grouping/provenance only).
    """
    import json

    if split != "test":
        raise ValueError(f"audio2tool is test-only per tier (got split={split!r})")

    items = json.load(open(_tier_json_path(tier), encoding="utf-8"))
    items_sorted = sorted(items, key=lambda it: it["id"])
    idx = _select_indices(len(items_sorted), n, seed)
    sel = [items_sorted[i] for i in idx]

    public = _public_dir()
    out = []
    ids = []
    for it in sel:
        wav = str(public / it["path"])
        item_id = f"{tier}|{it['id']}"
        ids.append(item_id)
        out.append({
            "wav": wav,
            "gold": {
                "expected_tool_call": it.get("expected_tool_call"),
                "extracted_params": it.get("extracted_params"),
                "additional_tool_calls": it.get("additional_tool_calls"),
            },
            "meta": {
                "dataset": "audio2tool", "item_id": item_id, "split": split, "tier": tier,
                "tool_name": it.get("tool_name"), "domain": it.get("domain"),
                "functions": it.get("functions"), "instruction": it.get("instruction"),
                "query_text": it.get("query"),
                # 2026-07-11 (ticket #26, group-split design doc §1.3/§4.1 K10 row): one query
                # (query_idx) is rendered by MANY speakers -- grouping/provenance only, never a
                # task label. Source jsonl already carries this per item (see module docstring's
                # item schema); previously not surfaced into meta at all.
                "query_idx": it.get("query_idx"),
            },
        })

    freeze_ids("audio2tool", ids, dataset="audio2tool", seed=seed, extra={"tier": tier})
    return out


def load_tools_registry() -> list[dict]:
    """Load the flat 152-tool registry (``tools_registry.csv``) as a list of dicts.

    Knowledge-source, not per-item Row data — see module docstring. Not called by
    ``load_audio2tool`` itself (thin-loader discipline: no implicit heavy joins).
    """
    import csv

    p = datasets_dir() / "audio2tool" / "tools_registry.csv"
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

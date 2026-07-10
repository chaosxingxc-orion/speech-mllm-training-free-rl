"""scripts/baselines/provenance.py — shared provenance-field helper (ticket #25 P4, "Also: add
provenance fields into run_baseline.write_result / run_mock result writer").

Before this task, a result JSON's reproducibility trail was scattered/partial: git state was never
recorded at all, "engine/build id" (e.g. which llama.cpp binary/commit served a GPU cell) was never
captured, dataset revision was only sometimes present (``run_baseline``'s ``sample_manifest`` path
string, not the revision value itself), and there was no manifest-hash / environment-version field
anywhere. ``collect_provenance`` is the ONE function both writers now call so a result JSON's
provenance block has an identical shape regardless of which pipeline produced it.

Every field here is BEST-EFFORT and never raises: a result JSON must still get written even when
(e.g.) git isn't on PATH, or this is a plain ``python`` invocation on a box with no GPU server up.
Lazy-import discipline (CLAUDE.md): only stdlib (``subprocess``/``sys``/``os``) at import time.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _git_state(repo_root) -> dict:
    """``{"sha": <7+ char short sha or None>, "dirty": bool or None}`` -- None/None if this isn't a
    git checkout or ``git`` isn't on PATH (never raises)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=10,
        )
        if sha.returncode != 0:
            return {"sha": None, "dirty": None}
        return {"sha": sha.stdout.strip() or None,
                "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}
    except Exception:  # noqa: BLE001 -- provenance must never block a result write
        return {"sha": None, "dirty": None}


def _env_versions() -> dict:
    """Best-effort interpreter + a few heavy-dep versions (only those ALREADY imported by the
    caller's process -- this never imports anything itself, honoring the lazy-import discipline:
    a CPU-only run_mock invocation that never touched numpy/torch reports those as ``None``, not
    by force-importing them just to fill in a provenance field)."""
    versions = {"python": sys.version.split()[0]}
    for mod_name in ("numpy", "torch", "transformers", "sklearn", "librosa"):
        mod = sys.modules.get(mod_name)
        versions[mod_name] = getattr(mod, "__version__", None) if mod is not None else None
    return versions


def _engine_build_id(backbone_cfg: dict | None) -> str | None:
    """The GPU-serving engine's own build/version identity, when a GPU cell recorded one via its
    ``server_env`` resident-server ``/props`` endpoint's ``build_info`` (llama.cpp exposes this).
    Never queries the server itself here (provenance collection must stay lazy/side-effect-free) --
    a caller that already queried ``/props`` this run (e.g. ``kb_embed._llama_server_embed``'s
    media_marker fetch) may pass the raw props dict via ``backbone_cfg={"props": {...}}``; absent
    that, this is ``None`` -- "not queried this run", not "no engine".
    """
    if not backbone_cfg:
        return None
    props = backbone_cfg.get("props")
    if isinstance(props, dict):
        return props.get("build_info") or props.get("model") or None
    return backbone_cfg.get("engine_build_id")


def collect_provenance(repo_root, dataset_revision: str | None = None,
                        manifest_hash: str | None = None, backbone_cfg: dict | None = None) -> dict:
    """The ONE provenance block both ``run_baseline.write_result`` and ``run_mock``'s result writer
    embed. ``dataset_revision``/``manifest_hash`` are caller-supplied (this function does not know
    which dataset/KB-source a given result cell used) -- ``None`` when not available/applicable,
    never fabricated.
    """
    git = _git_state(repo_root)
    return {
        "git_sha": git["sha"],
        "git_dirty": git["dirty"],
        "engine_build_id": _engine_build_id(backbone_cfg),
        "dataset_revision": dataset_revision,
        "manifest_hash": manifest_hash,
        "env_versions": _env_versions(),
    }

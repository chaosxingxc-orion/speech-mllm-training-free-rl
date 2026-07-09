"""_common — shared helpers for scripts/loaders/ dataset loaders.

Scaffold task (B0). This module is the ONLY place loader helpers live; every ``load_<name>``
in ``scripts/loaders/<name>.py`` should import from here rather than re-implementing path
resolution / wav materialization / snapshot freezing.

Lazy-import discipline (CLAUDE.md): heavy deps (soundfile, numpy, librosa, ...) stay inside
functions; ``import _common`` itself stays light and always succeeds, even before the ML stack
is installed.

Row convention (see README.md):
    Row = {"wav": <str path, or raw bytes>, "gold": <str or dict>, "meta": <dict>}

    - "wav"  — a path to a materialized wav file (preferred), or the raw undecoded audio bytes
               if the loader wants the caller to decode lazily. Loaders SHOULD hand back a path
               (via ``wav_from_bytes`` when the source dataset stores audio as embedded bytes,
               e.g. a HF parquet ``{"bytes": ...}`` column) so downstream code never has to care
               which shape a given dataset ships in.
    - "gold" — the reference label/answer/transcript/intent, in whatever shape the task needs
               (str for free-text/ASR gold, dict for structured gold e.g. {"choices": [...],
               "answer_idx": 2}). Callers key on ``meta["task"]`` to know how to interpret it.
    - "meta" — free-form provenance + task-shaping fields: at minimum {"dataset", "item_id",
               "split"}; loaders may add "task", "instr", "opts", or anything else a specific
               dataset needs. NOT reward/metric code — this is data, not scoring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, TypedDict

# Pinned 2026-07-05 (first used in p2_baselines.py) — the default slice seed every loader should
# sample with unless a caller explicitly overrides it. Keeping ONE shared constant here (rather
# than each loader hardcoding its own) is what makes cross-loader slices comparable/reproducible.
SLICE_SEED = 20260705


class Row(TypedDict, total=False):
    wav: Any  # str path to a wav file, OR raw audio bytes (see module docstring)
    gold: Any  # str or dict — the reference label/answer/transcript
    meta: dict  # provenance + task-shaping: {"dataset", "item_id", "split", ...}


# ---- path resolution (mirrors kb_schema.kb_root's SPEECHRL_KB_DIR pattern) ----
def data_root() -> Path:
    """Resolve SPEECHRL_DATA_DIR. Raises rather than silently defaulting.

    Data lives on the E: drive in this environment (CLAUDE.md) and is never assumed to be at a
    fixed path — every loader MUST read this env var (directly or via this helper), not hardcode
    a path, so a missing/misconfigured env fails loudly instead of silently reading nothing (or
    the wrong tree).
    """
    env = os.environ.get("SPEECHRL_DATA_DIR")
    if not env:
        raise RuntimeError(
            "SPEECHRL_DATA_DIR is not set. Loaders must read data from it explicitly, e.g.:\n"
            "  export SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data"
        )
    return Path(env)


def datasets_dir() -> Path:
    """SPEECHRL_DATA_DIR/datasets — where every dataset's raw files live."""
    return data_root() / "datasets"


def wav_cache_dir(name: str) -> Path:
    """Per-loader wav-materialization cache dir: SPEECHRL_DATA_DIR/_repro/loader_wavs/<name>.

    Mirrors ``p2_baselines.py``'s ``WAV = DATA / "_repro" / "p2_wavs"`` pattern, but namespaced
    per loader (``name``) so two loaders never collide on a cache key.
    """
    return data_root() / "_repro" / "loader_wavs" / name


def wav_from_bytes(b: bytes, key: str, cache_dir: Path) -> str:
    """Materialize raw audio bytes to a cached mono wav file; return its path (str).

    Adapted from ``p2_baselines.py:_wav_from_bytes`` — identical semantics (decode via
    soundfile, downmix to mono if multi-channel, cache by ``key`` so repeat calls are free)
    generalized with an explicit ``cache_dir`` argument instead of a module-global ``WAV``
    constant (use ``wav_cache_dir(name)`` to get one). Heavy deps (soundfile) are imported here,
    not at module top, per the lazy-import discipline.
    """
    import io

    import soundfile as sf

    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.wav"
    if not p.exists():
        y, sr = sf.read(io.BytesIO(b), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        sf.write(str(p), y, sr)
    return str(p)


def freeze_ids(
    name: str,
    ids: list,
    *,
    dataset: str | None = None,
    revision: str | None = None,
    seed: int = SLICE_SEED,
    extra: dict | None = None,
) -> str:
    """Freeze the sampled item ids for loader ``name`` via kb_snapshot; return the manifest path.

    Thin wrapper around ``scripts/knowledge/kb_snapshot.freeze_snapshot`` — see that module for
    the reproducibility contract (CLAUDE.md Stage-1 floor: every experiment must freeze an
    explicit ``item_ids`` list, not rely on file/row order staying put). Adds
    ``scripts/knowledge`` to ``sys.path`` lazily (mirrors ``kb_poc.py``'s own setup) so this
    module has no import-time dependency on it — importing ``_common`` never touches kb_snapshot
    until ``freeze_ids`` is actually called.

    Args:
        name: experiment/loader name — the snapshot is written under
            ``SPEECHRL_KB_DIR/snapshots/<name>/sample_manifest.json``.
        ids: the explicit item ids sampled (order-independent identity, not a row index).
        dataset: dataset name for the manifest; defaults to ``name``.
        revision: pinned dataset revision (see docs/datasets.lock.json), if applicable.
        seed: the slice seed used to draw ``ids`` (defaults to ``SLICE_SEED``).
        extra: any additional provenance to embed in the manifest.

    Returns:
        The path (str) to the written ``sample_manifest.json``.
    """
    here = os.path.dirname(os.path.abspath(__file__))  # scripts/loaders
    knowledge_dir = os.path.join(os.path.dirname(here), "knowledge")  # scripts/knowledge
    if knowledge_dir not in sys.path:
        sys.path.insert(0, knowledge_dir)
    import kb_snapshot  # noqa: E402

    return kb_snapshot.freeze_snapshot(
        experiment=name,
        dataset=dataset or name,
        revision=revision,
        slice_seed=seed,
        item_ids=ids,
        extra=extra,
    )


LoaderFn = Callable[..., list]  # def load_<name>(split='test', n=None, seed=SLICE_SEED) -> list[Row]

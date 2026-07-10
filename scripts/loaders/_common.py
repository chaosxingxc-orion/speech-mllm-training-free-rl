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


def load_snapshot_ids(name: str) -> list:
    """Read back the frozen ``item_ids`` list for kb_snapshot experiment ``name`` (thin read-side
    counterpart to ``freeze_ids`` above -- same lazy ``scripts/knowledge`` sys.path setup, so this
    module keeps zero import-time dependency on ``kb_snapshot``). Raises ``FileNotFoundError`` (via
    ``kb_snapshot.load_snapshot``'s own ``open()``) if no manifest was ever frozen under ``name``.

    Added 2026-07-10 for the dev/test disjoint-redraw task (owner ruling, Decision-Log 续10):
    ``scripts/baselines/redraw.py`` freezes new ``baselines-<dataset>-disjoint-{dev,test}``
    manifests here; ``scripts/baselines/run_baseline.py``'s ``--slice disjoint`` reads them back
    via this function rather than re-deriving a slice from ``DEV_SEED``/``TEST_SEED``.
    """
    here = os.path.dirname(os.path.abspath(__file__))  # scripts/loaders
    knowledge_dir = os.path.join(os.path.dirname(here), "knowledge")  # scripts/knowledge
    if knowledge_dir not in sys.path:
        sys.path.insert(0, knowledge_dir)
    import kb_snapshot  # noqa: E402

    return kb_snapshot.load_snapshot(name)["item_ids"]


# Pool-reconstruction seed for the dev/test disjoint-redraw task (owner ruling 2026-07-10,
# Decision-Log 续10): the ONE seed ``scripts/baselines/redraw.py`` uses to fetch a dataset's FULL
# natural pool (``n=None``) before drawing disjoint dev/test slices from it, and the SAME seed
# ``run_baseline.py``'s ``--slice disjoint`` path uses to re-fetch that pool at replay time.
# Deliberately reuses ``SLICE_SEED`` rather than inventing a third constant -- but calling it out
# BY NAME here matters: for registry loaders `n=None` skips ``rng.permutation`` entirely (see
# ``idx = ... if n is not None else np.arange(len(rows))`` in every loader), so the seed is inert
# for them either way. For the 7 ``scripts/p2_baselines.py``-native LEGACY_DATASETS loaders,
# though, `n=None` still runs `rng.permutation(len(rows))[:None]` (a full, but SEED-DEPENDENT,
# reordering), and `run_baseline._legacy_rows` numbers each returned row POSITIONALLY
# (``f"{dataset_key}#{i}"``, i = position in that fetch's output order, NOT the underlying pool
# index) -- so a legacy item_id like ``"SQuAD-zh#37"`` only refers to a stable, reproducible
# underlying row if EVERY full-pool refetch (freeze time in redraw.py, replay time in
# run_baseline.py) uses this exact same seed. Using a different seed at either end would silently
# rebind "#37" to a different row without either side ever raising.
POOL_RECONSTRUCTION_SEED = SLICE_SEED


def draw_disjoint(
    pool_ids: list,
    n_test: int = 60,
    n_dev: int = 40,
    seed_test: int = SLICE_SEED + 1000,   # mirrors run_baseline.TEST_SEED (20261705)
    seed_dev: int = SLICE_SEED,           # mirrors run_baseline.DEV_SEED (20260705)
) -> dict:
    """Draw a genuinely DISJOINT (dev, test) pair of id slices from ``pool_ids``.

    Added 2026-07-10 for the dev/test disjoint-redraw task (owner ruling, Decision-Log 续10):
    every pre-existing dev/test slice in this repo was drawn as two INDEPENDENT samples of the
    same pool (``DEV_SEED``-permutation head vs ``TEST_SEED``-permutation head) -- two views of
    one pool, not a real held-out split, so they overlap whenever the pool is small relative to
    dev_n+test_n (confirmed empirically: 52/56 wave-1 + 13/16 wave-2 dataset keys overlap). This
    function fixes that by construction: **test is drawn FIRST from the full pool, then dev is
    drawn from the REMAINDER** (pool minus the test draw) -- so a future change to dev_n/seed_dev
    can never perturb the (more consequential, held-out-scored) test slice, and the two id sets
    are disjoint by construction rather than by chance.

    Order/seed convention preserved from ``run_baseline.py``'s existing (non-disjoint) slicing:
    ``seed_test`` defaults to the documented ``SLICE_SEED + 1000`` offset (``run_baseline.
    TEST_SEED``), ``seed_dev`` to bare ``SLICE_SEED`` (``run_baseline.DEV_SEED``) -- so a caller
    that doesn't override either gets the exact same seeds the frozen (non-disjoint) grid already
    used, just applied disjointly.

    Small-pool handling: if ``len(pool_ids) < n_test + n_dev`` (default threshold 100), the
    100 = n_test + n_dev floor cannot be met -- allocate the two slices PROPORTIONALLY to the
    60/40 (test/dev) ratio over whatever the pool actually holds (test's share rounded, dev takes
    the remainder) rather than raising or silently over-drawing, and report the shortfall in the
    returned dict so a caller (redraw.py's manifest) can flag it rather than silently truncating.

    Args:
        pool_ids: the FULL pool's item ids, in ANY stable order (the order itself doesn't matter
            -- both draws re-permute internally with their own explicit seed -- only that calling
            this twice with the same ``pool_ids`` list reproduces the same two slices).
        n_test / n_dev: requested slice sizes (frozen convention: 60 / 40).
        seed_test / seed_dev: the two independent RNG seeds (test drawn first, from the full
            pool; dev drawn second, from pool-minus-test).

    Returns:
        {"test_ids": [...], "dev_ids": [...], "pool_size": int, "n_test": int, "n_dev": int,
         "shortfall": {...} | None}  -- ``shortfall`` is non-None iff the pool was too small to
        satisfy the requested (n_test, n_dev) floor, and records what was requested vs. what was
        actually drawn.
    """
    import numpy as np

    pool = list(pool_ids)
    pool_size = len(pool)
    if pool_size == 0:
        raise ValueError("draw_disjoint: pool_ids is empty -- nothing to draw from")

    requested_test, requested_dev = n_test, n_dev
    shortfall = None
    if pool_size < n_test + n_dev:
        # Proportional 60/40 allocation over the actual pool size (test's share rounded to the
        # nearest int, dev absorbs whatever remains) -- never draw more than the pool holds.
        ratio_test = n_test / (n_test + n_dev)
        n_test = min(pool_size, int(round(pool_size * ratio_test)))
        n_dev = pool_size - n_test
        shortfall = {
            "requested_test": requested_test, "requested_dev": requested_dev,
            "effective_test": n_test, "effective_dev": n_dev, "pool_size": pool_size,
        }

    rng_test = np.random.default_rng(seed_test)
    test_idx = rng_test.permutation(pool_size)[:n_test]
    test_idx_set = {int(i) for i in test_idx}
    test_ids = [pool[int(i)] for i in test_idx]

    remainder = [p for i, p in enumerate(pool) if i not in test_idx_set]
    rng_dev = np.random.default_rng(seed_dev)
    n_dev_eff = min(n_dev, len(remainder))
    dev_idx = rng_dev.permutation(len(remainder))[:n_dev_eff]
    dev_ids = [remainder[int(i)] for i in dev_idx]

    return {
        "test_ids": test_ids, "dev_ids": dev_ids, "pool_size": pool_size,
        "n_test": len(test_ids), "n_dev": len(dev_ids),
        "shortfall": shortfall,
    }


# ---------------------------------------------------------------------------------------------
# 2026-07-11 group-split + cluster-bootstrap redesign (ticket #26, QUESTION-INDEPENDENT machinery
# only -- design doc wiki/2026-07-11-group-split-statistics-design.md §2.2). Added BESIDE
# ``draw_disjoint`` above (kept unchanged, for backward replay of the non-locked cells) --
# ``draw_disjoint_grouped`` is NOT wired into any runner yet: no caller in this repo invokes it,
# the locked-test-seed / access-controlled manifest machinery the design doc sketches (§2.1/§2.3,
# ``locked_split.py``, ``_repro/LOCKED_HOLDOUT/``) does NOT exist, and no rerun has happened --
# those are owner-gated steps (design doc §5's five open questions), not this ticket's scope.
# ---------------------------------------------------------------------------------------------

def draw_disjoint_grouped(
    items: list,
    group_key_fn: Callable[[dict], "str | None"],
    n_test: int = 60,
    n_dev: int = 40,
    seed: int = SLICE_SEED,
) -> dict:
    """Group-disjoint (test, dev) draw over ``items`` -- ALL items of a group land on the SAME
    side (design doc §2.2, audit finding #1: item-level disjointness lets two items of the same
    group -- speaker/session/dialogue/source-question-family/clip -- land on opposite sides, so
    "test" isn't actually independent of "dev" when items correlate within a group).

    ``items`` is a list of loader ``Row``s (``{"wav", "gold", "meta"}`` with
    ``meta["item_id"]`` set) -- or a synthetic test double shaped the same way. ``group_key_fn`` is
    typically ``functools.partial(scripts.loaders.group_key.group_key_of, dataset_key)`` -- any
    callable ``item -> str | None``. Returning ``None`` for an item means "no group finer than the
    whole dataset" (G-NONE, see that module); this function then treats the item as its OWN
    singleton group (keyed by its ``item_id``), so a G-NONE dataset transparently degrades to the
    OLD item-level behavior with NO special-casing by the caller -- exactly mirroring
    ``draw_disjoint``'s semantics for that case (design doc: "the *same* function yields item-level
    splits for [G-NONE datasets] with no special-casing").

    Algorithm (design doc §2.2, adapted here to a ``group_key_fn`` callable instead of a
    precomputed ``group_of`` dict, and a single ``seed`` instead of separate seed_test/seed_dev --
    two internal RNGs are derived from it: ``seed`` for the TEST draw, ``seed + 1`` for the DEV
    draw, mirroring ``draw_disjoint``'s own TEST_SEED/DEV_SEED offset convention above). Uses a
    plain caller-supplied ``seed`` (default ``SLICE_SEED``) -- this module does NOT reference the
    design doc's proposed locked-holdout seed (no such constant exists yet; see the block comment
    above):
      1. ``group_of[item_id] = group_key_fn(item) or item_id`` (singleton fallback); invert into
         ``groups: {group_id: [item_ids]}``; sort group ids for a stable enumeration order.
      2. ``rng_test = default_rng(seed)``; permute the group-id list; greedily add WHOLE groups to
         ``test`` until the item count first reaches ``n_test`` (accept the overshoot from the
         last group rather than ever splitting a group).
      3. Remove the chosen test groups; ``rng_dev = default_rng(seed + 1)`` permutes the REMAINING
         groups; greedily fill ``dev`` to ``n_dev`` items the same way.
      4. Small-pool handling mirrors ``draw_disjoint``'s proportional 60/40 fallback when whole
         groups can't hit ``n_test + n_dev`` -- recorded in ``shortfall``. A single group already
         at/above ``n_test`` alone is flagged ``oversized_group`` (that cell is then effectively
         one-cluster -- item-level bootstrap, caveat; design doc §2.2 step 5).

    Returns:
        {"test_ids", "dev_ids": list[str], "test_groups", "dev_groups": list[str],
         "n_groups_total": int, "n_test", "n_dev": int (actual drawn counts),
         "group_disjoint_verified": bool, "shortfall": dict | None,
         "oversized_group": str | None, "fallback_item_level": bool} -- the last key is True iff
        EVERY item's ``group_key_fn`` returned None (a G-NONE dataset), i.e. every "group" is a
        singleton and this call is equivalent to a plain item-level ``draw_disjoint``.
    """
    import numpy as np

    def _item_id(it: dict) -> str:
        return it["meta"]["item_id"]

    group_of: dict = {}
    any_real_group = False
    for it in items:
        iid = _item_id(it)
        gk = group_key_fn(it)
        if gk is not None:
            any_real_group = True
        group_of[iid] = gk if gk is not None else iid

    groups: dict = {}
    for iid, gid in group_of.items():
        groups.setdefault(gid, []).append(iid)
    group_ids = sorted(groups.keys())
    n_groups_total = len(group_ids)

    def _greedy_fill(candidate_group_ids: list, target_n: int) -> tuple:
        """Whole-groups-only greedy fill: returns (chosen_group_ids, chosen_item_ids)."""
        chosen_groups, chosen_items = [], []
        for gid in candidate_group_ids:
            if len(chosen_items) >= target_n:
                break
            chosen_groups.append(gid)
            chosen_items.extend(groups[gid])
        return chosen_groups, chosen_items

    total_items = sum(len(v) for v in groups.values())
    requested_test, requested_dev = n_test, n_dev
    shortfall = None

    oversized_group = next((gid for gid in group_ids if len(groups[gid]) >= n_test), None)

    if total_items < n_test + n_dev:
        ratio_test = n_test / (n_test + n_dev)
        n_test = max(1, min(total_items, int(round(total_items * ratio_test))))
        n_dev = total_items - n_test
        shortfall = {
            "requested_test": requested_test, "requested_dev": requested_dev,
            "effective_test": n_test, "effective_dev": n_dev, "pool_size": total_items,
            "n_groups_total": n_groups_total,
        }

    rng_test = np.random.default_rng(seed)
    perm_test = [group_ids[i] for i in rng_test.permutation(len(group_ids))]
    test_groups, test_ids = _greedy_fill(perm_test, n_test)

    test_group_set = set(test_groups)
    remaining_group_ids = [g for g in group_ids if g not in test_group_set]
    rng_dev = np.random.default_rng(seed + 1)
    perm_dev = ([remaining_group_ids[i] for i in rng_dev.permutation(len(remaining_group_ids))]
                if remaining_group_ids else [])
    dev_groups, dev_ids = _greedy_fill(perm_dev, n_dev)

    return {
        "test_ids": test_ids, "dev_ids": dev_ids,
        "test_groups": test_groups, "dev_groups": dev_groups,
        "n_groups_total": n_groups_total,
        "n_test": len(test_ids), "n_dev": len(dev_ids),
        "group_disjoint_verified": len(test_group_set & set(dev_groups)) == 0,
        "shortfall": shortfall,
        "oversized_group": oversized_group,
        "fallback_item_level": not any_real_group,
    }

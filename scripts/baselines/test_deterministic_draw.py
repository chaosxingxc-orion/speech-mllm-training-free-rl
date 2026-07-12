"""scripts/baselines/test_deterministic_draw.py — offline (no data-root, no GPU) checks for
``deterministic_draw.py`` (续21-B① M1 item 1).

Each check is a bare ``test_*()`` function (zero args, plain ``assert``) so this file is directly
pytest-collectible (``pytest scripts/baselines/test_deterministic_draw.py -q``); ``main()`` ALSO
runs every ``test_*`` function standalone and prints a PASS/FAIL summary, mirroring this repo's
``scripts/baselines/test_phase_a_e2e.py`` / ``scripts/knowledge/test_kb_gate.py`` convention:

    python -u scripts/baselines/test_deterministic_draw.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/baselines
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import deterministic_draw as dd  # noqa: E402


def _synthetic_pool(n: int = 500, prefix: str = "item") -> list[str]:
    """A deterministic but NON-sorted, non-trivial-order pool -- important for the canonical-
    ordering test (the draw must not depend on this incidental order)."""
    ids = [f"{prefix}-{i:05d}" for i in range(n)]
    # scramble deterministically (independent of numpy's default_rng so this is a genuinely
    # different order than any permutation deterministic_draw itself might produce)
    ids = ids[::2] + ids[1::2]
    return ids


# ---------------------------------------------------------------------------------------------
# resolve_pool
# ---------------------------------------------------------------------------------------------

def test_resolve_pool_explicit_ids():
    ids = ["a", "b", "c"]
    assert dd.resolve_pool({"kind": "explicit_ids", "ids": ids}) == ids


def test_resolve_pool_file_bare_list():
    ids = ["x1", "x2", "x3"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pool.json"
        p.write_text(json.dumps(ids), encoding="utf-8")
        assert dd.resolve_pool({"kind": "file", "path": str(p)}) == ids


def test_resolve_pool_file_ids_dict():
    ids = ["y1", "y2"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pool.json"
        p.write_text(json.dumps({"ids": ids}), encoding="utf-8")
        assert dd.resolve_pool({"kind": "file", "path": str(p)}) == ids


def test_resolve_pool_unknown_kind_raises():
    raised = False
    try:
        dd.resolve_pool({"kind": "nonsense"})
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------------------------
# core determinism contract
# ---------------------------------------------------------------------------------------------

def test_same_inputs_byte_identical_manifest():
    pool = _synthetic_pool(300)
    spec = {"kind": "explicit_ids", "ids": pool}
    m1 = dd.deterministic_draw("ds-a", spec, n=40, draw_type="eligibility-split")
    m2 = dd.deterministic_draw("ds-a", spec, n=40, draw_type="eligibility-split")
    b1 = json.dumps(m1, ensure_ascii=False, sort_keys=True)
    b2 = json.dumps(m2, ensure_ascii=False, sort_keys=True)
    assert b1 == b2
    assert m1["ids"] == m2["ids"]


def test_same_inputs_byte_identical_manifest_on_disk():
    """Same as above, but through write_manifest -- proves the PERSISTED file bytes match, not
    just the in-memory dict (guards against any non-deterministic key ordering at json.dump time)."""
    pool = _synthetic_pool(120)
    spec = {"kind": "explicit_ids", "ids": pool}
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        m1 = dd.deterministic_draw("ds-b", spec, n=15, draw_type="confirmatory")
        m2 = dd.deterministic_draw("ds-b", spec, n=15, draw_type="confirmatory")
        p1 = dd.write_manifest(m1, out_dir=Path(td1))
        p2 = dd.write_manifest(m2, out_dir=Path(td2))
        assert p1.read_bytes() == p2.read_bytes()


def test_canonical_ordering_independent_of_input_order():
    """The SAME id set, handed in via a totally different original order, must draw the SAME ids
    (canonical sort-before-permute contract)."""
    pool_a = [f"item-{i:04d}" for i in range(200)]
    pool_b = list(reversed(pool_a))
    spec_a = {"kind": "explicit_ids", "ids": pool_a}
    spec_b = {"kind": "explicit_ids", "ids": pool_b}
    m_a = dd.deterministic_draw("ds-c", spec_a, n=25, draw_type="exploration-dev")
    m_b = dd.deterministic_draw("ds-c", spec_b, n=25, draw_type="exploration-dev")
    assert m_a["ids"] == m_b["ids"]
    assert m_a["pool_sha256"] == m_b["pool_sha256"]


def test_different_draw_types_yield_different_draws():
    """Distinct fixed-seed namespaces actually separate the three draw types (not just in theory --
    verified over the SAME pool/n)."""
    pool = _synthetic_pool(400)
    spec = {"kind": "explicit_ids", "ids": pool}
    ids_by_type = {
        dt: set(dd.deterministic_draw("ds-d", spec, n=50, draw_type=dt)["ids"])
        for dt in dd.DRAW_TYPES
    }
    types = list(dd.DRAW_TYPES)
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            assert ids_by_type[types[i]] != ids_by_type[types[j]], (
                f"{types[i]} and {types[j]} drew identical id sets -- seed namespaces not separated"
            )


def test_disjointness_across_draw_types_machine_verified():
    """The intended three-step recipe (eligibility-split -> exploration-dev [excludes prior] ->
    confirmatory [excludes both prior]) over ONE shared pool must yield PAIRWISE DISJOINT id sets
    -- the property this module's docstring claims and 续21-B①'s ruling relies on to retire the
    custodian/burn ritual (replaced by construction, not by convention)."""
    pool = _synthetic_pool(600)
    spec = {"kind": "explicit_ids", "ids": pool}

    m_elig = dd.deterministic_draw("ds-e", spec, n=60, draw_type="eligibility-split")
    m_dev = dd.deterministic_draw(
        "ds-e", spec, n=60, draw_type="exploration-dev", exclusion_lists=[m_elig["ids"]],
    )
    m_conf = dd.deterministic_draw(
        "ds-e", spec, n=60, draw_type="confirmatory",
        exclusion_lists=[m_elig["ids"], m_dev["ids"]],
    )

    s_elig, s_dev, s_conf = set(m_elig["ids"]), set(m_dev["ids"]), set(m_conf["ids"])
    assert not (s_elig & s_dev), "eligibility-split / exploration-dev overlap"
    assert not (s_elig & s_conf), "eligibility-split / confirmatory overlap"
    assert not (s_dev & s_conf), "exploration-dev / confirmatory overlap"
    assert len(s_elig) == 60 and len(s_dev) == 60 and len(s_conf) == 60


def test_exclusion_lists_never_appear_in_output():
    pool = _synthetic_pool(100)
    spec = {"kind": "explicit_ids", "ids": pool}
    excluded = pool[:80]  # exclude most of the pool
    m = dd.deterministic_draw("ds-f", spec, n=15, draw_type="confirmatory", exclusion_lists=[excluded])
    assert not (set(m["ids"]) & set(excluded))
    assert m["n_drawn"] == 15
    assert m["pool_size_after_exclusion"] == 20


def test_small_pool_shortfall_reported_honestly():
    pool = _synthetic_pool(10)
    spec = {"kind": "explicit_ids", "ids": pool}
    m = dd.deterministic_draw("ds-g", spec, n=40, draw_type="eligibility-split")
    assert m["n_drawn"] == 10
    assert m["shortfall"] is not None
    assert m["shortfall"]["requested"] == 40 and m["shortfall"]["drawn"] == 10


def test_duplicate_pool_ids_raise():
    spec = {"kind": "explicit_ids", "ids": ["a", "b", "a"]}
    raised = False
    try:
        dd.deterministic_draw("ds-h", spec, n=2, draw_type="eligibility-split")
    except ValueError:
        raised = True
    assert raised


def test_empty_pool_after_exclusion_raises():
    spec = {"kind": "explicit_ids", "ids": ["a", "b"]}
    raised = False
    try:
        dd.deterministic_draw("ds-i", spec, n=1, draw_type="eligibility-split",
                               exclusion_lists=[["a", "b"]])
    except ValueError:
        raised = True
    assert raised


def test_unknown_draw_type_raises():
    spec = {"kind": "explicit_ids", "ids": ["a", "b"]}
    raised = False
    try:
        dd.deterministic_draw("ds-j", spec, n=1, draw_type="not-a-real-type")
    except ValueError:
        raised = True
    assert raised


def test_seed_override_changes_the_draw():
    pool = _synthetic_pool(300)
    spec = {"kind": "explicit_ids", "ids": pool}
    m_default = dd.deterministic_draw("ds-k", spec, n=30, draw_type="eligibility-split")
    m_override = dd.deterministic_draw("ds-k", spec, n=30, draw_type="eligibility-split", seed=12345)
    assert m_default["seed"] == dd.DRAW_TYPE_SEEDS["eligibility-split"]
    assert m_override["seed"] == 12345
    assert m_default["ids"] != m_override["ids"]


def test_write_manifest_roundtrip():
    pool = _synthetic_pool(50)
    spec = {"kind": "explicit_ids", "ids": pool}
    m = dd.deterministic_draw("ds-l", spec, n=10, draw_type="exploration-dev", command="python foo.py")
    with tempfile.TemporaryDirectory() as td:
        p = dd.write_manifest(m, out_dir=Path(td))
        assert p.exists()
        loaded = dd.load_manifest("ds-l", "exploration-dev", out_dir=Path(td))
        assert loaded == m


# ---------------------------------------------------------------------------------------------
# standalone runner (mirrors test_kb_gate.py / test_phase_a_e2e.py / test_stats.py convention)
# ---------------------------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    results: dict[str, bool] = {}
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}", flush=True)
            results[name] = True
        except Exception as e:  # noqa: BLE001 -- one failing check must not stop the rest
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            results[name] = False
    all_pass = all(results.values())
    print("\n=== DETERMINISTIC_DRAW TEST ===")
    print(json.dumps(results, indent=2))
    print("DETERMINISTIC_DRAW_TEST_PASS" if all_pass else "DETERMINISTIC_DRAW_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

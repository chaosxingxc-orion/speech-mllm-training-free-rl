"""scripts/baselines/test_deterministic_draw.py — offline (no data-root, no GPU) checks for
``deterministic_draw.py`` (续21-B① M1 item 1; HARDENED 2026-07-13 ticket #38 item 3 / F-8
remediation, v4.2 doctoral review §3 F-8 + §6 P2_SPLIT_AND_CUSTODY).

Each check is a bare ``test_*()`` function (zero args, plain ``assert``) so this file is directly
pytest-collectible (``pytest scripts/baselines/test_deterministic_draw.py -q``); ``main()`` ALSO
runs every ``test_*`` function standalone and prints a PASS/FAIL summary, mirroring this repo's
``scripts/baselines/test_phase_a_e2e.py`` / ``scripts/knowledge/test_kb_gate.py`` convention:

    python -u scripts/baselines/test_deterministic_draw.py

Every ``write_manifest``/confirmatory ``deterministic_draw`` call below passes an EXPLICIT
``registry_dir=`` (a tmp dir, mirroring ``log_dir=``) — F-8(b)'s exposure registry now
self-registers on every ``write_manifest`` call, and a confirmatory draw READS it; omitting
``registry_dir`` would default to the real repo's ``_repro/draws/exposure_registry.json``, which
this test suite must NEVER touch.
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
# ticket #38 item 4 test fixtures: group-aware confirmatory draw. `_identity_group_manifest`
# degenerates the group-permutation-greedy-fill path to be BYTE-IDENTICAL to the plain item-level
# permutation (every group is a singleton, so "permute groups, take whole groups until n" over N
# singleton groups IS "permute items, take n items") -- lets tests that only care about the
# item-level contract keep their EXACT prior assertions (drawn count, disjointness, etc.) while
# satisfying the now-mandatory group_manifest param for EVERY draw type (F-8(a)).
# `_synthetic_group_manifest` is a REAL (non-singleton) grouping used by tests that actually
# exercise group semantics.
# ---------------------------------------------------------------------------------------------

def _identity_group_manifest(pool: list[str]) -> dict[str, str]:
    return {i: i for i in pool}


def _synthetic_group_manifest(pool: list[str], group_size: int = 4) -> dict[str, str]:
    """Consecutive items of the SORTED pool share a group id, `group_size` items per group --
    independent of any real dataset's scripts/loaders/group_key.py logic (deliberately so: this
    module never calls group_key_of itself, see deterministic_draw's own docstring)."""
    ordered = sorted(pool)
    return {iid: f"g{i // group_size}" for i, iid in enumerate(ordered)}


def _write_ids_manifest(path: Path, ids: list[str]) -> Path:
    """A minimal 'prior manifest' file -- just enough shape (``{"ids": [...]}``) for
    ``exclusion_manifest_paths``/``--exclude-manifest`` to read. NOT registered in any exposure
    registry (use ``_registered_prior`` below when a confirmatory draw's F-8(b) registry-coverage
    gate must also be satisfied)."""
    path.write_text(json.dumps({"ids": ids}), encoding="utf-8")
    return path


def _draw_and_register(dataset_key: str, spec: dict, n: int, draw_type: str, out_dir: Path,
                        registry_dir: Path, **kwargs) -> tuple[dict, Path]:
    """deterministic_draw + write_manifest, landing the manifest in ``out_dir`` AND registering it
    in ``registry_dir``'s exposure registry -- the REAL recipe shape F-8(b)'s confirmatory
    registry-coverage check now requires (a bare ``_write_ids_manifest`` fixture is no longer
    sufficient for a confirmatory draw's ``exclusion_manifest_paths``)."""
    m = dd.deterministic_draw(dataset_key, spec, n=n, draw_type=draw_type, registry_dir=registry_dir,
                               **kwargs)
    p = dd.write_manifest(m, out_dir=out_dir, log_dir=out_dir, registry_dir=registry_dir)
    return m, p


def _registered_throwaway_prior(dataset_key: str, out_dir: Path, registry_dir: Path,
                                 tag: str = "z") -> Path:
    """Registers ONE throwaway eligibility-split prior manifest for ``dataset_key``, drawn from a
    pool of ids in a namespace (``zz-registry-only-*``) that can NEVER intersect any of this test
    file's real synthetic pools (``item-*``) -- satisfies a confirmatory draw's F-8(b)
    exposure-registry-coverage requirement WITHOUT affecting that test's own exclusion-count math.
    Used by tests where the registry gate is a side requirement, not the thing under test."""
    disjoint_pool = [f"zz-registry-only-{tag}-{i}" for i in range(3)]
    spec = {"kind": "explicit_ids", "ids": disjoint_pool}
    gm = _identity_group_manifest(disjoint_pool)
    _m, p = _draw_and_register(dataset_key, spec, n=1, draw_type="eligibility-split",
                                out_dir=out_dir, registry_dir=registry_dir, group_manifest=gm)
    return p


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
    group_manifest = _identity_group_manifest(pool)
    m1 = dd.deterministic_draw("ds-a", spec, n=40, draw_type="eligibility-split",
                                group_manifest=group_manifest)
    m2 = dd.deterministic_draw("ds-a", spec, n=40, draw_type="eligibility-split",
                                group_manifest=group_manifest)
    b1 = json.dumps(m1, ensure_ascii=False, sort_keys=True)
    b2 = json.dumps(m2, ensure_ascii=False, sort_keys=True)
    assert b1 == b2
    assert m1["ids"] == m2["ids"]


def test_same_inputs_byte_identical_manifest_on_disk():
    """Same as above, but through write_manifest -- proves the PERSISTED file bytes match, not
    just the in-memory dict (guards against any non-deterministic key ordering at json.dump time).

    2026-07-13 (ticket #38 item 4 + item 3/F-8(b)): draw_type='confirmatory' now REQUIRES
    group_manifest + exclusion_manifest_paths + a non-empty, covering exposure registry -- a REAL
    prior eligibility-split is drawn+registered first (identical inputs both times), and BOTH
    confirmatory calls read the SAME registry_dir."""
    pool = _synthetic_pool(120)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    with tempfile.TemporaryDirectory() as td0, tempfile.TemporaryDirectory() as td1, \
            tempfile.TemporaryDirectory() as td2, tempfile.TemporaryDirectory() as tdr:
        registry_dir = Path(tdr)
        prior_path = _registered_throwaway_prior("ds-b", Path(td0), registry_dir, tag="b")
        m1 = dd.deterministic_draw("ds-b", spec, n=15, draw_type="confirmatory",
                                    group_manifest=group_manifest,
                                    exclusion_manifest_paths=[str(prior_path)],
                                    registry_dir=registry_dir)
        m2 = dd.deterministic_draw("ds-b", spec, n=15, draw_type="confirmatory",
                                    group_manifest=group_manifest,
                                    exclusion_manifest_paths=[str(prior_path)],
                                    registry_dir=registry_dir)
        p1 = dd.write_manifest(m1, out_dir=Path(td1), log_dir=Path(td1), registry_dir=registry_dir)
        p2 = dd.write_manifest(m2, out_dir=Path(td2), log_dir=Path(td2), registry_dir=registry_dir)
        assert p1.read_bytes() == p2.read_bytes()


def test_canonical_ordering_independent_of_input_order():
    """The SAME id set, handed in via a totally different original order, must draw the SAME ids
    (canonical sort-before-permute contract)."""
    pool_a = [f"item-{i:04d}" for i in range(200)]
    pool_b = list(reversed(pool_a))
    spec_a = {"kind": "explicit_ids", "ids": pool_a}
    spec_b = {"kind": "explicit_ids", "ids": pool_b}
    group_manifest = _identity_group_manifest(pool_a)  # same id SET either way
    m_a = dd.deterministic_draw("ds-c", spec_a, n=25, draw_type="exploration-dev",
                                 group_manifest=group_manifest)
    m_b = dd.deterministic_draw("ds-c", spec_b, n=25, draw_type="exploration-dev",
                                 group_manifest=group_manifest)
    assert m_a["ids"] == m_b["ids"]
    assert m_a["pool_sha256"] == m_b["pool_sha256"]


def test_different_draw_types_yield_different_draws():
    """Distinct fixed-seed namespaces actually separate the three draw types (not just in theory --
    verified over the SAME pool/n). 2026-07-13 (ticket #38 items 3+4): every draw type now needs
    group_manifest (identity, so this stays byte-equivalent to the pre-F-8 item-level path);
    'confirmatory' additionally needs exclusion_manifest_paths + a registered prior."""
    pool = _synthetic_pool(400)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        registry_dir = Path(tdr)
        prior_path = _registered_throwaway_prior("ds-d", Path(td), registry_dir, tag="d")
        ids_by_type = {}
        for dt in dd.DRAW_TYPES:
            kwargs = (
                {"exclusion_manifest_paths": [str(prior_path)], "registry_dir": registry_dir}
                if dt == "confirmatory" else {}
            )
            ids_by_type[dt] = set(dd.deterministic_draw(
                "ds-d", spec, n=50, draw_type=dt, group_manifest=group_manifest, **kwargs
            )["ids"])
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
    custodian/burn ritual (replaced by construction, not by convention).

    2026-07-13 (ticket #38 items 3+4): the confirmatory draw goes through write_manifest ->
    exclusion_manifest_paths + the exposure registry (both real eligibility-split/exploration-dev
    manifests are written -- i.e. self-registered -- BEFORE the confirmatory draw reads the
    registry) with an identity group_manifest (byte-equivalent to the item-level path, so the exact
    ``== 60`` counts still hold) -- also asserts the disjointness_proof block itself reports zero
    intersections."""
    pool = _synthetic_pool(600)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)

    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        d, registry_dir = Path(td), Path(tdr)
        m_elig, p_elig = _draw_and_register("ds-e", spec, n=60, draw_type="eligibility-split",
                                             out_dir=d, registry_dir=registry_dir,
                                             group_manifest=group_manifest)
        m_dev, p_dev = _draw_and_register("ds-e", spec, n=60, draw_type="exploration-dev",
                                           out_dir=d, registry_dir=registry_dir,
                                           group_manifest=group_manifest,
                                           exclusion_lists=[m_elig["ids"]])
        m_conf = dd.deterministic_draw(
            "ds-e", spec, n=60, draw_type="confirmatory",
            group_manifest=group_manifest, exclusion_manifest_paths=[str(p_elig), str(p_dev)],
            registry_dir=registry_dir,
        )

    s_elig, s_dev, s_conf = set(m_elig["ids"]), set(m_dev["ids"]), set(m_conf["ids"])
    assert not (s_elig & s_dev), "eligibility-split / exploration-dev overlap"
    assert not (s_elig & s_conf), "eligibility-split / confirmatory overlap"
    assert not (s_dev & s_conf), "exploration-dev / confirmatory overlap"
    assert len(s_elig) == 60 and len(s_dev) == 60 and len(s_conf) == 60

    proof = m_conf["disjointness_proof"]
    assert proof["all_zero"] is True
    assert set(proof["intersection_sizes"].values()) == {0}
    assert sorted(proof["exclusion_manifest_paths"]) == sorted([str(p_elig), str(p_dev)])


def test_exclusion_lists_never_appear_in_output():
    """2026-07-13 (ticket #38 items 3+4): 'confirmatory' requires group_manifest +
    exclusion_manifest_paths + registry coverage; the ACTUAL exclusion mechanism under test here
    is still the plain in-memory ``exclusion_lists=`` arg (unchanged), with a throwaway registered
    prior satisfying the registry gate without affecting the exclusion-count math (its ids live in
    a disjoint namespace -- see ``_registered_throwaway_prior``), so the exact
    ``pool_size_after_exclusion == 20`` assertion below is unaffected."""
    pool = _synthetic_pool(100)
    spec = {"kind": "explicit_ids", "ids": pool}
    excluded = pool[:80]  # exclude most of the pool
    group_manifest = _identity_group_manifest(pool)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        registry_dir = Path(tdr)
        prior_path = _registered_throwaway_prior("ds-f", Path(td), registry_dir, tag="f")
        m = dd.deterministic_draw("ds-f", spec, n=15, draw_type="confirmatory",
                                   group_manifest=group_manifest,
                                   exclusion_lists=[excluded],
                                   exclusion_manifest_paths=[str(prior_path)],
                                   registry_dir=registry_dir)
    assert not (set(m["ids"]) & set(excluded))
    assert m["n_drawn"] == 15
    assert m["pool_size_after_exclusion"] == 20


def test_small_pool_shortfall_reported_honestly():
    pool = _synthetic_pool(10)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m = dd.deterministic_draw("ds-g", spec, n=40, draw_type="eligibility-split",
                               group_manifest=group_manifest)
    assert m["n_drawn"] == 10
    assert m["shortfall"] is not None
    assert m["shortfall"]["requested"] == 40 and m["shortfall"]["drawn"] == 10


def test_duplicate_pool_ids_raise():
    spec = {"kind": "explicit_ids", "ids": ["a", "b", "a"]}
    group_manifest = {"a": "a", "b": "b"}
    raised = False
    try:
        dd.deterministic_draw("ds-h", spec, n=2, draw_type="eligibility-split",
                               group_manifest=group_manifest)
    except ValueError:
        raised = True
    assert raised


def test_empty_pool_after_exclusion_raises():
    spec = {"kind": "explicit_ids", "ids": ["a", "b"]}
    group_manifest = {"a": "a", "b": "b"}
    raised = False
    try:
        dd.deterministic_draw("ds-i", spec, n=1, draw_type="eligibility-split",
                               group_manifest=group_manifest, exclusion_lists=[["a", "b"]])
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
    group_manifest = _identity_group_manifest(pool)
    m_default = dd.deterministic_draw("ds-k", spec, n=30, draw_type="eligibility-split",
                                       group_manifest=group_manifest)
    m_override = dd.deterministic_draw("ds-k", spec, n=30, draw_type="eligibility-split",
                                        group_manifest=group_manifest, seed=12345)
    assert m_default["seed"] == dd.DRAW_TYPE_SEEDS["eligibility-split"]
    assert m_override["seed"] == 12345
    assert m_default["ids"] != m_override["ids"]


def test_write_manifest_roundtrip():
    pool = _synthetic_pool(50)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m = dd.deterministic_draw("ds-l", spec, n=10, draw_type="exploration-dev",
                               group_manifest=group_manifest, command="python foo.py")
    with tempfile.TemporaryDirectory() as td:
        p = dd.write_manifest(m, out_dir=Path(td), log_dir=Path(td), registry_dir=Path(td))
        assert p.exists()
        loaded = dd.load_manifest("ds-l", "exploration-dev", out_dir=Path(td))
        assert loaded == m


# ---------------------------------------------------------------------------------------------
# ticket #37 item 7: confirmatory hygiene (seed hard-error / refuse-overwrite / append-only log)
# ---------------------------------------------------------------------------------------------

def test_seed_override_confirmatory_hard_error():
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    raised = False
    try:
        dd.deterministic_draw("ds-m", spec, n=10, draw_type="confirmatory", seed=999)
    except ValueError:
        raised = True
    assert raised


def test_seed_override_still_allowed_for_non_confirmatory_types():
    """Existing behavior (test_seed_override_changes_the_draw) covers eligibility-split; this adds
    exploration-dev for symmetry -- both non-confirmatory types must still accept an override."""
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m_default = dd.deterministic_draw("ds-n", spec, n=10, draw_type="exploration-dev",
                                       group_manifest=group_manifest)
    m_override = dd.deterministic_draw("ds-n", spec, n=10, draw_type="exploration-dev",
                                        group_manifest=group_manifest, seed=777)
    assert m_override["seed"] == 777
    assert m_default["ids"] != m_override["ids"]


def test_write_manifest_refuses_overwrite_without_force_supersede():
    pool = _synthetic_pool(40)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m1 = dd.deterministic_draw("ds-o", spec, n=8, draw_type="eligibility-split",
                                group_manifest=group_manifest)
    with tempfile.TemporaryDirectory() as td:
        p1 = dd.write_manifest(m1, out_dir=Path(td), log_dir=Path(td), registry_dir=Path(td))
        original_bytes = p1.read_bytes()
        raised = False
        try:
            dd.write_manifest(m1, out_dir=Path(td), log_dir=Path(td), registry_dir=Path(td))
        except FileExistsError:
            raised = True
        assert raised
        assert p1.read_bytes() == original_bytes  # untouched by the refused re-write


def test_write_manifest_force_supersede_renames_old_and_writes_new():
    pool = _synthetic_pool(80)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m1 = dd.deterministic_draw("ds-p", spec, n=10, draw_type="eligibility-split",
                                group_manifest=group_manifest)
    m2 = dd.deterministic_draw("ds-p", spec, n=20, draw_type="eligibility-split",
                                group_manifest=group_manifest)  # different n -> different content
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        p1 = dd.write_manifest(m1, out_dir=d, log_dir=d, registry_dir=d)
        v1_bytes = p1.read_bytes()
        p2 = dd.write_manifest(m2, out_dir=d, force_supersede=True, log_dir=d, registry_dir=d)
        assert p2 == p1  # same path, fresh content
        assert json.loads(p2.read_text(encoding="utf-8"))["n_drawn"] == 20

        superseded = [f for f in os.listdir(td) if ".superseded." in f]
        assert len(superseded) == 1
        superseded_path = d / superseded[0]
        assert superseded_path.read_bytes() == v1_bytes  # old content preserved byte-for-byte
        assert json.loads(superseded_path.read_text(encoding="utf-8"))["n_drawn"] == 10


def test_draw_log_appends_one_jsonl_line_per_write_manifest_call():
    pool = _synthetic_pool(30)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m1 = dd.deterministic_draw("ds-q", spec, n=5, draw_type="eligibility-split",
                                group_manifest=group_manifest, command="python x.py")
    m2 = dd.deterministic_draw("ds-q2", spec, n=6, draw_type="exploration-dev",
                                group_manifest=group_manifest, command="python y.py")
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as log_td:
        d, ld = Path(td), Path(log_td)
        dd.write_manifest(m1, out_dir=d, log_dir=ld, registry_dir=ld)
        dd.write_manifest(m2, out_dir=d, log_dir=ld, registry_dir=ld)
        log_path = ld / dd.DRAW_LOG_FILENAME
        assert log_path.exists()
        lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        for entry, m in zip(lines, (m1, m2)):
            assert entry["dataset_key"] == m["dataset_key"]
            assert entry["draw_type"] == m["draw_type"]
            assert entry["seed"] == m["seed"]
            assert entry["pool_sha256"] == m["pool_sha256"]
            assert entry["ids_sha256"] == m["ids_sha256"]
            assert entry["command"] == m["command"]
            assert "utc_timestamp" in entry and entry["utc_timestamp"]


# ---------------------------------------------------------------------------------------------
# ticket #38 item 4: group-aware confirmatory draw (fail-closed requirements + real group
# semantics + disjointness_proof + byte-identical reproducibility under grouping).
# ---------------------------------------------------------------------------------------------

def test_confirmatory_without_group_manifest_raises():
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        prior_path = _write_ids_manifest(Path(td) / "prior.json", ids=[])
        raised = False
        try:
            dd.deterministic_draw("ds-r", spec, n=10, draw_type="confirmatory",
                                   exclusion_manifest_paths=[str(prior_path)],
                                   registry_dir=Path(tdr))
        except ValueError as e:
            raised = "group_manifest" in str(e)
        assert raised


def test_confirmatory_without_exclusion_index_raises():
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    raised = False
    try:
        dd.deterministic_draw("ds-s", spec, n=10, draw_type="confirmatory",
                               group_manifest=group_manifest)
    except ValueError as e:
        raised = "exclusion_manifest_paths" in str(e)
    assert raised


def test_confirmatory_without_either_raises_naming_both_missing():
    """Belt-and-suspenders: omitting BOTH REQUIRED args names both in the one raised error (not
    just the first it happens to check)."""
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    raised_msg = None
    try:
        dd.deterministic_draw("ds-t", spec, n=10, draw_type="confirmatory")
    except ValueError as e:
        raised_msg = str(e)
    assert raised_msg is not None
    assert "group_manifest" in raised_msg and "exclusion_manifest_paths" in raised_msg


def test_eligibility_split_without_group_manifest_raises():
    """2026-07-13 (F-8(a) remediation, v4.2 doctoral review item 3): REPLACES the old
    'only warns, never raises' behavior. Non-confirmatory draw types now hard-require
    group_manifest exactly like confirmatory does -- omitting it is a ValueError, not a printed
    warning."""
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    raised = False
    try:
        dd.deterministic_draw("ds-u", spec, n=10, draw_type="eligibility-split")
    except ValueError as e:
        raised = "group_manifest" in str(e)
    assert raised


def test_exploration_dev_without_group_manifest_raises():
    """Symmetry check: 'exploration-dev' is the other draw type the old carve-out applied to."""
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    raised = False
    try:
        dd.deterministic_draw("ds-u2", spec, n=10, draw_type="exploration-dev")
    except ValueError as e:
        raised = "group_manifest" in str(e)
    assert raised


def test_eligibility_split_with_identity_group_manifest_still_works():
    """The one-line migration path for a caller with no real per-item grouping: an EXPLICIT
    identity mapping degenerates to the old item-level behavior, intentionally and auditably."""
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    m = dd.deterministic_draw("ds-u3", spec, n=10, draw_type="eligibility-split",
                               group_manifest=_identity_group_manifest(pool))
    assert m["group_manifest_provided"] is True
    assert m["sampling_algorithm"].startswith("group-permutation-greedy-fill")
    assert m["n_drawn"] == 10


def test_group_never_split_across_a_real_multi_item_grouping():
    """The load-bearing group-aware invariant: for every group that appears ANYWHERE in the drawn
    ids, ALL of that group's members (per the group_manifest) are in the drawn set -- never a
    partial group. Uses a REAL (non-singleton, group_size=4) synthetic grouping."""
    pool = _synthetic_pool(200)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _synthetic_group_manifest(pool, group_size=4)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        registry_dir = Path(tdr)
        prior_path = _registered_throwaway_prior("ds-v", Path(td), registry_dir, tag="v")
        m = dd.deterministic_draw("ds-v", spec, n=50, draw_type="confirmatory",
                                   group_manifest=group_manifest,
                                   exclusion_manifest_paths=[str(prior_path)],
                                   registry_dir=registry_dir)

    assert m["group_manifest_provided"] is True
    assert m["sampling_algorithm"].startswith("group-permutation-greedy-fill")

    # invert group_manifest -> group_id -> members (restricted to the full synthetic pool, which
    # is also the full eligible pool here since the throwaway prior's ids are disjoint from `pool`)
    members_of: dict[str, list[str]] = {}
    for iid, gid in group_manifest.items():
        members_of.setdefault(gid, []).append(iid)

    drawn = set(m["ids"])
    touched_groups = {group_manifest[iid] for iid in drawn}
    for gid in touched_groups:
        full_group = set(members_of[gid])
        assert full_group <= drawn, (
            f"group {gid!r} was PARTIALLY drawn ({len(full_group & drawn)}/{len(full_group)}) -- "
            "a group must NEVER be split (ticket #38 item 4)"
        )

    # per_group_counts/group_ids_drawn are consistent with the actual drawn ids.
    assert set(m["group_ids_drawn"]) == touched_groups
    assert sum(m["per_group_counts"].values()) == m["n_drawn"]
    for gid, cnt in m["per_group_counts"].items():
        assert cnt == len(members_of[gid])

    # "take groups until n items reached" -- drawn count is >= n_requested (never undershoots
    # while eligible groups remain; may OVERSHOOT since group_size=4 doesn't evenly divide 50).
    assert m["n_drawn"] >= m["n_requested"]


def test_disjointness_proof_correct_when_group_manifest_and_prior_draws_both_real():
    """disjointness_proof must correctly report ZERO intersection against each REAL prior manifest
    path when the exclusion mechanism actually worked, over a REAL (non-singleton) grouping -- not
    just the identity-group case other tests use for byte-identical-legacy-behavior purposes."""
    pool = _synthetic_pool(300)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _synthetic_group_manifest(pool, group_size=5)
    identity_gm = _identity_group_manifest(pool)  # elig/dev use identity (item-level) grouping;

    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        d, registry_dir = Path(td), Path(tdr)
        m_elig, p_elig = _draw_and_register("ds-w", spec, n=40, draw_type="eligibility-split",
                                             out_dir=d, registry_dir=registry_dir,
                                             group_manifest=identity_gm)
        m_dev, p_dev = _draw_and_register("ds-w", spec, n=40, draw_type="exploration-dev",
                                           out_dir=d, registry_dir=registry_dir,
                                           group_manifest=identity_gm,
                                           exclusion_lists=[m_elig["ids"]])
        m_conf = dd.deterministic_draw(
            "ds-w", spec, n=40, draw_type="confirmatory",
            group_manifest=group_manifest, exclusion_manifest_paths=[str(p_elig), str(p_dev)],
            registry_dir=registry_dir,
        )

    proof = m_conf["disjointness_proof"]
    assert proof["all_zero"] is True
    assert proof["intersection_sizes"][str(p_elig)] == 0
    assert proof["intersection_sizes"][str(p_dev)] == 0
    # independently re-derive the same fact from the raw ids (never just trust the proof block
    # blindly -- this is the "machine-verified, not by construction alone" bar the ticket asks for)
    assert not (set(m_conf["ids"]) & set(m_elig["ids"]))
    assert not (set(m_conf["ids"]) & set(m_dev["ids"]))


def test_disjointness_proof_vacuous_true_when_no_exclusion_paths_given():
    """Non-confirmatory draw with no exclusion_manifest_paths at all: the proof block still exists
    (always computed, not confirmatory-only) and is vacuously all_zero=True (nothing to intersect
    against)."""
    pool = _synthetic_pool(50)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m = dd.deterministic_draw("ds-x", spec, n=10, draw_type="eligibility-split",
                               group_manifest=group_manifest)
    proof = m["disjointness_proof"]
    assert proof["exclusion_manifest_paths"] == []
    assert proof["intersection_sizes"] == {}
    assert proof["all_zero"] is True


def test_group_aware_confirmatory_byte_identical_reproducibility_preserved():
    """The core determinism contract (module docstring) must survive the group-aware code path
    too: identical (pool, group_manifest, exclusion_manifest_paths, seed-namespace) inputs ->
    byte-identical manifest, over a REAL (non-singleton) grouping this time (not the identity-group
    trick other tests use)."""
    pool = _synthetic_pool(150)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _synthetic_group_manifest(pool, group_size=3)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        registry_dir = Path(tdr)
        prior_path = _registered_throwaway_prior("ds-y", Path(td), registry_dir, tag="y")
        m1 = dd.deterministic_draw("ds-y", spec, n=30, draw_type="confirmatory",
                                    group_manifest=group_manifest,
                                    exclusion_manifest_paths=[str(prior_path)],
                                    registry_dir=registry_dir)
        m2 = dd.deterministic_draw("ds-y", spec, n=30, draw_type="confirmatory",
                                    group_manifest=group_manifest,
                                    exclusion_manifest_paths=[str(prior_path)],
                                    registry_dir=registry_dir)
    b1 = json.dumps(m1, ensure_ascii=False, sort_keys=True)
    b2 = json.dumps(m2, ensure_ascii=False, sort_keys=True)
    assert b1 == b2
    assert m1["ids"] == m2["ids"]
    assert m1["group_ids_drawn"] == m2["group_ids_drawn"]


def test_group_manifest_missing_ids_raises():
    """2026-07-13 (F-8(a) remediation): REPLACES the old 'silent gid=iid singleton fallback' test.
    An id in the (post-exclusion) pool but absent from group_manifest is now a hard ValueError --
    never a silent singleton fallback."""
    pool = _synthetic_pool(40)
    spec = {"kind": "explicit_ids", "ids": pool}
    # deliberately partial: only give a real group_key for HALF the pool.
    ordered = sorted(pool)
    partial_manifest = {iid: f"g{i // 4}" for i, iid in enumerate(ordered[:20])}
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        registry_dir = Path(tdr)
        prior_path = _registered_throwaway_prior("ds-z", Path(td), registry_dir, tag="z")
        raised_msg = None
        try:
            dd.deterministic_draw("ds-z", spec, n=15, draw_type="confirmatory",
                                   group_manifest=partial_manifest,
                                   exclusion_manifest_paths=[str(prior_path)],
                                   registry_dir=registry_dir)
        except ValueError as e:
            raised_msg = str(e)
    assert raised_msg is not None
    assert "missing" in raised_msg.lower() or "coverage" in raised_msg.lower()


def test_group_manifest_empty_group_key_raises():
    """A group id mapped to an EMPTY/whitespace-only string is also a hard error -- 'unknown group
    id' per F-8(a), not just an outright-missing entry."""
    pool = _synthetic_pool(20)
    spec = {"kind": "explicit_ids", "ids": pool}
    invalid_manifest = _identity_group_manifest(pool)
    invalid_manifest[pool[0]] = "   "  # whitespace-only -- invalid, not a real group label
    raised = False
    try:
        dd.deterministic_draw("ds-z2", spec, n=5, draw_type="eligibility-split",
                               group_manifest=invalid_manifest)
    except ValueError as e:
        raised = "invalid" in str(e).lower() or "coverage" in str(e).lower()
    assert raised


# ---------------------------------------------------------------------------------------------
# 2026-07-13 (ticket #38 item 3 / F-8 remediation, v4.2 doctoral review §3 F-8 + §6
# P2_SPLIT_AND_CUSTODY): exposure registry, force_supersede-refused-for-confirmatory, custody
# hashes, and duplicate-group-manifest-file detection.
# ---------------------------------------------------------------------------------------------

def test_write_manifest_self_registers_in_exposure_registry():
    pool = _synthetic_pool(30)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m = dd.deterministic_draw("ds-reg1", spec, n=5, draw_type="eligibility-split",
                               group_manifest=group_manifest)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        assert dd.load_exposure_registry(d) == []  # nothing registered yet at a fresh dir
        p = dd.write_manifest(m, out_dir=d, log_dir=d, registry_dir=d)
        entries = dd.load_exposure_registry(d)
        assert len(entries) == 1
        assert entries[0]["dataset_key"] == "ds-reg1"
        assert entries[0]["draw_type"] == "eligibility-split"
        assert entries[0]["manifest_path"] == str(p.resolve())
        assert entries[0]["ids_sha256"] == m["ids_sha256"]


def test_confirmatory_requires_nonempty_registry_for_its_dataset_key():
    """F-8(b): even with group_manifest + a non-empty exclusion_manifest_paths, confirmatory
    refuses to draw when the exposure registry has NOTHING registered for this dataset_key yet."""
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        # an ad-hoc, NEVER-registered exclusion file -- exists on disk, but the registry (tdr) has
        # nothing for this dataset_key.
        prior_path = _write_ids_manifest(Path(td) / "prior.json", ids=[])
        raised_msg = None
        try:
            dd.deterministic_draw("ds-reg2", spec, n=10, draw_type="confirmatory",
                                   group_manifest=group_manifest,
                                   exclusion_manifest_paths=[str(prior_path)],
                                   registry_dir=Path(tdr))
        except ValueError as e:
            raised_msg = str(e)
    assert raised_msg is not None
    assert "registry" in raised_msg.lower()


def test_confirmatory_exclusion_list_must_cover_full_registry_union():
    """F-8(b): TWO priors get registered for the same dataset_key, but the confirmatory draw's
    exclusion_manifest_paths names only ONE of them -- fail-closed."""
    pool = _synthetic_pool(80)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        d, registry_dir = Path(td), Path(tdr)
        m_elig, p_elig = _draw_and_register("ds-reg3", spec, n=10, draw_type="eligibility-split",
                                             out_dir=d, registry_dir=registry_dir,
                                             group_manifest=group_manifest)
        m_dev, p_dev = _draw_and_register("ds-reg3", spec, n=10, draw_type="exploration-dev",
                                           out_dir=d, registry_dir=registry_dir,
                                           group_manifest=group_manifest,
                                           exclusion_lists=[m_elig["ids"]])
        raised_msg = None
        try:
            dd.deterministic_draw("ds-reg3", spec, n=10, draw_type="confirmatory",
                                   group_manifest=group_manifest,
                                   exclusion_manifest_paths=[str(p_elig)],  # p_dev MISSING
                                   registry_dir=registry_dir)
        except ValueError as e:
            raised_msg = str(e)
    assert raised_msg is not None
    assert "registry" in raised_msg.lower() or "cover" in raised_msg.lower()


def test_confirmatory_registry_check_is_scoped_per_dataset_key():
    """A registered prior for a DIFFERENT dataset_key must never block (or satisfy) this
    dataset_key's confirmatory registry-coverage check."""
    pool_a = _synthetic_pool(40, prefix="itemA")
    pool_b = _synthetic_pool(40, prefix="itemB")
    spec_a = {"kind": "explicit_ids", "ids": pool_a}
    spec_b = {"kind": "explicit_ids", "ids": pool_b}
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        d, registry_dir = Path(td), Path(tdr)
        _draw_and_register("ds-other", spec_a, n=5, draw_type="eligibility-split",
                            out_dir=d, registry_dir=registry_dir,
                            group_manifest=_identity_group_manifest(pool_a))
        # ds-reg4 has NOTHING registered -- must still raise even though the registry is non-empty
        # overall (it's just for a different dataset_key).
        raised = False
        try:
            dd.deterministic_draw("ds-reg4", spec_b, n=5, draw_type="confirmatory",
                                   group_manifest=_identity_group_manifest(pool_b),
                                   exclusion_manifest_paths=[str(d / "nonexistent.json")],
                                   registry_dir=registry_dir)
        except (ValueError, FileNotFoundError):
            raised = True
        assert raised


def test_force_supersede_refused_for_confirmatory():
    """F-8(c): force_supersede=True is a hard ValueError for draw_type='confirmatory', REGARDLESS
    of whether a file already exists at the target path (checked before the existence test)."""
    pool = _synthetic_pool(60)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as tdr:
        d, registry_dir = Path(td), Path(tdr)
        prior_path = _registered_throwaway_prior("ds-fs1", d, registry_dir, tag="fs1")
        m = dd.deterministic_draw("ds-fs1", spec, n=10, draw_type="confirmatory",
                                   group_manifest=group_manifest,
                                   exclusion_manifest_paths=[str(prior_path)],
                                   registry_dir=registry_dir)
        raised = False
        try:
            dd.write_manifest(m, out_dir=d, force_supersede=True, log_dir=d, registry_dir=registry_dir)
        except ValueError as e:
            raised = "confirmatory" in str(e).lower()
        assert raised
        # nothing was written by the refused call -- no file exists yet at all for this manifest.
        assert not (d / f"{m['dataset_key']}__{m['draw_type']}.json").exists()


def test_force_supersede_still_allowed_for_non_confirmatory():
    pool = _synthetic_pool(30)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m1 = dd.deterministic_draw("ds-fs2", spec, n=5, draw_type="eligibility-split",
                                group_manifest=group_manifest)
    m2 = dd.deterministic_draw("ds-fs2", spec, n=6, draw_type="eligibility-split",
                                group_manifest=group_manifest)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        dd.write_manifest(m1, out_dir=d, log_dir=d, registry_dir=d)
        p2 = dd.write_manifest(m2, out_dir=d, force_supersede=True, log_dir=d, registry_dir=d)
        assert json.loads(p2.read_text(encoding="utf-8"))["n_drawn"] == 6


def test_manifest_has_custody_hash_fields():
    """F-8(d): every manifest now records group_manifest_hash / exclusion_definition_hash /
    pool_hash / code_sha."""
    pool = _synthetic_pool(30)
    spec = {"kind": "explicit_ids", "ids": pool}
    group_manifest = _identity_group_manifest(pool)
    m = dd.deterministic_draw("ds-hash1", spec, n=5, draw_type="eligibility-split",
                               group_manifest=group_manifest, exclusion_lists=[pool[:3]])
    assert m["pool_hash"] == m["pool_sha256"]
    assert isinstance(m["group_manifest_hash"], str) and len(m["group_manifest_hash"]) == 64
    assert isinstance(m["exclusion_definition_hash"], str) and len(m["exclusion_definition_hash"]) == 64
    assert isinstance(m["code_sha"], str) and len(m["code_sha"]) > 0

    # exclusion_definition_hash actually reflects the exclusion set: a different exclusion_lists
    # input yields a different hash (same everything else).
    m2 = dd.deterministic_draw("ds-hash1", spec, n=5, draw_type="eligibility-split",
                                group_manifest=group_manifest, exclusion_lists=[pool[:5]])
    assert m["exclusion_definition_hash"] != m2["exclusion_definition_hash"]

    # group_manifest_hash actually reflects the group_manifest input: a different grouping yields
    # a different hash (same everything else).
    m3 = dd.deterministic_draw("ds-hash1", spec, n=5, draw_type="eligibility-split",
                                group_manifest=_synthetic_group_manifest(pool, group_size=4),
                                exclusion_lists=[pool[:3]])
    assert m["group_manifest_hash"] != m3["group_manifest_hash"]


def test_load_group_manifest_file_bare_object():
    ids = ["a", "b", "c"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gm.json"
        p.write_text(json.dumps({"a": "g0", "b": "g0", "c": "g1"}), encoding="utf-8")
        loaded = dd.load_group_manifest_file(str(p))
        assert loaded == {"a": "g0", "b": "g0", "c": "g1"}


def test_load_group_manifest_file_pair_array():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gm.json"
        p.write_text(json.dumps([
            {"item_id": "a", "group_id": "g0"},
            {"item_id": "b", "group_id": "g1"},
        ]), encoding="utf-8")
        loaded = dd.load_group_manifest_file(str(p))
        assert loaded == {"a": "g0", "b": "g1"}


def test_load_group_manifest_file_pair_array_duplicate_raises():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gm.json"
        p.write_text(json.dumps([
            {"item_id": "a", "group_id": "g0"},
            {"item_id": "a", "group_id": "g1"},  # CONFLICTING duplicate
        ]), encoding="utf-8")
        raised = False
        try:
            dd.load_group_manifest_file(str(p))
        except ValueError as e:
            raised = "duplicate" in str(e).lower()
        assert raised


def test_load_group_manifest_file_bare_object_duplicate_key_raises():
    """A raw JSON object literally CANNOT have two keys named 'a' after json.loads' normal
    dict-building collapses them -- this proves the object_pairs_hook actually intercepts the
    duplicate BEFORE that collapse happens (hand-written duplicate-key JSON text, since Python's
    own json.dumps can never produce one)."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gm.json"
        p.write_text('{"a": "g0", "b": "g1", "a": "g2"}', encoding="utf-8")
        raised = False
        try:
            dd.load_group_manifest_file(str(p))
        except ValueError as e:
            raised = "duplicate" in str(e).lower()
        assert raised


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

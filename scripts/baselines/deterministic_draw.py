"""scripts/baselines/deterministic_draw.py — deterministic sampling (owner directive 续21-B①,
Decision-Log 2026-07-13: "采样隔离简化——确定性脚本+固定种子=无偏抽样全部；保留提交先于选择+
程序性防火墙，废除信标/全新会话/burn 仪式").

## What this replaces

Every draw layer in this repo before this module (``locked_split.py``'s ``LOCKED_TEST_SEED``,
the proposal's §5.7 "NIST beacon + independent co-signer + fresh-session custodian + burn on
read" ritual) used an ELABORATE anti-tampering protocol to guarantee a draw could not be
cherry-picked after the fact: a public randomness beacon (unpredictable seed), an independent
human co-signer, a brand-new AI session with no memory of prior attempts, and a "burn on read"
rule (an opened confirmatory artifact could never be re-used). The owner's 续21-B① ruling
replaces ALL of that with something simpler: **a plain deterministic script + a FIXED,
pre-committed seed** — honest evidence tier: **replayable, not selection-blind** (see the note
below this paragraph; "equally tamper-evident" was the prior, retracted framing). Unbiasedness no
longer comes from unpredictability of the
seed (nobody can force a favorable draw out of a FIXED seed either way — there is nothing to
"pick" once the seed is fixed and the code is deterministic) but from **commit-before-selection**:
the manifest this module writes (ids + sha256 + the exact command) must be committed to git BEFORE
any arm/prompt/threshold selection reads it, so a later "redo the draw and see if it helps" attempt
is visible in the commit history rather than silently invisible. This is the "procedural
firewall" the ruling explicitly KEEPS ("保留提交先于选择+程序性防火墙"); only the beacon/fresh-
session/burn *ritual* is retired, not the discipline itself.

**Honest evidence-tier statement** (2026-07-13, F-8 remediation — v4.2 doctoral review §3 F-8: a
prior version of this paragraph called the fixed-seed replacement "*equally* tamper-evident" to
the beacon/co-signer/burn ritual it replaces; that claim does not survive scrutiny and is retracted
here). A fixed, pre-committed seed over a PUBLIC pool makes a draw **replayable** — anyone can
re-run the exact manifest inputs and reproduce byte-identical ids, and any REDRAW attempt leaves a
visible trace via commit-before-selection — but it is explicitly **NOT selection-blind**: a public
pool + a public, pre-known seed means the exact confirmatory ids ARE computable by anyone before
the config/model/prompt freeze (see the review's F-3 finding on `preregistered
fixed-public-benchmark evaluation` vs. blinded confirmatory). This module's actual, defensible
claim is **"replayable, not selection-blind"** — never "equally tamper-evident" to a scheme whose
entire point was making the FINAL IDS THEMSELVES unknowable in advance. Genuine selection-blindness
needs either a still-secret (not yet fixed/committed) seed at draw time, or third-party one-shot
scoring after a full config freeze (see the review's P2_SPLIT_AND_CUSTODY alternatives).

## The three draw types (each its own fixed-seed namespace)

  - ``eligibility-split``  — draws the slice a dataset's pre-registration ELIGIBILITY check reads
    (e.g. the "is closed-book dev accuracy < 0.85" gate) — drawn FIRST, from the full pool.
  - ``exploration-dev``    — the dev slice Stage-1/M2 exploration tunes against — drawn from the
    pool MINUS whatever ``eligibility-split`` already took (pass its ids via ``exclusion_lists``).
  - ``confirmatory``       — the held-out slice a confirmatory pass scores — drawn from the pool
    MINUS both prior draws (pass both prior ids lists via ``exclusion_lists``).

Each draw type has its OWN fixed seed constant (``DRAW_TYPE_SEEDS`` below) — never derived from
the other two, never overridable implicitly — so a draw of one type can never coincide with
another type's permutation by construction, and chaining ``exclusion_lists`` correctly (as the
three-step recipe above does) makes the three draws PAIRWISE DISJOINT by construction too (see
``test_deterministic_draw.py``'s disjointness check, which verifies this end to end rather than
trusting the argument).

## Determinism contract

1. **Canonical ordering**: the resolved pool's ids are ``sorted()`` (plain lexicographic string
   sort) BEFORE any exclusion or permutation — this makes the draw depend ONLY on the *set* of
   pool ids (and the exclusion set, and the seed), never on the incidental order a loader/DB
   happened to hand them back in. Two callers who resolve "the same pool" via different code paths
   (e.g. a freshly re-queried DB vs. a cached id list) but land on the same id SET reproduce the
   identical draw.
2. **PCG64**: ``numpy.random.default_rng(seed)`` — numpy's modern default bit generator (PCG64),
   explicitly named here (not left as an implicit numpy-version detail) because reproducibility
   depends on it: a caller must use the same numpy major version's ``default_rng`` implementation
   to replay a draw bit-for-bit forever, which is why this module also stamps ``numpy_version`` and
   ``bit_generator`` into every manifest.
3. **No wall-clock, no hidden state**: the manifest embeds every input (dataset_key, pool_spec,
   n, draw_type, seed, exclusion summary) plus every output (ids, hashes) — replaying
   ``deterministic_draw`` with the identical arguments against the identical pool must yield a
   byte-identical manifest (``json.dumps(..., sort_keys=True)``), proven by
   ``test_deterministic_draw.py``.

## Confirmatory hygiene (2026-07-13, ticket #37 item 7 -- cheap correctness guards, NOT a custody
## ceremony; the owner explicitly rejected re-adding lockdown/beacon-style ritual)

  (a) ``--seed``/``seed=`` override is a HARD ERROR when ``draw_type='confirmatory'`` (still
      allowed, with a printed warning, for the other two draw types) — see ``deterministic_draw``'s
      own docstring.
  (b) ``write_manifest`` refuses to overwrite an existing manifest file unless
      ``force_supersede=True`` (CLI: ``--force-supersede``), in which case the old file is renamed
      to ``<name>.superseded.<UTC-timestamp>.json`` (never deleted) before the fresh one is written
      — EXCEPT ``draw_type='confirmatory'``, where ``force_supersede`` is now hard-REFUSED
      unconditionally (see item (f) below, F-8(c)).
  (c) every ``write_manifest`` call appends one JSON line (draw_type, seed used, pool_sha256,
      ids_sha256, UTC timestamp, command) to the append-only ``_repro/draws/draw_log.jsonl`` log
      (see ``_append_draw_log``) AND registers itself in the exposure registry (item (e) below).

## Isolation hardening (2026-07-13, ticket #38 v4.2 doctoral review item 3, F-8 remediation)

  (d) group-manifest coverage is now a HARD REQUIREMENT for EVERY draw type that supplies a
      ``group_manifest`` (not just ``confirmatory`` — ``eligibility-split``/``exploration-dev`` used
      to only WARN when omitted; now ALL THREE draw types in ``DRAW_TYPES`` hard-require a
      ``group_manifest`` covering 100% of the post-exclusion pool). A MISSING id or one mapped to an
      empty/invalid group key is a hard ``ValueError`` — the prior silent ``gid=iid`` singleton
      fallback for an uncovered id is REMOVED (see ``_validate_group_manifest_coverage``).
  (e) exposure registry: ``_repro/draws/exposure_registry.json`` (see ``EXPOSURE_REGISTRY_FILENAME``)
      enumerates EVERY draw manifest ever successfully ``write_manifest``'d (self-registering — see
      ``_register_draw_manifest``, called from ``write_manifest``). ``draw_type='confirmatory'``
      REQUIRES this registry to be non-empty for its ``dataset_key`` and machine-verifies its own
      ``exclusion_manifest_paths`` covers the registry's FULL union for that ``dataset_key`` —
      fail-closed (``ValueError``) if any registered prior manifest is not named in the exclusion
      list (see ``load_exposure_registry``).
  (f) ``force_supersede=True`` is now a hard ``ValueError`` when ``manifest['draw_type'] ==
      'confirmatory'`` — a confirmatory manifest is create-once, no exceptions; see
      ``write_manifest``.
  (g) every manifest now additionally records ``group_manifest_hash``, ``exclusion_definition_hash``,
      ``pool_hash`` (an explicitly-named alias of the pre-existing ``pool_sha256``, kept for
      terminology parity with the review's ask), and ``code_sha`` (``git rev-parse HEAD`` of this
      repo at draw time, best-effort — see ``_git_sha_of``).

## Usage

    # library
    from deterministic_draw import deterministic_draw, write_manifest
    manifest = deterministic_draw(
        dataset_key="squtr", pool_spec={"kind": "explicit_ids", "ids": [...]},
        n=40, draw_type="exploration-dev", exclusion_lists=[eligibility_ids],
    )
    write_manifest(manifest, out_dir=...)

    # CLI
    python scripts/baselines/deterministic_draw.py --dataset-key squtr \\
        --pool-ids-file pool.json --draw-type eligibility-split --n 40
    python scripts/baselines/deterministic_draw.py --dataset-key squtr \\
        --pool-ids-file pool.json --draw-type exploration-dev --n 40 \\
        --exclude-manifest _repro/deterministic_draws/squtr__eligibility-split.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/baselines
sys.path.insert(0, HERE)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "_repro" / "deterministic_draws"
DRAW_LOG_DIR = REPO_ROOT / "_repro" / "draws"          # ticket #37 item 7c: append-only draw log
DRAW_LOG_FILENAME = "draw_log.jsonl"
# F-8(b) (2026-07-13, v4.2 doctoral review item 3): the exposure registry lives ALONGSIDE the draw
# log (same directory) -- one JSON array file, self-registering (see _register_draw_manifest).
EXPOSURE_REGISTRY_FILENAME = "exposure_registry.json"

# ---------------------------------------------------------------------------------------------
# distinct fixed-seed namespaces — one per draw type (续21-B①: "each a distinct fixed seed
# namespace, seeds recorded in the manifest"). Brand-new constants, never used by any prior seed
# in this repo (SLICE_SEED=20260705, TEST_SEED=20261705, LOCKED_TEST_SEED=611741209,
# COHORT_SEED=20260711, NOISE_SEED=20260712 — see locked_split.py's own "burned seeds" audit) so a
# draw made under this module can never be confused with (or accidentally replay) an older,
# already-decision-informing draw.
# ---------------------------------------------------------------------------------------------
DRAW_TYPES = ("eligibility-split", "exploration-dev", "confirmatory")

DRAW_TYPE_SEEDS: dict[str, int] = {
    "eligibility-split": 721_001_101,
    "exploration-dev": 721_002_202,
    "confirmatory": 721_003_303,
}

SCHEMA_VERSION = 3  # v2 (2026-07-13, ticket #38 item 4): group-aware confirmatory draw fields
                     # (group_manifest_provided/sampling_algorithm/group_ids_drawn/
                     # per_group_counts/disjointness_proof) -- purely ADDITIVE, a v1 manifest
                     # (schema_version==1 or missing) never had these keys at all, never
                     # reinterpreted as if it did.
                     # v3 (2026-07-13, F-8(d), v4.2 doctoral review item 3): group_manifest_hash/
                     # exclusion_definition_hash/pool_hash/code_sha -- also purely ADDITIVE; a v2
                     # manifest never had these either.

# 2026-07-13 (ticket #38 item 4 -- group-aware confirmatory draw, Decision-Log 续26; HARDENED
# 2026-07-13 ticket #38 item 3 / F-8(a), v4.2 doctoral review §3 F-8 + §6 P2_SPLIT_AND_CUSTODY):
# 'confirmatory' HARD-REQUIRES both a group manifest (item_id -> group_key, reusing
# scripts/loaders/group_key.py's classification logic -- see deterministic_draw's own docstring)
# and an exposure-exclusion index (a list of PRIOR deterministic_draw manifest file paths,
# auto-loaded and unioned) -- missing either is a fail-closed hard error.
#
# F-8(a): ALL THREE draw types now hard-require group_manifest WHENEVER one is relevant to this
# draw -- the prior "eligibility-split/exploration-dev may omit it, warning only" carve-out is
# REMOVED (the review's exact finding: a silent item-level fallback for those two types could let
# correlated items of the same speaker/session/source-family land split across draws with no
# error). GROUP_REQUIRED_DRAW_TYPES now equals DRAW_TYPES in full; GROUP_RECOMMENDED_DRAW_TYPES no
# longer exists (nothing is "recommended but optional" any more -- everything in DRAW_TYPES is
# required). A caller with no real per-item grouping for eligibility-split/exploration-dev must
# pass an EXPLICIT identity mapping (``{iid: iid for iid in pool}``) -- an intentional, auditable
# choice that degenerates to the old item-level behavior, not a silent default.
GROUP_REQUIRED_DRAW_TYPES = DRAW_TYPES


# ---------------------------------------------------------------------------------------------
# pool resolution — pool_spec describes HOW to get the candidate id list; this module never
# reimplements dataset loading itself (that machinery already exists in redraw.py/locked_split.py)
# — it only knows how to reach it. "explicit_ids" needs nothing extra and is what every unit test
# uses; "file" / "redraw_full_pool" are lazy-imported convenience resolvers for real callers.
# ---------------------------------------------------------------------------------------------

def resolve_pool(pool_spec: dict) -> list[str]:
    """pool_spec -> list[str] of candidate ids (any order — canonicalized by the caller via sort).

    Supported ``kind`` values:
      "explicit_ids"      — {"kind": "explicit_ids", "ids": [...]}
      "file"               — {"kind": "file", "path": <json file>, "id_field": <optional, for a
                              list-of-dict/{"ids":[...]} shaped file; default reads a bare JSON
                              list or a {"ids": [...]} dict>}
      "redraw_full_pool"    — {"kind": "redraw_full_pool", "dataset_key": <str>} — lazy-imports
                              ``scripts/baselines/redraw.py`` and calls its ``full_pool_ids``
                              (audio-decode-stubbed, CPU/metadata-only pool reconstruction already
                              used by ``locked_split.py``) — never re-derived here.
    """
    kind = pool_spec.get("kind")
    if kind == "explicit_ids":
        return [str(i) for i in pool_spec["ids"]]
    if kind == "file":
        data = json.loads(Path(pool_spec["path"]).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("ids", data.get(pool_spec.get("id_field", "ids"), []))
        id_field = pool_spec.get("id_field")
        if id_field and data and isinstance(data[0], dict):
            return [str(d[id_field]) for d in data]
        return [str(i) for i in data]
    if kind == "redraw_full_pool":
        import redraw  # scripts/baselines/redraw.py (already on sys.path via HERE above)

        return [str(i) for i in redraw.full_pool_ids(pool_spec["dataset_key"])]
    raise ValueError(
        f"deterministic_draw.resolve_pool: unknown pool_spec kind {kind!r} "
        "(expected 'explicit_ids' | 'file' | 'redraw_full_pool')"
    )


def _sha256_of_ids(ids: list[str]) -> str:
    """Order-independent content fingerprint: sha256 over the SORTED id list, newline-joined."""
    h = hashlib.sha256()
    h.update("\n".join(sorted(ids)).encode("utf-8"))
    return h.hexdigest()


def _load_exclusion_manifest_ids(paths: list[str]) -> dict[str, set]:
    """The exposure-exclusion index (ticket #38 item 4b): ``paths`` -- a list of PRIOR
    ``deterministic_draw`` manifest file paths -- each auto-loaded and its ``"ids"`` read into its
    OWN set (keyed by the path string, kept separate so ``disjointness_proof`` can report a
    per-path intersection size, not just one opaque combined count). Raises the plain
    ``FileNotFoundError``/``KeyError`` a caller would get from a bad path/malformed file — never
    silently skips an unreadable exclusion manifest (that would be a silent leak in the very
    guard this mechanism exists to provide)."""
    out: dict[str, set] = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        out[str(p)] = {str(i) for i in data["ids"]}
    return out


def _validate_group_manifest_coverage(remaining_ids: list[str], group_manifest: dict[str, str],
                                       draw_type: str) -> None:
    """F-8(a) (2026-07-13, v4.2 doctoral review item 3): 100% group-manifest coverage is now a HARD
    requirement for EVERY draw type that supplies a ``group_manifest`` -- raises ``ValueError`` on:

      - a MISSING id: some id in ``remaining_ids`` (the POST-exclusion pool this draw actually
        samples from) has no entry in ``group_manifest`` at all.
      - an UNKNOWN/invalid group id: ``group_manifest`` maps some (remaining) id to a falsy/empty
        group key (``None``, ``''``, whitespace-only) -- present in the mapping but not a real
        group label.

    NEVER a silent ``gid=iid`` singleton fallback for either case (the prior behavior this
    remediation removes) -- a caller that genuinely wants item-level (no real grouping) behavior
    must pass an EXPLICIT identity mapping instead, an intentional, auditable choice.

    (Duplicate group-manifest entries are caught earlier, at FILE-LOAD time, by
    ``load_group_manifest_file`` below -- a plain python ``dict`` in memory cannot itself represent
    a duplicate key, so there is nothing to check for duplicates here.)
    """
    missing = sorted(i for i in remaining_ids if i not in group_manifest)
    invalid = sorted(
        i for i in remaining_ids
        if i in group_manifest and not str(group_manifest[i]).strip()
    )
    if missing or invalid:
        raise ValueError(
            f"deterministic_draw: draw_type={draw_type!r} group_manifest coverage is NOT 100% of "
            f"the post-exclusion pool -- {len(missing)} id(s) missing entirely "
            f"(first few: {missing[:5]}), {len(invalid)} id(s) map to an empty/invalid group key "
            f"(first few: {invalid[:5]}). F-8(a) remediation (2026-07-13): a missing/unknown group "
            "id is a HARD ERROR, never a silent gid=iid singleton fallback -- fix group_manifest to "
            "cover every id in the post-exclusion pool (or pass an explicit identity mapping for "
            "item-level draws) before drawing."
        )


def load_group_manifest_file(path) -> dict[str, str]:
    """Load a ``{item_id: group_key}`` mapping from a JSON file, detecting DUPLICATE ``item_id``
    keys at parse time (2026-07-13, F-8(a): a plain ``json.loads`` of a JSON *object* silently
    keeps only the LAST value for a repeated key -- indistinguishable from a clean file once
    parsed). Supports two on-disk shapes: a bare JSON object (``{"id1": "g1", ...}``) or a JSON
    array of ``{"item_id": ..., "group_id": ...}`` pairs (which CAN have genuine key collisions
    that a bare-object file structurally cannot). Raises ``ValueError`` naming every duplicated id
    if any are found -- never silently drops one.
    """
    raw = Path(path).read_text(encoding="utf-8")

    if raw.lstrip().startswith("["):
        pairs = json.loads(raw)
        seen: dict[str, str] = {}
        dupes: list[str] = []
        for entry in pairs:
            iid, gid = str(entry["item_id"]), str(entry["group_id"])
            if iid in seen and seen[iid] != gid:
                dupes.append(iid)
            seen[iid] = gid
        if dupes:
            raise ValueError(
                f"deterministic_draw.load_group_manifest_file({path!r}): {len(sorted(set(dupes)))} "
                f"duplicate item_id key(s) with CONFLICTING group_id values (first few: "
                f"{sorted(set(dupes))[:5]}) -- F-8(a) remediation: a duplicate/ambiguous group id "
                "is a hard error, never silently resolved by last-value-wins."
            )
        return seen

    # object_pairs_hook sees every (key, value) pair BEFORE python's dict construction collapses
    # duplicates -- this is the only way to detect a duplicate key in a bare JSON object at all.
    def _detect_dupes(pairs):
        seen: dict[str, str] = {}
        dupes: list[str] = []
        for k, v in pairs:
            if k in seen and seen[k] != v:
                dupes.append(k)
            seen[k] = v
        if dupes:
            raise ValueError(
                f"deterministic_draw.load_group_manifest_file({path!r}): {len(sorted(set(dupes)))} "
                f"duplicate item_id key(s) with CONFLICTING group_id values (first few: "
                f"{sorted(set(dupes))[:5]}) -- F-8(a) remediation: a duplicate/ambiguous group id "
                "is a hard error, never silently resolved by last-value-wins."
            )
        return seen

    return {str(k): str(v) for k, v in json.loads(raw, object_pairs_hook=_detect_dupes).items()}


def _hash_str_dict(d: dict[str, str]) -> str:
    """sha256 over a canonical (sort_keys) JSON dump -- used for ``group_manifest_hash``."""
    canonical = json.dumps(d, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_sha_of(repo_root) -> str:
    """Best-effort ``git rev-parse HEAD`` of ``repo_root`` -- never raises (degrades to an
    ``"UNKNOWN (...)"`` placeholder), mirroring ``scripts/knowledge/kb_schema.py``'s
    ``git_sha_of`` (duplicated rather than imported: scripts/baselines stays independent of
    scripts/knowledge, mirroring the layering discipline ``kb_batch_build.py``'s own docstring
    states in the other direction)."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"UNKNOWN ({exc})"


# ---------------------------------------------------------------------------------------------
# exposure registry (2026-07-13, ticket #38 item 3 / F-8(b), v4.2 doctoral review §6
# P2_SPLIT_AND_CUSTODY: "canonical exposure registry proves all previous manifests included").
# ONE JSON array file, self-registering: every successful ``write_manifest`` call appends itself
# (see ``_register_draw_manifest``). ``draw_type='confirmatory'`` REQUIRES this registry to be
# non-empty for its ``dataset_key`` and machine-verifies its own ``exclusion_manifest_paths``
# covers the registry's FULL union for that ``dataset_key`` -- fail-closed otherwise (see
# ``deterministic_draw``'s own confirmatory gate).
# ---------------------------------------------------------------------------------------------

def _exposure_registry_path(registry_dir: Path | None = None) -> Path:
    return (registry_dir if registry_dir is not None else DRAW_LOG_DIR) / EXPOSURE_REGISTRY_FILENAME


def load_exposure_registry(registry_dir: Path | None = None) -> list[dict]:
    """All entries ever registered (``[]`` if the registry file does not exist yet -- a fresh repo
    genuinely has no prior draws)."""
    p = _exposure_registry_path(registry_dir)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _register_draw_manifest(manifest: dict, manifest_path, registry_dir: Path | None = None) -> Path:
    """Append ONE entry (never rewrites/removes an existing one) recording that ``manifest`` was
    successfully persisted at ``manifest_path`` -- called from ``write_manifest`` on every
    successful write, so the registry is always a complete, self-maintained ledger of every draw
    this module ever wrote to disk (under whichever ``registry_dir``/``out_dir`` the caller used --
    tests pass a tmp ``registry_dir`` exactly like ``log_dir``/``out_dir``, so this NEVER touches
    the real repo's ``_repro/draws/exposure_registry.json`` during a test run)."""
    entries = load_exposure_registry(registry_dir)
    entries.append({
        "dataset_key": manifest["dataset_key"],
        "draw_type": manifest["draw_type"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "ids_sha256": manifest["ids_sha256"],
        "pool_sha256": manifest["pool_sha256"],
        "registered_at": datetime.now(timezone.utc).isoformat(),
    })
    p = _exposure_registry_path(registry_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------------------------
# the draw itself
# ---------------------------------------------------------------------------------------------

def deterministic_draw(
    dataset_key: str,
    pool_spec: dict,
    n: int,
    draw_type: str,
    seed: int | None = None,
    exclusion_lists: list[list[str]] | None = None,
    command: str | None = None,
    group_manifest: dict[str, str] | None = None,
    exclusion_manifest_paths: list[str] | None = None,
    registry_dir: Path | None = None,
) -> dict:
    """Pure function (no filesystem writes): resolve the pool, canonically sort it, exclude any
    ids in ``exclusion_lists``/``exclusion_manifest_paths``, permute with a PCG64 RNG seeded by
    ``seed`` (default: this ``draw_type``'s fixed namespace constant), take the first ``n`` (or,
    when ``group_manifest`` is given, whole GROUPS until ``n`` items are reached — see below) —
    and return the full manifest dict (NOT yet written to disk; see ``write_manifest``).

    Raises ``ValueError`` for an unknown ``draw_type``, an empty resolved pool, or a resolved pool
    with duplicate ids (a draw over a pool with dupes is not well-defined — mirrors
    ``redraw.full_pool_ids``'s own duplicate guard).

    2026-07-13 (ticket #37 item 7a, confirmatory hygiene): an explicit ``seed`` override is a HARD
    ERROR when ``draw_type='confirmatory'`` — the confirmatory draw is the one type where an
    overridden seed could look like cherry-picking after the fact (draw, look, redraw with a
    different seed if the first one was unfavorable), so it must ALWAYS use its fixed namespace
    constant, no exceptions. The other two draw types still accept an explicit override (debug/test
    only) — allowed exactly as before, now with a printed warning making the override visible in
    the run log too, not just in this docstring/the CLI help text.

    ## Group-aware confirmatory draw (2026-07-13, ticket #38 item 4)

    ``group_manifest`` — a ``{item_id: group_key}`` mapping (the caller typically builds this via
    ``functools.partial(scripts.loaders.group_key.group_key_of, dataset_key)`` applied over its own
    loaded rows, mirroring ``scripts/loaders/_common.py``'s ``draw_disjoint_grouped`` convention —
    this module deliberately does NOT call ``group_key_of`` itself, since ``pool_spec`` only
    resolves a bare id list, never full Row objects with ``meta``).

    2026-07-13 (F-8(a) remediation, v4.2 doctoral review item 3): ``group_manifest`` is now
    REQUIRED for EVERY draw type (``ValueError`` if omitted — see ``GROUP_REQUIRED_DRAW_TYPES``),
    and must cover 100% of the POST-EXCLUSION pool with a non-empty group key for every id — a
    missing id, OR one mapped to an empty/falsy group key, is a hard ``ValueError`` (see
    ``_validate_group_manifest_coverage``). The prior silent "id missing from the mapping falls
    back to its own id as a singleton group" behavior is REMOVED — a caller that genuinely wants
    plain item-level draws (no real grouping) must pass an EXPLICIT identity mapping
    (``{iid: iid for iid in pool}``), an intentional, auditable choice rather than an implicit
    default.

    ``exclusion_manifest_paths`` — a list of PRIOR ``deterministic_draw`` manifest FILE PATHS
    (typically the eligibility-split and/or exploration-dev manifests already written via
    ``write_manifest``); each is auto-loaded and its ``"ids"`` UNIONED into the exclusion set (see
    ``_load_exclusion_manifest_ids``) — a strictly PATH-based sibling of the existing
    ``exclusion_lists`` (already-resolved, in-memory id lists — kept, unchanged, for
    library callers/tests that already have the ids on hand and don't need the path-based
    auto-load or the per-path ``disjointness_proof`` this new parameter also produces).

    ``draw_type='confirmatory'`` HARD-REQUIRES both ``group_manifest`` and
    ``exclusion_manifest_paths`` (``ValueError`` if either is missing/empty) — a confirmatory draw
    is exactly the one place an ungrouped item-level draw (two correlated items of the same
    speaker/session/source-family landing on opposite sides of a prior draw vs. this one) or a
    missing exposure-exclusion check would silently reintroduce the very contamination this
    module's whole "pairwise disjoint by construction" design exists to prevent.

    2026-07-13 (F-8(b) remediation): ``draw_type='confirmatory'`` ADDITIONALLY requires a non-empty
    exposure registry (``_repro/draws/exposure_registry.json``, see ``load_exposure_registry``) for
    this ``dataset_key``, and machine-verifies its OWN ``exclusion_manifest_paths`` covers the
    registry's FULL union for that ``dataset_key`` — ``ValueError`` (fail-closed) if any registered
    prior manifest for this dataset is not named in ``exclusion_manifest_paths``. ``registry_dir``
    overrides the registry's directory (default ``DRAW_LOG_DIR`` — tests pass a tmp dir, mirroring
    ``write_manifest``'s ``log_dir``).

    When ``group_manifest`` is given, sampling permutes GROUPS instead of items: sort this pool's
    group ids (canonical, like the item-id sort above), permute with the SAME PCG64 RNG/seed
    namespace, then greedily accept WHOLE groups (never splitting one) until the drawn item count
    reaches ``n`` — mirrring ``scripts/loaders/_common.py``'s ``draw_disjoint_grouped`` greedy-fill
    algorithm, adapted here to a single target ``n`` (that function draws a disjoint test+dev PAIR;
    this module draws one slice at a time, chained via ``exclusion_lists``/
    ``exclusion_manifest_paths`` instead). This can OVERSHOOT ``n`` by the last group's size (never
    splits it) — recorded honestly via ``shortfall`` exactly like the item-level pool-too-small
    case, distinguished by ``shortfall.reason``.
    """
    if draw_type not in DRAW_TYPES:
        raise ValueError(f"deterministic_draw: draw_type={draw_type!r} not in {DRAW_TYPES}")
    if seed is not None and draw_type == "confirmatory":
        raise ValueError(
            "deterministic_draw: --seed/seed= override is NOT allowed for draw_type='confirmatory' "
            "(ticket #37 item 7a) -- the confirmatory draw must ALWAYS use its fixed seed namespace "
            f"({DRAW_TYPE_SEEDS['confirmatory']}); overriding it is exactly the cherry-picking "
            "pattern commit-before-selection exists to prevent. Omit seed for a confirmatory draw. "
            "The other draw types (eligibility-split / exploration-dev) still accept an explicit "
            "override for debug/test purposes."
        )
    if seed is not None:
        print(
            f"  [deterministic_draw] WARNING: overriding draw_type={draw_type!r}'s fixed seed "
            f"namespace ({DRAW_TYPE_SEEDS[draw_type]}) with seed={seed} -- debug/test only; a real "
            "pre-registered draw should NEVER do this.",
            flush=True,
        )

    # ---- F-8(a) (2026-07-13, v4.2 doctoral review item 3): group_manifest is now REQUIRED for
    # EVERY draw type in GROUP_REQUIRED_DRAW_TYPES (== DRAW_TYPES in full -- the prior
    # "eligibility-split/exploration-dev may omit it, warning only" carve-out is removed).
    # confirmatory ADDITIONALLY requires exclusion_manifest_paths (ticket #38 item 4, unchanged) --
    # both checked together so a caller missing BOTH sees one combined, complete error. ----
    missing_required = []
    if draw_type in GROUP_REQUIRED_DRAW_TYPES and group_manifest is None:
        missing_required.append("group_manifest")
    if draw_type == "confirmatory" and not exclusion_manifest_paths:
        missing_required.append("exclusion_manifest_paths")
    if missing_required:
        raise ValueError(
            f"deterministic_draw: draw_type={draw_type!r} REQUIRES group_manifest (an item_id -> "
            "group_key mapping, reusing scripts/loaders/group_key.py's classification logic -- or "
            "an explicit identity mapping {iid: iid for iid in pool} for a genuinely ungrouped "
            "item-level draw)" + (
                " and exclusion_manifest_paths (a non-empty list of prior deterministic_draw "
                "manifest file paths)" if draw_type == "confirmatory" else ""
            ) + f" -- missing: {missing_required}. F-8(a) remediation (2026-07-13): ALL draw types "
            "now hard-require group_manifest (no more warning-only carve-out for "
            "eligibility-split/exploration-dev); confirmatory additionally hard-requires "
            "exclusion_manifest_paths (ticket #38 item 4, fail-closed). An ungrouped or "
            "exposure-unchecked draw could silently reintroduce cross-side contamination or "
            "replay-exposed ids."
        )

    import numpy as np

    raw_pool = resolve_pool(pool_spec)
    if not raw_pool:
        raise ValueError(f"deterministic_draw: resolved pool for {dataset_key!r} is empty")
    if len(raw_pool) != len(set(raw_pool)):
        dupes = sorted({i for i in raw_pool if raw_pool.count(i) > 1})
        raise ValueError(
            f"deterministic_draw: resolved pool for {dataset_key!r} has {len(dupes)} duplicate "
            f"id(s) (first few: {dupes[:5]}) -- a draw needs unique pool ids"
        )

    # ---- canonical ordering: sort BEFORE any exclusion or permutation ----
    sorted_pool = sorted(raw_pool)
    pool_sha256 = _sha256_of_ids(sorted_pool)

    excluded: set[str] = set()
    for lst in exclusion_lists or []:
        excluded.update(str(i) for i in lst)
    # ticket #38 item 4b: the exposure-exclusion index -- prior manifest FILE PATHS, auto-loaded and
    # unioned (kept per-path too, for the disjointness_proof block below).
    exclusion_manifest_ids_by_path = _load_exclusion_manifest_ids(exclusion_manifest_paths or [])
    for ids in exclusion_manifest_ids_by_path.values():
        excluded.update(ids)
    remaining = [i for i in sorted_pool if i not in excluded]  # filter preserves sorted order
    if not remaining:
        raise ValueError(
            f"deterministic_draw: pool for {dataset_key!r} is empty after excluding "
            f"{len(excluded)} id(s) -- nothing left to draw from"
        )

    # F-8(a): 100% group-manifest coverage of the POST-exclusion pool -- see
    # _validate_group_manifest_coverage's own docstring for exactly what this catches. Guaranteed
    # group_manifest is not None here (the earlier gate already required it for every draw_type in
    # GROUP_REQUIRED_DRAW_TYPES).
    _validate_group_manifest_coverage(remaining, group_manifest, draw_type)

    # F-8(b): confirmatory ONLY -- the exposure registry must be non-empty for this dataset_key,
    # and this draw's OWN exclusion_manifest_paths must cover its FULL union -- fail-closed.
    if draw_type == "confirmatory":
        registry_entries = load_exposure_registry(registry_dir)
        same_dataset_entries = [e for e in registry_entries if e["dataset_key"] == dataset_key]
        if not same_dataset_entries:
            raise ValueError(
                f"deterministic_draw: draw_type='confirmatory' REQUIRES a non-empty exposure "
                f"registry entry for dataset_key={dataset_key!r} at "
                f"{_exposure_registry_path(registry_dir)} (F-8(b), 2026-07-13) -- no prior draw for "
                "this dataset is registered yet. A confirmatory draw with nothing registered as "
                "'already exposed' cannot prove its exclusion list covers everything prior — write "
                "the eligibility-split/exploration-dev manifests (via write_manifest, which "
                "self-registers) before drawing confirmatory."
            )
        registry_paths = {e["manifest_path"] for e in same_dataset_entries}
        supplied_paths = {str(Path(p).resolve()) for p in (exclusion_manifest_paths or [])}
        uncovered = registry_paths - supplied_paths
        if uncovered:
            raise ValueError(
                f"deterministic_draw: draw_type='confirmatory' exclusion_manifest_paths does not "
                f"cover {len(uncovered)} prior draw manifest(s) registered in the exposure registry "
                f"for dataset_key={dataset_key!r} (first few: {sorted(uncovered)[:5]}) -- F-8(b) "
                "remediation: a confirmatory draw's exclusion list must cover the registry's FULL "
                "union, fail-closed. Add these manifest paths to exclusion_manifest_paths (or "
                "verify the registry is not stale/wrong) before drawing."
            )

    seed_source = "explicit override"
    eff_seed = seed
    if eff_seed is None:
        eff_seed = DRAW_TYPE_SEEDS[draw_type]
        seed_source = f"draw-type fixed namespace ({draw_type!r})"

    rng = np.random.default_rng(eff_seed)  # PCG64 (numpy's default bit generator)
    n_requested = n

    # ---- sampling: group-permutation greedy-fill. F-8(a) (2026-07-13): group_manifest is now
    # REQUIRED for every draw_type (the earlier gate already raised if it were None), so
    # group_sampling_used is always True reachable via the public deterministic_draw() API --
    # `group_sampling_used`/the `else` branch below are kept as an explicit, still-correct
    # fallback rather than deleted (defensive: any future internal caller that constructs a
    # manifest without going through the public gate stays correct too), never silently dead. ----
    group_sampling_used = group_manifest is not None
    group_ids_drawn: list[str] | None = None
    per_group_counts: dict[str, int] | None = None
    n_groups_total_in_pool: int | None = None

    if group_sampling_used:
        # group_of restricted to `remaining` (the POST-exclusion pool) only -- an excluded id's
        # group membership is irrelevant to what's still eligible to draw from. F-8(a): NO silent
        # gid=iid fallback here any more -- _validate_group_manifest_coverage (above) already
        # proved every id in `remaining` has a real, non-empty group_manifest entry, so a direct
        # lookup is safe and the old .get(iid, iid) fallback would now be dead code, never taken.
        groups: dict[str, list[str]] = {}
        for iid in remaining:  # `remaining` preserves sorted order -> each groups[gid] list does too
            gid = group_manifest[iid]
            groups.setdefault(gid, []).append(iid)
        group_ids_sorted = sorted(groups.keys())
        n_groups_total_in_pool = len(group_ids_sorted)

        perm = rng.permutation(len(group_ids_sorted))
        chosen_groups: list[str] = []
        drawn_ids: list[str] = []
        for gi in perm:
            if len(drawn_ids) >= n:
                break  # never split a group -- stop BEFORE adding the next one once n is reached
            gid = group_ids_sorted[int(gi)]
            chosen_groups.append(gid)
            drawn_ids.extend(groups[gid])
        group_ids_drawn = chosen_groups
        per_group_counts = {gid: len(groups[gid]) for gid in chosen_groups}
        n_eff = len(drawn_ids)
    else:
        n_eff = min(n, len(remaining))
        perm = rng.permutation(len(remaining))
        drawn_idx = perm[:n_eff]
        drawn_ids = [remaining[int(i)] for i in drawn_idx]

    shortfall = None
    if n_eff != n_requested:
        shortfall = {
            "requested": n_requested, "drawn": n_eff, "pool_size_after_exclusion": len(remaining),
            "reason": (
                "group-mode: whole-group indivisibility caused an overshoot/undershoot relative to "
                "n_requested -- a group is NEVER split (ticket #38 item 4)" if group_sampling_used
                else "pool smaller than n_requested after exclusion"
            ),
        }

    # ---- disjointness_proof (ticket #38 item 4): machine-verify (not just trust-by-construction)
    # that this draw's ids do not overlap ANY of the exclusion_manifest_paths -- always computed
    # (empty/vacuously-True when no paths were given), not confirmatory-only. ----
    drawn_ids_set = set(drawn_ids)
    intersection_sizes = {
        path: len(drawn_ids_set & ids) for path, ids in exclusion_manifest_ids_by_path.items()
    }
    disjointness_proof = {
        "exclusion_manifest_paths": list(exclusion_manifest_paths or []),
        "intersection_sizes": intersection_sizes,
        "all_zero": all(v == 0 for v in intersection_sizes.values()),
    }

    # ---- F-8(d) (2026-07-13, v4.2 doctoral review item 3): custody hashes -- group_manifest_hash
    # (over exactly the {iid: gid} pairs USED, i.e. restricted to `remaining`), exclusion_definition_
    # hash (over the FULL resolved exclusion id SET -- both exclusion_lists and exclusion_manifest_
    # paths folded together, since that combined set is what actually defines "excluded" for this
    # draw), pool_hash (an explicitly-named alias of pool_sha256, kept for terminology parity with
    # the review's ask), code_sha (best-effort git rev-parse HEAD at draw time). ----
    group_manifest_hash = _hash_str_dict({iid: group_manifest[iid] for iid in remaining})
    exclusion_definition_hash = _sha256_of_ids(sorted(excluded)) if excluded else _sha256_of_ids([])
    code_sha = _git_sha_of(REPO_ROOT)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "draw_type": draw_type,
        "pool_spec": pool_spec,
        "n_requested": n_requested,
        "n_drawn": n_eff,
        "shortfall": shortfall,
        "seed": int(eff_seed),
        "seed_source": seed_source,
        "draw_type_seed_namespace": DRAW_TYPE_SEEDS[draw_type],
        "rng": "numpy.random.default_rng (PCG64)",
        "numpy_version": np.__version__,
        "canonical_ordering": "sorted(pool_ids) [lexicographic str sort] before exclusion/permutation",
        "pool_size": len(sorted_pool),
        "pool_sha256": pool_sha256,
        "n_excluded": len(excluded),
        "pool_size_after_exclusion": len(remaining),
        "remaining_pool_sha256": _sha256_of_ids(remaining),
        "ids": drawn_ids,
        "ids_sha256": _sha256_of_ids(drawn_ids),
        "command": command,
        # ---- ticket #38 item 4: group-aware confirmatory draw fields (all additive; None/empty
        # when group_manifest was not given -- see SCHEMA_VERSION's own comment) ----
        "group_manifest_provided": group_sampling_used,
        "sampling_algorithm": (
            "group-permutation-greedy-fill (never splits a group)" if group_sampling_used
            else "item-level-permutation"
        ),
        "n_groups_total_in_pool": n_groups_total_in_pool,
        "group_ids_drawn": group_ids_drawn,
        "per_group_counts": per_group_counts,
        "disjointness_proof": disjointness_proof,
        # ---- F-8(d): custody hashes (2026-07-13) ----
        "group_manifest_hash": group_manifest_hash,
        "exclusion_definition_hash": exclusion_definition_hash,
        "pool_hash": pool_sha256,
        "code_sha": code_sha,
    }
    return manifest


def _append_draw_log(manifest: dict, log_dir: Path | None = None) -> Path:
    """Ticket #37 item 7c: append ONE JSON line per draw to an append-only log — never rewritten,
    never truncated, one line per call. Recorded fields: draw_type, seed USED (the effective seed,
    post any override), pool_sha256, ids_sha256, a UTC timestamp, and the exact command (when the
    caller supplied one, e.g. the CLI's ``sys.argv`` reconstruction) — enough to independently
    audit every draw ever persisted through this module without re-deriving anything from the
    manifest file itself (which a caller could, in principle, later inspect/edit; the log is a
    separate, append-only trail)."""
    d = log_dir if log_dir is not None else DRAW_LOG_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / DRAW_LOG_FILENAME
    entry = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_key": manifest["dataset_key"],
        "draw_type": manifest["draw_type"],
        "seed": manifest["seed"],
        "pool_sha256": manifest["pool_sha256"],
        "ids_sha256": manifest["ids_sha256"],
        "command": manifest.get("command"),
    }
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return p


def write_manifest(manifest: dict, out_dir: Path | None = None, force_supersede: bool = False,
                    log_dir: Path | None = None, registry_dir: Path | None = None) -> Path:
    """Persist ``manifest`` (byte-identical for byte-identical inputs — ``sort_keys=True``) at
    ``<out_dir>/<dataset_key>__<draw_type>.json`` (default ``out_dir``:
    ``_repro/deterministic_draws/``), then append one line to the draw log (see
    ``_append_draw_log``) AND register this write in the exposure registry (see
    ``_register_draw_manifest``) — every successful write is now a self-registering event.

    2026-07-13 (ticket #37 item 7b, confirmatory hygiene): REFUSES to overwrite an existing
    manifest file in place — raises ``FileExistsError`` unless ``force_supersede=True``, in which
    case the OLD file is renamed (never deleted) to
    ``<dataset_key>__<draw_type>.superseded.<UTC-timestamp>.json`` before the fresh manifest is
    written at the original path (append-only: both the old and the new manifest remain on disk).
    Before this fix, a re-drawn/re-written manifest silently clobbered the prior file with no
    on-disk trace of what was there before — the commit-before-selection discipline (module
    docstring) relies on git history for that, but a LOCAL uncommitted overwrite (or a script bug)
    could still silently destroy a not-yet-committed manifest with no recovery path; this makes
    that failure mode impossible without an explicit, named opt-in.

    2026-07-13 (F-8(c), v4.2 doctoral review item 3): ``force_supersede=True`` is now a hard
    ``ValueError`` when ``manifest['draw_type'] == 'confirmatory'`` — REGARDLESS of whether a file
    already exists at the target path. A confirmatory manifest is create-once, no exceptions: the
    review's exact attack this closes is "draw confirmatory, look at the result, force_supersede a
    fresh draw if the first one was unfavorable" — the pre-existing refuse-overwrite gate alone did
    not prevent this since ``force_supersede`` was always available as an opt-out. If a confirmatory
    manifest is genuinely wrong (a real bug, wrong pool), the correct fix is a NEW ``dataset_key``/
    program id, never superseding the same one.

    ``log_dir`` overrides the append-only draw-log directory (default ``_repro/draws/``) — tests
    pass a tmp dir here (mirroring ``out_dir``) so running the test suite never writes into this
    repo's real ``_repro/draws/draw_log.jsonl``. ``registry_dir`` overrides the exposure-registry
    directory the SAME way (default ``_repro/draws/`` too — see ``EXPOSURE_REGISTRY_FILENAME``).
    """
    if force_supersede and manifest.get("draw_type") == "confirmatory":
        raise ValueError(
            "deterministic_draw.write_manifest: force_supersede=True is REFUSED for "
            "draw_type='confirmatory' (F-8(c), 2026-07-13) -- a confirmatory manifest is "
            "create-once; silently re-writing it over an existing file is exactly the "
            "redo-until-favorable pattern commit-before-selection exists to prevent. If a "
            "confirmatory manifest is genuinely wrong, fix the underlying bug and draw again under "
            "a NEW dataset_key/program id -- never supersede the same one."
        )
    d = out_dir or OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{manifest['dataset_key']}__{manifest['draw_type']}.json"
    if p.exists():
        if not force_supersede:
            raise FileExistsError(
                f"deterministic_draw.write_manifest: {p} already exists -- refusing to overwrite a "
                "draw manifest in place (ticket #37 item 7b). Pass force_supersede=True (CLI: "
                "--force-supersede) to rename the existing file to "
                f"'{p.stem}.superseded.<UTC-timestamp>.json' (append-only, never deleted) and write "
                "the fresh manifest at the original path."
            )
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        superseded_path = d / f"{manifest['dataset_key']}__{manifest['draw_type']}.superseded.{ts}.json"
        p.rename(superseded_path)
        print(f"  [deterministic_draw] force_supersede=True: renamed existing {p} -> "
              f"{superseded_path} (never deleted)", flush=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _append_draw_log(manifest, log_dir=log_dir)
    _register_draw_manifest(manifest, p, registry_dir=registry_dir)
    return p


def load_manifest(dataset_key: str, draw_type: str, out_dir: Path | None = None) -> dict:
    d = out_dir or OUT_DIR
    return json.loads((d / f"{dataset_key}__{draw_type}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _cli_pool_spec(args: argparse.Namespace) -> dict:
    if args.pool_ids_file:
        return {"kind": "file", "path": args.pool_ids_file, "id_field": args.pool_id_field}
    if args.pool_dataset_key:
        return {"kind": "redraw_full_pool", "dataset_key": args.pool_dataset_key}
    raise SystemExit("deterministic_draw: pass --pool-ids-file or --pool-dataset-key")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-key", required=True)
    ap.add_argument("--draw-type", required=True, choices=DRAW_TYPES)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=None,
                     help="override this draw-type's fixed seed namespace (debug/test only -- a "
                          "real pre-registered draw should NEVER override the namespace default). "
                          "HARD ERROR (ValueError) if --draw-type=confirmatory (ticket #37 item 7a).")
    ap.add_argument("--pool-ids-file", default=None, help="JSON file: a list of ids, or {'ids': [...]}")
    ap.add_argument("--pool-id-field", default=None, help="if --pool-ids-file is a list-of-dict, the id field name")
    ap.add_argument("--pool-dataset-key", default=None,
                     help="resolve the pool via redraw.full_pool_ids(<this key>) instead of a file")
    ap.add_argument("--exclude-manifest", action="append", default=[],
                     help="path to a prior deterministic_draw manifest whose 'ids' are excluded "
                          "(repeatable -- e.g. pass the eligibility-split manifest when drawing "
                          "exploration-dev, and both prior manifests when drawing confirmatory). "
                          "2026-07-13 (ticket #38 item 4): this is now ALSO the exposure-exclusion "
                          "index deterministic_draw's own disjointness_proof machine-verifies -- "
                          "REQUIRED (non-empty) for --draw-type confirmatory.")
    ap.add_argument("--group-manifest-file", default=None,
                     help="F-8(a), 2026-07-13: path to a JSON {item_id: group_key} mapping (e.g. "
                          "built via scripts/loaders/group_key.py's group_key_of over the caller's "
                          "own loaded rows), or a JSON array of {'item_id':..., 'group_id':...} "
                          "pairs -- REQUIRED for every --draw-type now (no more optional/warn-only "
                          "carve-out). Duplicate item_id keys with conflicting group_id values are "
                          "a hard error (see load_group_manifest_file).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--registry-dir", default=None,
                     help="F-8(b), 2026-07-13: override the exposure-registry directory (default "
                          "_repro/draws/, same as the draw log) -- confirmatory reads/verifies "
                          "against this registry; write_manifest self-registers into it.")
    ap.add_argument("--force-supersede", action="store_true",
                     help="ticket #37 item 7b: allow overwriting an existing manifest at this "
                          "(dataset_key, draw_type) path -- the existing file is renamed to "
                          "'<name>.superseded.<UTC-timestamp>.json' (never deleted), never silently "
                          "clobbered. Without this flag, write_manifest refuses (FileExistsError). "
                          "REFUSED unconditionally (ValueError) for --draw-type confirmatory "
                          "(F-8(c), 2026-07-13).")
    ap.add_argument("--dry-run", action="store_true", help="compute + print; write nothing")
    args = ap.parse_args()

    pool_spec = _cli_pool_spec(args)
    group_manifest = None
    if args.group_manifest_file:
        group_manifest = load_group_manifest_file(args.group_manifest_file)

    registry_dir = Path(args.registry_dir) if args.registry_dir else None
    command = "python " + shlex.join(sys.argv)
    manifest = deterministic_draw(
        dataset_key=args.dataset_key, pool_spec=pool_spec, n=args.n, draw_type=args.draw_type,
        seed=args.seed, command=command,
        exclusion_manifest_paths=args.exclude_manifest or None,
        group_manifest=group_manifest, registry_dir=registry_dir,
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return
    out_dir = Path(args.out_dir) if args.out_dir else None
    p = write_manifest(manifest, out_dir=out_dir, force_supersede=args.force_supersede,
                        registry_dir=registry_dir)
    print(f"wrote {p} (n_drawn={manifest['n_drawn']} seed={manifest['seed']} "
          f"ids_sha256={manifest['ids_sha256'][:16]}...)")


if __name__ == "__main__":
    main()

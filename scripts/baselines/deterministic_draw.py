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
replaces ALL of that with something simpler and *equally* tamper-evident: **a plain deterministic
script + a FIXED, pre-committed seed**. Unbiasedness no longer comes from unpredictability of the
seed (nobody can force a favorable draw out of a FIXED seed either way — there is nothing to
"pick" once the seed is fixed and the code is deterministic) but from **commit-before-selection**:
the manifest this module writes (ids + sha256 + the exact command) must be committed to git BEFORE
any arm/prompt/threshold selection reads it, so a later "redo the draw and see if it helps" attempt
is visible in the commit history rather than silently invisible. This is the "procedural
firewall" the ruling explicitly KEEPS ("保留提交先于选择+程序性防火墙"); only the beacon/fresh-
session/burn *ritual* is retired, not the discipline itself.

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
      to ``<name>.superseded.<UTC-timestamp>.json`` (never deleted) before the fresh one is written.
  (c) every ``write_manifest`` call appends one JSON line (draw_type, seed used, pool_sha256,
      ids_sha256, UTC timestamp, command) to the append-only ``_repro/draws/draw_log.jsonl`` log
      (see ``_append_draw_log``).

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

SCHEMA_VERSION = 1


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
) -> dict:
    """Pure function (no filesystem writes): resolve the pool, canonically sort it, exclude any
    ids in ``exclusion_lists``, permute with a PCG64 RNG seeded by ``seed`` (default: this
    ``draw_type``'s fixed namespace constant), take the first ``n`` — and return the full manifest
    dict (NOT yet written to disk; see ``write_manifest``).

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
    remaining = [i for i in sorted_pool if i not in excluded]  # filter preserves sorted order
    if not remaining:
        raise ValueError(
            f"deterministic_draw: pool for {dataset_key!r} is empty after excluding "
            f"{len(excluded)} id(s) -- nothing left to draw from"
        )

    seed_source = "explicit override"
    eff_seed = seed
    if eff_seed is None:
        eff_seed = DRAW_TYPE_SEEDS[draw_type]
        seed_source = f"draw-type fixed namespace ({draw_type!r})"

    rng = np.random.default_rng(eff_seed)  # PCG64 (numpy's default bit generator)
    n_requested = n
    n_eff = min(n, len(remaining))
    perm = rng.permutation(len(remaining))
    drawn_idx = perm[:n_eff]
    drawn_ids = [remaining[int(i)] for i in drawn_idx]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "draw_type": draw_type,
        "pool_spec": pool_spec,
        "n_requested": n_requested,
        "n_drawn": n_eff,
        "shortfall": None if n_eff == n_requested else {
            "requested": n_requested, "drawn": n_eff, "pool_size_after_exclusion": len(remaining),
        },
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
                    log_dir: Path | None = None) -> Path:
    """Persist ``manifest`` (byte-identical for byte-identical inputs — ``sort_keys=True``) at
    ``<out_dir>/<dataset_key>__<draw_type>.json`` (default ``out_dir``:
    ``_repro/deterministic_draws/``), then append one line to the draw log (see
    ``_append_draw_log``).

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

    ``log_dir`` overrides the append-only draw-log directory (default ``_repro/draws/``) — tests
    pass a tmp dir here (mirroring ``out_dir``) so running the test suite never writes into this
    repo's real ``_repro/draws/draw_log.jsonl``.
    """
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
                          "exploration-dev, and both prior manifests when drawing confirmatory)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force-supersede", action="store_true",
                     help="ticket #37 item 7b: allow overwriting an existing manifest at this "
                          "(dataset_key, draw_type) path -- the existing file is renamed to "
                          "'<name>.superseded.<UTC-timestamp>.json' (never deleted), never silently "
                          "clobbered. Without this flag, write_manifest refuses (FileExistsError).")
    ap.add_argument("--dry-run", action="store_true", help="compute + print; write nothing")
    args = ap.parse_args()

    pool_spec = _cli_pool_spec(args)
    exclusion_lists = []
    for p in args.exclude_manifest:
        exclusion_lists.append(json.loads(Path(p).read_text(encoding="utf-8"))["ids"])

    command = "python " + shlex.join(sys.argv)
    manifest = deterministic_draw(
        dataset_key=args.dataset_key, pool_spec=pool_spec, n=args.n, draw_type=args.draw_type,
        seed=args.seed, exclusion_lists=exclusion_lists or None, command=command,
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return
    out_dir = Path(args.out_dir) if args.out_dir else None
    p = write_manifest(manifest, out_dir=out_dir, force_supersede=args.force_supersede)
    print(f"wrote {p} (n_drawn={manifest['n_drawn']} seed={manifest['seed']} "
          f"ids_sha256={manifest['ids_sha256'][:16]}...)")


if __name__ == "__main__":
    main()

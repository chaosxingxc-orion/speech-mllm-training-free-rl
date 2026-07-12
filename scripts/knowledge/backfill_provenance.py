"""scripts/knowledge/backfill_provenance.py — ticket #37 (engineering remediation) item 6.

For every EXISTING KB source in the real store (built before ``kb_build.build_source``'s
2026-07-13 provenance extension — ``code_git_sha``/``code_git_dirty``/``embedder_revision``/
``normalization``/``index_backend_params``/``content_hash_schema_version`` on
``kb_schema.SourceManifest``), compute a SIDECAR ``provenance.sidecar.json`` next to its
``manifest.json`` — WITHOUT touching ``manifest.json`` itself. This store is append-only by
design (see ``kb_build.build_source``'s refuse-overwrite/supersede-archive gate, RI item 8) — an
existing manifest is never rewritten in place, so this script only ever ADDS a new file next to
an existing one, never edits or deletes anything.

Per-source sidecar contents (best-effort; each field is ``None`` when not recoverable from disk
alone):

  - ``manifest_content_hash`` — the source's own (pre-existing) ``manifest.content_hash``, if any.
  - ``content_hash_recomputed_v1_now`` — ``kb_schema.content_hash_of(..., schema_version=1)``
    computed FRESH from the source's own ``values.jsonl``/``keys.npy`` bytes + ``from_item_id``s on
    disk, using the ORIGINAL (schema v1, pre-item-6) formula — the SAME inputs
    ``manifest_content_hash`` was originally computed from, so this is the closest
    apples-to-apples recomputation possible. ``None`` if either file is missing (e.g. an archived-
    only / partially-corrupted source).
  - ``content_hash_note`` — an explicit caveat: the recomputation uses THIS backfill's git sha
    (the ORIGINAL build-time sha was never recorded pre-item-6 and cannot be recovered after the
    fact), so a mismatch against ``manifest_content_hash`` may simply reflect that input differing,
    not necessarily byte drift in ``values.jsonl``/``keys.npy`` themselves.
  - ``code_git_sha_at_backfill`` — this repo's ``git rev-parse HEAD`` AT BACKFILL TIME (never
    confused with a build-time value — see the note above).
  - ``embedder_revision_guess`` — ``kb_embed.embedder_revision_of(token)`` where ``token`` is
    ``manifest.embedder_token`` or (for a pre-ticket-#25 source with no ``embedder_token``)
    ``kb_embed.infer_embedder_token(manifest.embedder)`` — best-effort, may be ``None``.
  - ``normalization_assumed`` — ``"l2"``: every embedder this repo has ever shipped L2-normalizes
    its output (see ``kb_embed``'s shared ``_l2`` helper) — an ASSUMPTION recorded explicitly (this
    script never re-derives it from the persisted vectors), not an independent measurement.
  - ``backfilled`` — always ``true``.
  - ``backfill_date`` — UTC ISO timestamp of this run.

Real store, DEV-safe: this MOVES/WRITES nothing except new ``provenance.sidecar.json`` files —
read-only with respect to every existing ``manifest.json``/``values.jsonl``/``keys.npy``. No GPU,
no eval/confirmatory-data consumption.

Run (real store):
    python -u scripts/knowledge/backfill_provenance.py
    python -u scripts/knowledge/backfill_provenance.py --dry-run     # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/knowledge
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def backfill_one(source: str, dry_run: bool = False) -> dict:
    import kb_embed
    from kb_schema import content_hash_of, git_sha_of, load_manifest, load_values, source_dir

    d = source_dir(source)
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        return {"source": source, "status": "skipped-no-manifest", "has_computable_content_hash": None}
    sidecar_path = d / "provenance.sidecar.json"
    if sidecar_path.exists():
        return {"source": source, "status": "skipped-sidecar-exists", "has_computable_content_hash": None}

    manifest = load_manifest(source)
    values_path, keys_path = d / "values.jsonl", d / "keys.npy"
    repo_root = Path(__file__).resolve().parents[2]  # W1 repo root
    code_git_sha_now = git_sha_of(repo_root)

    content_hash_recomputed = None
    if values_path.exists() and keys_path.exists():
        try:
            values = load_values(source)
            from_item_ids = [v.provenance.get("from_item_id") for v in values]
            content_hash_recomputed = content_hash_of(
                values_path, keys_path, from_item_ids, code_git_sha_now, schema_version=1,
            )
        except Exception as e:  # noqa: BLE001 -- degrade to "not computable", never abort the sweep
            content_hash_recomputed = None
            print(f"  [backfill_provenance] {source}: content_hash recompute failed "
                  f"({type(e).__name__}: {e})", flush=True)

    token = manifest.embedder_token or kb_embed.infer_embedder_token(manifest.embedder)
    embedder_revision_guess = kb_embed.embedder_revision_of(token) if token else None

    sidecar = {
        "source": source,
        "manifest_content_hash": manifest.content_hash,
        "content_hash_recomputed_v1_now": content_hash_recomputed,
        "content_hash_note": (
            "recomputed with the ORIGINAL (schema_version=1) content_hash_of formula against the "
            "source's CURRENT on-disk bytes + this backfill's CURRENT git sha (NOT the build-time "
            "sha, which was never recorded pre-item-6) -- a mismatch against manifest_content_hash "
            "may reflect the git-sha input alone, not necessarily byte drift in "
            "values.jsonl/keys.npy; see code_git_sha_at_backfill."
        ),
        "code_git_sha_at_backfill": code_git_sha_now,
        "embedder_revision_guess": embedder_revision_guess,
        "normalization_assumed": "l2",
        "backfilled": True,
        "backfill_date": datetime.now(timezone.utc).isoformat(),
    }
    if not dry_run:
        sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "source": source, "status": "would-backfill" if dry_run else "backfilled",
        "has_computable_content_hash": content_hash_recomputed is not None,
    }


def main() -> int:
    from kb_schema import list_sources

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report only; write no sidecar files")
    args = ap.parse_args()

    sources = list_sources()
    results = [backfill_one(s, dry_run=args.dry_run) for s in sources]
    print(json.dumps(results, indent=2, ensure_ascii=False))

    n_total = len(sources)
    n_computable = sum(1 for r in results if r.get("has_computable_content_hash"))
    n_skipped_sidecar = sum(1 for r in results if r["status"] == "skipped-sidecar-exists")
    n_skipped_no_manifest = sum(1 for r in results if r["status"] == "skipped-no-manifest")
    print(
        f"\n[backfill_provenance] {n_total} source(s) total; {n_computable}/{n_total} got a "
        f"computable content hash; {n_skipped_sidecar} already had a sidecar (skipped); "
        f"{n_skipped_no_manifest} had no manifest.json (skipped) -- "
        f"{'DRY-RUN, nothing written' if args.dry_run else 'sidecars written for the rest'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

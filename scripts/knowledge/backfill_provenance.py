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

## Amend mode (2026-07-13, ticket #38 item 6)

``--amend`` writes a SECOND, SEPARATE sidecar — ``provenance.sidecar.v2.json`` — next to
``manifest.json``, for every real KB source, classifying it into exactly one of two buckets. This
is a bare "amend + note" operation: it NEVER touches ``manifest.json`` (append-only store, same
discipline as ``kb_build.build_source``'s refuse-overwrite gate) and NEVER touches the pre-existing
``provenance.sidecar.json`` (v1, written by the default mode above) — only a brand-new v2 file is
added, or the source is skipped if a v2 sidecar already exists for it.

  - ``"qrels-conditioned-DEV-smoke"`` — for the squtr corpus-side sources built under the LEGACY
    ``'qrels-mini'`` mode (``kb_batch_build.build_squtr_corpus_source``'s pre-ticket-#38 default,
    now an explicit opt-in) — named ``squtr-fiqa-clean__corpus__<embedder>__text-keyed`` (the two
    real ones on this store: glap + omni-embed-nemotron, 310 qrels-linked evidence docs each).
    Cites the doctoral review finding F'-3 (Decision-Log 续26: "squtr 确证检索语料重建为官方全语
    料...310 库永久降为 qrels-conditioned DEV smoke") — these sources are gold-linked docs +
    sampled distractors CONDITIONED on the qrels/eval slice, NOT evaluation-independent, and must
    never be mistaken for a confirmatory-grade corpus.
  - ``"legacy/incomplete-provenance"`` — every OTHER real KB source (built before ticket #37 item
    6's provenance extension landed, or otherwise missing the fuller provenance fields) — an
    honest, distinctly-named bucket, never silently merged with the squtr classification above.

Run (real store):
    python -u scripts/knowledge/backfill_provenance.py --amend
    python -u scripts/knowledge/backfill_provenance.py --amend --dry-run   # report only, write nothing
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


# ---------------------------------------------------------------------------------------------
# amend mode (2026-07-13, ticket #38 item 6) -- see module docstring's "Amend mode" section.
# ---------------------------------------------------------------------------------------------

AMEND_SCHEMA_VERSION = 2

# Prefix identifying the squtr corpus-side sources built under the LEGACY 'qrels-mini' mode
# (kb_batch_build.build_squtr_corpus_source's pre-ticket-#38 default) -- the two real ones on this
# store (glap / omni-embed-nemotron) both match this exact prefix; a prefix match (not a hardcoded
# 2-name list) so this correctly classifies any future qrels-mini build under the SAME legacy
# naming convention too, not just the two known today.
QRELS_MINI_LEGACY_SOURCE_PREFIX = "squtr-fiqa-clean__corpus__"

CLASSIFICATION_QRELS_MINI = "qrels-conditioned-DEV-smoke"
CLASSIFICATION_LEGACY_INCOMPLETE = "legacy/incomplete-provenance"


def classification_for(source: str) -> str:
    """The exact two-bucket classification ticket #38 item 6 asks for -- prefix-matched, not a
    hardcoded source-name list (see ``QRELS_MINI_LEGACY_SOURCE_PREFIX``'s own comment)."""
    if source.startswith(QRELS_MINI_LEGACY_SOURCE_PREFIX):
        return CLASSIFICATION_QRELS_MINI
    return CLASSIFICATION_LEGACY_INCOMPLETE


def _classification_reason(classification: str) -> str:
    if classification == CLASSIFICATION_QRELS_MINI:
        return (
            "qrels-conditioned corpus-side KB build (kb_batch_build.build_squtr_corpus_source's "
            "LEGACY 'qrels-mini' mode, the pre-ticket-#38 default) -- gold-linked docs + sampled "
            "distractors CONDITIONED on the qrels/eval slice, NOT evaluation-independent. Cites "
            "doctoral review finding F'-3 (Decision-Log 续26: \"squtr 确证检索语料重建为官方全语料"
            "...310 库永久降为 qrels-conditioned DEV smoke\") -- permanently reclassified, never "
            "to be used as a confirmatory-grade / evaluation-independent corpus."
        )
    return (
        "pre-ticket-#37-item-6 (or otherwise incomplete) KB source build -- provenance fields "
        "(code_git_sha/code_git_dirty/embedder_revision/normalization/index_backend_params/"
        "content_hash_schema_version) were never fully captured at build time; see this source's "
        "v1 provenance.sidecar.json (default backfill mode above) for the best-effort "
        "recomputation. Classified 'legacy/incomplete-provenance' as a distinct, honest bucket "
        "from the squtr qrels-conditioned-DEV-smoke sources above -- never conflated with them."
    )


def amend_one(source: str, dry_run: bool = False) -> dict:
    """Write ONE ``provenance.sidecar.v2.json`` for ``source`` -- NEVER touches ``manifest.json``
    or the (possibly already-present) v1 ``provenance.sidecar.json``. Skips (no write, no error) a
    source with no ``manifest.json`` at all, or one that already has a v2 sidecar (append-only:
    this script never overwrites its own prior output either)."""
    from kb_schema import source_dir

    d = source_dir(source)
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        return {"source": source, "status": "skipped-no-manifest", "classification": None}
    v2_path = d / "provenance.sidecar.v2.json"
    if v2_path.exists():
        return {"source": source, "status": "skipped-v2-sidecar-exists", "classification": None}

    classification = classification_for(source)
    sidecar = {
        "source": source,
        "amend_schema_version": AMEND_SCHEMA_VERSION,
        "classification": classification,
        "classification_reason": _classification_reason(classification),
        "amended_date": datetime.now(timezone.utc).isoformat(),
        "amended_by": "scripts/knowledge/backfill_provenance.py --amend (ticket #38 item 6)",
    }
    if not dry_run:
        v2_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "source": source, "status": "would-amend" if dry_run else "amended",
        "classification": classification,
    }


def main() -> int:
    from kb_schema import list_sources

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report only; write no sidecar files")
    ap.add_argument("--amend", action="store_true",
                     help="ticket #38 item 6: write provenance.sidecar.v2.json (classification "
                          "only -- 'qrels-conditioned-DEV-smoke' | 'legacy/incomplete-provenance') "
                          "instead of the default v1 content_hash/git-sha/embedder-revision "
                          "backfill above. Never touches manifest.json or the v1 sidecar.")
    args = ap.parse_args()

    sources = list_sources()

    if args.amend:
        results = [amend_one(s, dry_run=args.dry_run) for s in sources]
        print(json.dumps(results, indent=2, ensure_ascii=False))
        n_total = len(sources)
        n_qrels_mini = sum(1 for r in results if r["classification"] == CLASSIFICATION_QRELS_MINI)
        n_legacy = sum(1 for r in results if r["classification"] == CLASSIFICATION_LEGACY_INCOMPLETE)
        n_skipped_v2 = sum(1 for r in results if r["status"] == "skipped-v2-sidecar-exists")
        n_skipped_no_manifest = sum(1 for r in results if r["status"] == "skipped-no-manifest")
        print(
            f"\n[backfill_provenance --amend] {n_total} source(s) total; {n_qrels_mini} classified "
            f"{CLASSIFICATION_QRELS_MINI!r}; {n_legacy} classified "
            f"{CLASSIFICATION_LEGACY_INCOMPLETE!r}; {n_skipped_v2} already had a v2 sidecar "
            f"(skipped); {n_skipped_no_manifest} had no manifest.json (skipped) -- "
            f"{'DRY-RUN, nothing written' if args.dry_run else 'v2 sidecars written for the rest'}",
            flush=True,
        )
        return 0

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

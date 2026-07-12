"""scripts/knowledge/archive_legacy_squtr_sources.py — ticket #37 (engineering remediation) item 3.

Archives (NEVER deletes) the OLD 4-field ``squtr__<embedder>__<key_org>__<value_org>`` KB sources
whose VALUES are the FiQA QUERY's own text — the P0-1 wrong-object bug (see
``kb_batch_build.build_squtr_corpus_source``'s module docstring for the full history: the same bug
class also hit vocalbench-knowledge). These are superseded by the corpus-side replacement sources
(``squtr-fiqa-clean__corpus__<embedder>__text-keyed``, 310 qrels-linked evidence docs each, built by
``build_squtr_corpus_source``) that ``run_mock.source_name_for`` now PREFERS whenever one exists on
disk (see that module's ``CORPUS_SOURCE_NAME_FOR`` routing table).

## Why not ``kb_build.build_source(supersede=True)``

That mechanism archives an existing source dir and then immediately REBUILDS a fresh one under the
SAME name (it needs ``records=`` to rebuild from) — appropriate when a source is being replaced
in-place under its own name. Here the replacement lives under a COMPLETELY DIFFERENT name (the
corpus-side sources, already built) — there is nothing to rebuild under the OLD ``squtr__*__*__*``
names; the entire point is to STOP using them. This script is therefore a bare "archive + note"
operation: it moves each legacy source dir to
``<kb_root>/archive/<source>__superseded_<UTC-timestamp>/`` (the EXACT same archive-path convention
``kb_build.build_source``'s own supersede path uses) and writes a ``SUPERSEDED_NOTE.json`` sidecar
inside the archived directory recording why/when/what superseded it — but does NOT rebuild
anything under the vacated name. Once archived, ``kb_schema.source_dir(legacy_name)`` no longer
resolves to a manifest at all, so ``kb_retrieve.load_source(legacy_name)``/
``kb_schema.load_manifest(legacy_name)`` naturally raise ``FileNotFoundError`` for anyone still
trying to consume it by its old name — no special-casing needed, the move itself is the guard.

Real store, DEV-safe: this MOVES (never deletes) directories under ``SPEECHRL_KB_DIR``. No GPU, no
eval/confirmatory-data consumption.

Run (real store):
    python -u scripts/knowledge/archive_legacy_squtr_sources.py            # archive for real
    python -u scripts/knowledge/archive_legacy_squtr_sources.py --dry-run  # list only, touch nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/knowledge
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SUPERSEDE_REASON = (
    "P0-1 wrong-object: values are queries; superseded by corpus-side sources "
    "(ticket #37 item 3, 2026-07-13)"
)


def find_legacy_squtr_sources() -> list[str]:
    """Every real KB source named ``squtr__<embedder>__<key_org>__<value_org>`` (the OLD 4-field,
    query-valued convention) -- excludes the NEW ``__corpus__`` sources (those are the
    replacement, never archived) and anything not literally prefixed ``squtr__``."""
    from kb_schema import list_sources

    return sorted(
        name for name in list_sources()
        if name.startswith("squtr__") and "__corpus__" not in name
    )


def archive_source(source: str, dry_run: bool = False) -> dict:
    """Archive ONE source dir (never delete) -- mirrors ``kb_build.build_source``'s own supersede
    move exactly (identical archive-path convention), plus a ``SUPERSEDED_NOTE.json`` sidecar
    recording why. Returns ``{"source", "status", "archived_path"}``; ``status`` is one of
    ``"archived"`` / ``"would-archive"`` (dry-run) / ``"skipped-no-manifest"`` (nothing there to
    archive -- e.g. already archived by a prior run)."""
    import shutil

    from kb_schema import kb_root, source_dir

    d = source_dir(source)
    if not (d / "manifest.json").exists():
        return {"source": source, "status": "skipped-no-manifest", "archived_path": None}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived_dir = kb_root() / "archive" / f"{source}__superseded_{ts}"
    if dry_run:
        return {"source": source, "status": "would-archive", "archived_path": str(archived_dir)}

    archived_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(archived_dir))
    note = {
        "source": source,
        "superseded_at": datetime.now(timezone.utc).isoformat(),
        "reason": SUPERSEDE_REASON,
        "note": ("archived by scripts/knowledge/archive_legacy_squtr_sources.py "
                  "(ticket #37 item 3) -- never deleted; values.jsonl/keys.npy/manifest.json are "
                  "byte-intact inside this directory."),
    }
    (archived_dir / "SUPERSEDED_NOTE.json").write_text(
        json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  [archive_legacy_squtr_sources] {source}: archived -> {archived_dir} (never deleted)",
          flush=True)
    return {"source": source, "status": "archived", "archived_path": str(archived_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="list what WOULD be archived; touch nothing")
    args = ap.parse_args()

    legacy = find_legacy_squtr_sources()
    print(f"[archive_legacy_squtr_sources] {len(legacy)} legacy squtr__* source(s) found "
          f"(excludes '__corpus__' sources): {legacy}", flush=True)
    results = [archive_source(s, dry_run=args.dry_run) for s in legacy]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

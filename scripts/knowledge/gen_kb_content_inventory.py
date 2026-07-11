"""scripts/knowledge/gen_kb_content_inventory.py — ONE-TIME, READ-ONLY content-hash inventory of
every EXISTING persisted KB source (RI mechanical remediation 2026-07-12, item 8).

Why: ``kb_schema.SourceManifest.content_hash`` (added 2026-07-12, same remediation pass) is a
sha256 fingerprint over a source's ACTUAL PERSISTED BYTES (``values.jsonl`` + ``keys.npy`` + sorted
``from_item_id``s + the code git sha that built them) -- unlike the pre-existing ``build_hash``
(pure metadata: dataset/embedder/seed/n_entries as strings, which cannot detect a re-embed or a
corrupted file with the same metadata). Every source built BEFORE this field existed has
``content_hash: None`` in its own on-disk ``manifest.json`` (backward-compatible default) and this
script does NOT rewrite those manifests in place (that would be exactly the in-place-overwrite
pattern this same remediation pass forbids elsewhere, RI item 8's refuse-overwrite gate) -- it only
READS every existing source once and records a computed content_hash in a SEPARATE, external
inventory file, so the current state of every real KB source is captured/auditable without
touching the sources themselves.

Retrospective caveat (stated honestly): the ``code_git_sha`` component of a NEWLY-computed
content_hash here is the CURRENT repo HEAD at inventory time, not the actual git sha at each
source's original build time (which was never recorded for these pre-2026-07-12 sources) -- so
these are NOT directly comparable to a content_hash computed by a FUTURE ``kb_build.build_source``
call for the same source (which would legitimately use its own build-time HEAD). Their purpose here
is a same-day baseline snapshot: re-running this script later and diffing detects whether any of
these sources' bytes changed since 2026-07-12, not "this matches what the original builder would
have computed".

Run (read-only; touches nothing under SPEECHRL_KB_DIR):
    cd projects/speech-mllm-training-free-rl
    python scripts/knowledge/gen_kb_content_inventory.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # scripts/knowledge -- bare `import kb_schema`

REPO_ROOT = HERE.parents[1]
OUT_PATH = REPO_ROOT / "_repro" / "kb_content_inventory_20260712.json"


def main() -> int:
    import kb_schema

    root = kb_schema.kb_root() / "knowledge_base"
    code_git_sha = kb_schema.git_sha_of(REPO_ROOT)
    generated_at = datetime.now(timezone.utc).isoformat()

    entries = []
    if not root.exists():
        print(f"WARNING: {root} does not exist -- nothing to inventory", file=sys.stderr)
    else:
        for source_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            source = source_dir.name
            manifest_path = source_dir / "manifest.json"
            values_path = source_dir / "values.jsonl"
            keys_path = source_dir / "keys.npy"

            entry: dict = {"source": source}
            if not manifest_path.exists():
                entry["status"] = "SKIPPED (no manifest.json -- not a standard kb_build source)"
                entries.append(entry)
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                entry["status"] = f"SKIPPED (manifest.json unreadable: {e})"
                entries.append(entry)
                continue

            entry["declared_content_hash"] = manifest.get("content_hash")  # None expected (pre-existing)
            entry["build_hash"] = manifest.get("build_hash")
            entry["dataset"] = manifest.get("dataset")
            entry["embedder"] = manifest.get("embedder")
            entry["embedder_token"] = manifest.get("embedder_token")
            entry["key_modality"] = manifest.get("key_modality")
            entry["value_type"] = manifest.get("value_type")
            entry["n_entries"] = manifest.get("n_entries")
            entry["forced"] = manifest.get("forced")
            entry["leakage_verdict"] = kb_schema.leakage_verdict(manifest.get("leakage_audit", {}))

            if not values_path.exists() or not keys_path.exists():
                entry["status"] = (
                    f"SKIPPED (missing values.jsonl={not values_path.exists()} "
                    f"and/or keys.npy={not keys_path.exists()})"
                )
                entries.append(entry)
                continue

            try:
                from_item_ids = [
                    json.loads(line).get("provenance", {}).get("from_item_id")
                    for line in values_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                computed = kb_schema.content_hash_of(values_path, keys_path, from_item_ids, code_git_sha)
                entry["computed_content_hash_2026_07_12"] = computed
                entry["computed_content_hash_code_git_sha"] = code_git_sha
                entry["values_jsonl_bytes"] = values_path.stat().st_size
                entry["keys_npy_bytes"] = keys_path.stat().st_size
                entry["n_from_item_ids"] = len(from_item_ids)
                entry["status"] = "OK"
            except Exception as e:  # noqa: BLE001 - one bad source must not sink the whole inventory
                entry["status"] = f"ERROR computing content_hash: {type(e).__name__}: {e}"

            entries.append(entry)

    out = {
        "_comment": (
            "One-time, READ-ONLY content-hash inventory of every source under SPEECHRL_KB_DIR at "
            "2026-07-12 (RI mechanical remediation item 8). Nothing under SPEECHRL_KB_DIR was "
            "modified to produce this file -- see this script's module docstring for the "
            "retrospective-hash caveat (code_git_sha is the CURRENT HEAD at inventory time, not "
            "each source's original build-time HEAD, which was never recorded pre-2026-07-12)."
        ),
        "generated_at_utc": generated_at,
        "kb_root": str(root),
        "code_git_sha_used": code_git_sha,
        "n_sources": len(entries),
        "n_ok": sum(1 for e in entries if e.get("status") == "OK"),
        "sources": entries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(OUT_PATH.parent), prefix=OUT_PATH.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(out, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, str(OUT_PATH))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    print(f"wrote {OUT_PATH} ({out['n_ok']}/{out['n_sources']} sources OK)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

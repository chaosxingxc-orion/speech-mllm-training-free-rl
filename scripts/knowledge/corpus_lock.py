"""corpus_lock — F-6 remediation (2026-07-13 v4.2 doctoral review, wiki
2026-07-13-v42-doctoral-adversarial-integrity-review.md §3 F-6 + §6 P1_CORPUS_INDEPENDENCE):
pin upstream FiQA/squtr corpus IDENTITY so a downstream audit ("is this corpus the official,
complete, unaltered one?") is answered from VERIFIED EVIDENCE computed off the real
``corpus.jsonl`` bytes -- never from a caller-supplied ``corpus_mode`` string or boolean flag
(the review's exact finding: "``query_independent_corpus = PASS iff corpus_mode == 'full'``" --
a mode string can never prove the underlying corpus is actually complete/official/unaltered).

Four independent checks, ALL must match for a candidate corpus to verify against a lock:

  1. ``corpus_jsonl_sha256`` -- sha256 of the RAW ``corpus.jsonl`` bytes as they physically sit
     inside the upstream ``source_data.zip`` member (before any JSON parsing) -- the tightest,
     byte-exact check ("archive sha256 of the local corpus.jsonl" per the review's item 1 ask).
  2. ``ordered_doc_id_hash`` -- sha256 over each document's ``_id`` field, in the EXACT order the
     candidate corpus list is given (order-SENSITIVE, not just set-equality -- catches a
     resort/reshuffle even when every id is still present).
  3. ``normalized_content_hash`` -- sha256 over each document's NFKC-normalized ``title``+``text``,
     same order -- catches a CONTENT mutation (e.g. a doc's text silently edited) independent of
     whitespace/encoding noise a byte-exact check would also flag as a false mismatch.
  4. ``doc_count`` -- must equal BOTH the lock file's own recorded ``doc_count`` AND the
     hardcoded ``EXPECTED_DOC_COUNT`` anchor below (the review's literal "doc_count=57638" ask,
     field-verified in ``scripts/loaders/squtr.py``'s own module docstring 2026-07-09) -- a
     redundant, code-level anchor independent of whatever the lock FILE says, so a corrupted or
     hand-edited lock file cannot silently relax this specific invariant.

``verify_corpus_lock`` FAIL-CLOSES: any of the four checks not matching (or the subset/lock file
missing outright) is treated as a verification failure -- raises ``CorpusLockError`` by default
(``raise_on_mismatch=True``), or returns a ``{"verified": False, "mismatches": [...]}`` dict when
the caller passes ``raise_on_mismatch=False`` (used by ``kb_batch_build``'s audit-axis reporting
path, which needs a non-raising signal to fold into ``five_axis_audit`` rather than aborting the
whole call).

Usage:
    # one-time (or re-run whenever the upstream corpus is refreshed), against the REAL corpus:
    python scripts/knowledge/corpus_lock.py --generate --subset fiqa \\
        --out ../../../docs/corpus.lock.json          # umbrella docs/, NOT this repo's docs/

    # verification (real corpus, real lock file):
    python scripts/knowledge/corpus_lock.py --verify --subset fiqa \\
        --lock ../../../docs/corpus.lock.json

Lazy-import discipline (CLAUDE.md): ``squtr``/``zipfile`` stay inside functions so
``import corpus_lock`` stays light and this module's dataclasses/constants are always importable
for a synthetic-corpus test with zero data-root dependency.

--------------------------------------------------------------------------------------------------
TODO (integration deferred -- ``build_full_corpus.py`` is off-limits to this change: another agent
is concurrently editing it; the coordinator should wire the one-liner below once that lands):

``kb_batch_build.build_squtr_corpus_source(corpus_mode='full')`` ALREADY calls this module's
``EXPECTED_DOC_COUNT`` hard assert + ``verify_corpus_lock`` (see its own ``_corpus_lock_verification``
helper) at PERSIST time — every doc-count/lock check in this file's docstring is enforced there
today, with zero required changes to ``build_full_corpus.py``. The one OPTIONAL improvement left on
the table (not required for correctness, purely a fail-fast/cost optimization) is at the START of
``build_full_corpus.embed_full_corpus_checkpointed`` (or its ``main()``), BEFORE the multi-hour CPU
embedding loop runs at all:

    import corpus_lock
    corpus = squtr.load_full_corpus(subset)          # already the first line of that function
    corpus_lock.verify_corpus_lock(                  # ADD roughly here
        str(corpus_lock.default_lock_path()), subset, corpus_docs=corpus, raise_on_mismatch=True,
    )

This would fail an embedding run against an altered/wrong-count local corpus snapshot in ~seconds
(cheap: just re-reads ``corpus.jsonl``'s already-parsed doc list + the archive member bytes, no
model calls) instead of only at the final persist step after however many batches already ran. NOT
wired in this task (file-conflict constraint) -- ``build_squtr_corpus_source``'s own persist-time
gate is a complete, correct backstop either way; this is strictly an earlier-failure convenience.
--------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "loaders"))  # scripts/loaders -- bare `import squtr`

# ---------------------------------------------------------------------------------------------
# Frozen upstream identity (2026-07-13). SQuTR (SIGIR-2026, arXiv 2602.12783) ships from HF
# ``SLLMCommunity/SQuTR`` as a single ``source_data.zip`` (~21GB) -- ``hf_revision_sha`` below is
# that repo's own ``sha`` field from the local ``.hfd/repo_metadata.json`` snapshot metadata (the
# HF commit hash of the EXACT snapshot this repo fetched, 2026-07-07) -- this is the upstream
# REVISION pin (mirrors every other entry's "revision" field in docs/datasets.lock.json, which
# squtr was never added to -- see this review's finding: "docs/datasets.lock.json 没有
# FiQA/squtr corpus lock"). This dict is NOT itself recomputed from the zip (a 21GB archive whose
# own whole-file hash is not what "corpus identity" means here, and would take unreasonably long
# to hash for a value that doesn't change run to run) -- the doc-content hashes below (computed
# fresh by ``compute_corpus_lock``) are what's actually re-derived from the REAL bytes each time.
# ---------------------------------------------------------------------------------------------
UPSTREAM_SOURCE = {
    "fiqa": {
        "hf_repo": "SLLMCommunity/SQuTR",
        "hf_revision_sha": "2f1b041e2e98e0d28ed68fbcf22126ef247eb719",
        "arxiv": "2602.12783",
        "license": "cc-by-sa-4.0",
        "zip_member": "source_data/en/fiqa/corpus.jsonl",
        "note": ("SQuTR bilingual robustness benchmark (SIGIR 2026); fiqa subset (en) -- Stage-1 "
                 "default per wiki/survey/2026-07-09-datasets-candidates11.json. hf_revision_sha "
                 "is the HF dataset repo commit this local snapshot was fetched at "
                 "(speechrl-data/datasets/squtr/.hfd/repo_metadata.json 'sha' field, 2026-07-07)."),
    },
}

# Field-verified doc counts (scripts/loaders/squtr.py's own module docstring, 2026-07-09 field
# verification against the real zip) -- the hardcoded anchor `verify_corpus_lock` ALWAYS checks
# in addition to whatever a (possibly stale/tampered) lock file itself records.
EXPECTED_DOC_COUNT = {"fiqa": 57638}


class CorpusLockError(RuntimeError):
    """Raised by ``verify_corpus_lock`` (default ``raise_on_mismatch=True``) on ANY mismatch
    between a candidate corpus and its persisted lock -- fail-closed, per F-6 remediation: a
    corpus that does not verify is NEVER treated as query-independent/official evidence."""


@dataclass(frozen=True)
class CorpusLock:
    subset: str
    hf_repo: str
    hf_revision_sha: str
    arxiv: str
    license: str
    zip_member: str
    corpus_jsonl_sha256: str        # sha256 of the raw corpus.jsonl member bytes (archive-exact)
    corpus_jsonl_size_bytes: int
    doc_count: int
    ordered_doc_id_hash: str        # sha256 over doc _ids, order-sensitive
    normalized_content_hash: str    # sha256 over NFKC-normalized title+text, order-sensitive
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_content(s: str) -> str:
    """NFKC-normalize + collapse whitespace -- robust to encoding/whitespace noise while still
    catching a genuine content change (deliberately NOT the same as ``kb_audit.norm``, which also
    strips punctuation for answer-matching purposes; here we want content FIDELITY, not
    answer-matching, so punctuation is kept)."""
    s = unicodedata.normalize("NFKC", str(s))
    return " ".join(s.split())


def _hash_doc_ids(docs: list[dict]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(str(d["_id"]).encode("utf-8"))
        h.update(b"\x1f")  # unit separator -- prevents id1+id2 vs id12+id (concat) collisions
    return h.hexdigest()


def _hash_normalized_content(docs: list[dict]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(_normalize_content(d.get("title", "")).encode("utf-8"))
        h.update(b"\x1e")
        h.update(_normalize_content(d.get("text", "")).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _read_corpus_jsonl_member_bytes(subset: str) -> bytes:
    """Raw bytes of ``subset``'s ``corpus.jsonl`` zip member, straight off the E: data root --
    NEVER decompressed-then-reserialized (that would hash the PARSED representation, not the
    actual upstream archive bytes)."""
    import zipfile

    import squtr  # scripts/loaders/squtr.py

    member = UPSTREAM_SOURCE[subset]["zip_member"]
    with zipfile.ZipFile(squtr._zip_path()) as zf:
        return zf.read(member)


def compute_corpus_lock(subset: str = "fiqa", *, corpus_docs: list[dict] | None = None,
                          archive_member_bytes: bytes | None = None) -> CorpusLock:
    """Compute a fresh ``CorpusLock`` for ``subset``.

    Default (both ``corpus_docs``/``archive_member_bytes`` ``None``): reads the REAL corpus off
    the E: data root (``squtr.load_full_corpus(subset)`` for the doc list,
    ``_read_corpus_jsonl_member_bytes`` for the raw archive bytes) -- this is the "generate the
    lock from the real corpus file" path (F-6 remediation's core ask).

    Both params also accept caller-supplied values (a synthetic in-memory corpus / synthetic
    bytes) so tests can exercise this function, and ``verify_corpus_lock`` below, with ZERO data
    -root dependency -- see ``test_corpus_lock.py``.
    """
    if subset not in UPSTREAM_SOURCE:
        raise ValueError(f"corpus_lock.compute_corpus_lock: unknown subset {subset!r} -- no "
                          f"UPSTREAM_SOURCE entry (known: {sorted(UPSTREAM_SOURCE)})")
    meta = UPSTREAM_SOURCE[subset]

    if corpus_docs is None:
        import squtr

        corpus_docs = squtr.load_full_corpus(subset)
    if archive_member_bytes is None:
        archive_member_bytes = _read_corpus_jsonl_member_bytes(subset)

    return CorpusLock(
        subset=subset,
        hf_repo=meta["hf_repo"],
        hf_revision_sha=meta["hf_revision_sha"],
        arxiv=meta["arxiv"],
        license=meta["license"],
        zip_member=meta["zip_member"],
        corpus_jsonl_sha256=hashlib.sha256(archive_member_bytes).hexdigest(),
        corpus_jsonl_size_bytes=len(archive_member_bytes),
        doc_count=len(corpus_docs),
        ordered_doc_id_hash=_hash_doc_ids(corpus_docs),
        normalized_content_hash=_hash_normalized_content(corpus_docs),
        note=meta["note"],
    )


def generate_corpus_lock_file(path, subsets: tuple[str, ...] = ("fiqa",)) -> dict:
    """Compute + persist a lock covering ``subsets`` (default: just ``fiqa``, the Stage-1
    default) to ``path`` (typically the UMBRELLA repo's ``docs/corpus.lock.json`` -- this is a
    new, separate file from this W1 repo's own docs/; see module docstring's CLI usage). Every
    value is computed from the REAL corpus on the E: data root (no ``corpus_docs=``/
    ``archive_member_bytes=`` override here -- this is the real-generation entry point, unlike
    ``compute_corpus_lock`` which a test can override).
    """
    from datetime import datetime, timezone

    locks = {s: compute_corpus_lock(s).to_dict() for s in subsets}
    for s in subsets:
        expected = EXPECTED_DOC_COUNT.get(s)
        if expected is not None and locks[s]["doc_count"] != expected:
            raise CorpusLockError(
                f"corpus_lock.generate_corpus_lock_file: freshly computed doc_count="
                f"{locks[s]['doc_count']} for subset={s!r} != EXPECTED_DOC_COUNT[{s!r}]="
                f"{expected} -- refusing to WRITE a lock file that already contradicts the "
                "hardcoded field-verified anchor (something is wrong with the local corpus "
                "snapshot; investigate before locking it)."
            )
    out = {
        "_comment": ("F-6 remediation (2026-07-13 v4.2 doctoral review) -- pinned upstream "
                     "corpus identity for squtr subsets, computed from the REAL corpus.jsonl on "
                     "the E: data root. Regenerate with `python "
                     "projects/speech-mllm-training-free-rl/scripts/knowledge/corpus_lock.py "
                     "--generate`. See scripts/knowledge/corpus_lock.py (W1 repo) for the "
                     "verification contract (verify_corpus_lock, fail-closed)."),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/knowledge/corpus_lock.py (W1 repo)",
        "expected_doc_count": dict(EXPECTED_DOC_COUNT),
        "subsets": locks,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, sort_keys=False)
    return out


def default_lock_path() -> Path:
    """The umbrella repo's ``docs/corpus.lock.json`` -- ``HERE`` is this W1 repo's own
    ``scripts/knowledge``; ``parents[3]`` is the umbrella root (``exploring-l4-intelligence``),
    mirroring ``docs/checks/v42_conformance.py``'s existing umbrella-``docs/`` placement pattern.
    ONE canonical place this path is computed (both the CLI below and
    ``kb_batch_build.build_squtr_corpus_source`` call this, rather than each hardcoding it)."""
    return HERE.parents[3] / "docs" / "corpus.lock.json"


def load_corpus_lock_file(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_corpus_lock(path, subset: str = "fiqa", *, corpus_docs: list[dict] | None = None,
                         archive_member_bytes: bytes | None = None,
                         raise_on_mismatch: bool = True) -> dict:
    """Fail-closed verification: recompute ``corpus_jsonl_sha256``/``doc_count``/
    ``ordered_doc_id_hash``/``normalized_content_hash`` for the CANDIDATE corpus and compare
    every field against the persisted lock at ``path``.

    ``corpus_docs``/``archive_member_bytes`` default to ``None`` -- reads the REAL corpus off the
    E: data root exactly like ``compute_corpus_lock``'s own defaults. Pass a synthetic in-memory
    ``corpus_docs`` (and, optionally, synthetic ``archive_member_bytes``) to verify an ARBITRARY
    candidate corpus (e.g. a build-time in-progress ``corpus_sample`` a caller already has in
    hand, or a synthetic corpus in a test) without re-reading the real zip.

    Missing lock file / missing subset entry / ANY of the four checks not matching -> a
    verification FAILURE. Default ``raise_on_mismatch=True`` raises ``CorpusLockError`` (the
    literal "fail-closes on any mismatch" contract F-6 asks for). ``raise_on_mismatch=False``
    instead returns ``{"verified": False, "subset":..., "mismatches": [...]}`` -- used by
    ``kb_batch_build``'s non-fatal five-axis audit reporting path.
    """
    mismatches: list[str] = []

    if not os.path.exists(path):
        mismatches.append(f"lock file not found at {path!r}")
        lock = None
    else:
        lock_all = load_corpus_lock_file(path)
        lock = (lock_all.get("subsets") or {}).get(subset)
        if lock is None:
            mismatches.append(f"lock file {path!r} has no subset={subset!r} entry")

    real_mode = corpus_docs is None
    if real_mode:
        import squtr

        corpus_docs = squtr.load_full_corpus(subset)
        if archive_member_bytes is None:
            archive_member_bytes = _read_corpus_jsonl_member_bytes(subset)
    doc_count = len(corpus_docs)
    ordered_doc_id_hash = _hash_doc_ids(corpus_docs)
    normalized_content_hash = _hash_normalized_content(corpus_docs)

    expected_count = EXPECTED_DOC_COUNT.get(subset)
    if expected_count is not None and doc_count != expected_count:
        mismatches.append(f"doc_count={doc_count} != EXPECTED_DOC_COUNT[{subset!r}]={expected_count}")

    if lock is not None:
        if doc_count != lock["doc_count"]:
            mismatches.append(f"doc_count={doc_count} != lock.doc_count={lock['doc_count']}")
        if ordered_doc_id_hash != lock["ordered_doc_id_hash"]:
            mismatches.append("ordered_doc_id_hash mismatch (doc set and/or order changed)")
        if normalized_content_hash != lock["normalized_content_hash"]:
            mismatches.append("normalized_content_hash mismatch (document content changed)")

        if archive_member_bytes is not None:
            actual_sha = hashlib.sha256(archive_member_bytes).hexdigest()
            if actual_sha != lock["corpus_jsonl_sha256"]:
                mismatches.append("corpus_jsonl_sha256 mismatch (archive bytes changed)")
        # else: no raw archive bytes supplied (build-time in-memory corpus_sample or a synthetic
        # test corpus) -- the archive-byte check is SKIPPED, not silently treated as a pass;
        # `archive_byte_check_ran` in the returned dict below records this explicitly.

    result = {
        "subset": subset,
        "verified": not mismatches,
        "doc_count": doc_count,
        "ordered_doc_id_hash": ordered_doc_id_hash,
        "normalized_content_hash": normalized_content_hash,
        "archive_byte_check_ran": archive_member_bytes is not None,
        "mismatches": mismatches,
    }
    if mismatches and raise_on_mismatch:
        raise CorpusLockError(
            f"corpus_lock.verify_corpus_lock(subset={subset!r}, path={path!r}): "
            f"{len(mismatches)} mismatch(es): {'; '.join(mismatches)}"
        )
    return result


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--subset", default="fiqa")
    ap.add_argument("--out", default=None, help="--generate: lock file path to write")
    ap.add_argument("--lock", default=None, help="--verify: lock file path to check against")
    args = ap.parse_args()

    if args.generate:
        out_path = args.out or str(default_lock_path())
        result = generate_corpus_lock_file(out_path, subsets=(args.subset,))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.verify:
        lock_path = args.lock or str(default_lock_path())
        result = verify_corpus_lock(lock_path, subset=args.subset, raise_on_mismatch=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["verified"] else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

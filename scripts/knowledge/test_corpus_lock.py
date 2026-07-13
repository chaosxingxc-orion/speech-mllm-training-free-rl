"""test_corpus_lock — proves ``corpus_lock``'s fail-closed verification contract (F-6 remediation,
2026-07-13 v4.2 doctoral review §3 F-6 + §6 P1_CORPUS_INDEPENDENCE).

Entirely OFFLINE / synthetic — every case below passes ``corpus_docs=``/``archive_member_bytes=``
explicitly to ``compute_corpus_lock``/``verify_corpus_lock``, so NONE of this touches the real
21GB squtr zip on the E: data root (that real-corpus round trip is exercised once, for real, by
``corpus_lock.py --generate`` itself — a one-off generation run, not a pytest case).

Cases:
  1. round trip: a lock generated from a synthetic corpus verifies CLEAN against that exact corpus.
  2. content mutation (one doc's text changed) -> ``normalized_content_hash`` mismatch -> FAIL.
  3. reorder (same docs, different order) -> ``ordered_doc_id_hash`` mismatch -> FAIL.
  4. archive-byte mutation -> ``corpus_jsonl_sha256`` mismatch -> FAIL.
  5. missing lock file / missing subset entry -> FAIL, not a crash.
  6. ``raise_on_mismatch=True`` (the default) actually raises ``CorpusLockError``.
  7. THE literal F-6 ask: a subset='fiqa' candidate corpus with 57,637 docs (one under the
     field-verified ``EXPECTED_DOC_COUNT['fiqa']=57638`` anchor) FAILS -- even when the lock FILE
     itself is internally self-consistent with the wrong (57,637) count, i.e. even a tampered
     lock file cannot relax the hardcoded anchor.
  8. same for 57,639 (one over).
  9. ``generate_corpus_lock_file`` itself refuses to WRITE a lock whose freshly-computed doc_count
     contradicts ``EXPECTED_DOC_COUNT`` (monkeypatched ``compute_corpus_lock``, still offline).

Run:
    PYTHONPATH=src pytest scripts/knowledge/test_corpus_lock.py -q
or:
    python -u scripts/knowledge/test_corpus_lock.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)  # scripts/knowledge -- bare `import corpus_lock`

import corpus_lock as cl  # noqa: E402


def check(label: str, cond: bool, results: dict) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}", flush=True)
    results[label] = bool(cond)


def _synthetic_docs(n: int = 5, *, mutate_text_idx: int | None = None, shuffle: bool = False) -> list[dict]:
    docs = [{"_id": f"d{i}", "title": f"Title {i}", "text": f"Body text number {i}."} for i in range(n)]
    if mutate_text_idx is not None:
        docs[mutate_text_idx] = dict(docs[mutate_text_idx], text=docs[mutate_text_idx]["text"] + " MUTATED")
    if shuffle:
        docs = list(reversed(docs))
    return docs


def _write_lock_file(path: str, subset: str, lock: "cl.CorpusLock") -> None:
    out = {
        "_comment": "test fixture -- synthetic, not a real corpus lock",
        "subsets": {subset: lock.to_dict()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)


def _check_corpus_lock(results: dict) -> None:
    # register a throwaway synthetic subset in UPSTREAM_SOURCE so compute_corpus_lock's
    # subset-registry check passes without touching the real 'fiqa' entry (which is asserted
    # against EXPECTED_DOC_COUNT=57638 elsewhere, case 7/8 below) -- restored in a finally.
    test_subset = "unit-test-subset"
    orig_upstream = dict(cl.UPSTREAM_SOURCE)
    cl.UPSTREAM_SOURCE[test_subset] = {
        "hf_repo": "unit-test/repo", "hf_revision_sha": "deadbeef", "arxiv": "0000.00000",
        "license": "test", "zip_member": "unit-test/corpus.jsonl", "note": "synthetic test subset",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="corpus_lock_test_") as tmpdir:
            docs = _synthetic_docs(5)
            archive_bytes = b"synthetic-archive-bytes-v1"
            lock = cl.compute_corpus_lock(test_subset, corpus_docs=docs, archive_member_bytes=archive_bytes)
            check("compute_corpus_lock: doc_count matches input", lock.doc_count == 5, results)

            lock_path = os.path.join(tmpdir, "lock.json")
            _write_lock_file(lock_path, test_subset, lock)

            # --- case 1: exact round trip verifies CLEAN ---
            result = cl.verify_corpus_lock(lock_path, test_subset, corpus_docs=docs,
                                            archive_member_bytes=archive_bytes, raise_on_mismatch=False)
            check("case1 round-trip: verified True, zero mismatches",
                  result["verified"] is True and result["mismatches"] == [], results)

            # --- case 2: content mutation -> normalized_content_hash mismatch ---
            mutated = _synthetic_docs(5, mutate_text_idx=2)
            result = cl.verify_corpus_lock(lock_path, test_subset, corpus_docs=mutated,
                                            archive_member_bytes=archive_bytes, raise_on_mismatch=False)
            check("case2 content mutation: verified False", result["verified"] is False, results)
            check("case2 content mutation: flags normalized_content_hash",
                  any("normalized_content_hash" in m for m in result["mismatches"]), results)

            # --- case 3: reorder -> ordered_doc_id_hash mismatch (doc set unchanged) ---
            reordered = _synthetic_docs(5, shuffle=True)
            result = cl.verify_corpus_lock(lock_path, test_subset, corpus_docs=reordered,
                                            archive_member_bytes=archive_bytes, raise_on_mismatch=False)
            check("case3 reorder: verified False", result["verified"] is False, results)
            check("case3 reorder: flags ordered_doc_id_hash",
                  any("ordered_doc_id_hash" in m for m in result["mismatches"]), results)
            check("case3 reorder: doc_count itself still matches (proves this is an ORDER check)",
                  not any("doc_count" in m for m in result["mismatches"]), results)

            # --- case 4: archive-byte mutation -> corpus_jsonl_sha256 mismatch ---
            result = cl.verify_corpus_lock(lock_path, test_subset, corpus_docs=docs,
                                            archive_member_bytes=b"tampered-bytes", raise_on_mismatch=False)
            check("case4 archive-byte mutation: verified False", result["verified"] is False, results)
            check("case4 archive-byte mutation: flags corpus_jsonl_sha256",
                  any("corpus_jsonl_sha256" in m for m in result["mismatches"]), results)

            # --- case 5a: missing lock file ---
            result = cl.verify_corpus_lock(os.path.join(tmpdir, "does_not_exist.json"), test_subset,
                                            corpus_docs=docs, archive_member_bytes=archive_bytes,
                                            raise_on_mismatch=False)
            check("case5a missing lock file: verified False, not a crash", result["verified"] is False, results)

            # --- case 5b: lock file exists but has no entry for this subset ---
            other_path = os.path.join(tmpdir, "other_subset_lock.json")
            _write_lock_file(other_path, "some-other-subset", lock)
            result = cl.verify_corpus_lock(other_path, test_subset, corpus_docs=docs,
                                            archive_member_bytes=archive_bytes, raise_on_mismatch=False)
            check("case5b missing subset entry: verified False", result["verified"] is False, results)

            # --- case 6: raise_on_mismatch=True (the default) actually raises ---
            raised = False
            try:
                cl.verify_corpus_lock(lock_path, test_subset, corpus_docs=mutated,
                                       archive_member_bytes=archive_bytes)
            except cl.CorpusLockError:
                raised = True
            check("case6 raise_on_mismatch=True raises CorpusLockError on mismatch", raised, results)
    finally:
        cl.UPSTREAM_SOURCE.clear()
        cl.UPSTREAM_SOURCE.update(orig_upstream)

    # --- case 7/8: THE literal F-6 ask -- subset='fiqa', 57637/57639 must fail, even against a
    # SELF-CONSISTENT (attacker-tampered) lock file that already agrees with the wrong count.
    # 100% offline: corpus_docs + archive_member_bytes are both supplied explicitly, so
    # compute_corpus_lock/verify_corpus_lock never touch the real zip for subset='fiqa' either.
    for bad_n, label in ((57637, "case7 fiqa doc_count=57637 (one under)"),
                          (57639, "case8 fiqa doc_count=57639 (one over)")):
        with tempfile.TemporaryDirectory(prefix="corpus_lock_fiqa_test_") as tmpdir:
            bad_docs = _synthetic_docs(bad_n)
            bad_bytes = f"fake-fiqa-bytes-n{bad_n}".encode()
            bad_lock = cl.compute_corpus_lock("fiqa", corpus_docs=bad_docs, archive_member_bytes=bad_bytes)
            check(f"{label}: compute_corpus_lock itself records the (wrong) count faithfully",
                  bad_lock.doc_count == bad_n, results)
            tampered_path = os.path.join(tmpdir, "tampered_lock.json")
            _write_lock_file(tampered_path, "fiqa", bad_lock)  # file is internally self-consistent

            result = cl.verify_corpus_lock(tampered_path, "fiqa", corpus_docs=bad_docs,
                                            archive_member_bytes=bad_bytes, raise_on_mismatch=False)
            check(f"{label}: verified False despite a self-consistent tampered lock file",
                  result["verified"] is False, results)
            check(f"{label}: mismatch cites EXPECTED_DOC_COUNT anchor",
                  any("EXPECTED_DOC_COUNT" in m for m in result["mismatches"]), results)

    # --- case 9: generate_corpus_lock_file refuses to WRITE a lock that already contradicts
    # EXPECTED_DOC_COUNT -- monkeypatch compute_corpus_lock to return a wrong-count lock for
    # 'fiqa' (still fully offline: never calls the real one this monkeypatch replaces).
    orig_compute = cl.compute_corpus_lock

    def _fake_compute(subset, **kwargs):
        docs = _synthetic_docs(57637)
        return orig_compute(subset, corpus_docs=docs, archive_member_bytes=b"fake")

    cl.compute_corpus_lock = _fake_compute
    try:
        with tempfile.TemporaryDirectory(prefix="corpus_lock_gen_test_") as tmpdir:
            out_path = os.path.join(tmpdir, "would_be_lock.json")
            raised = False
            try:
                cl.generate_corpus_lock_file(out_path, subsets=("fiqa",))
            except cl.CorpusLockError:
                raised = True
            check("case9 generate_corpus_lock_file refuses a wrong-count fresh compute", raised, results)
            check("case9 generate_corpus_lock_file wrote NOTHING on refusal",
                  not os.path.exists(out_path), results)
    finally:
        cl.compute_corpus_lock = orig_compute


def test_corpus_lock_all_cases() -> None:
    results: dict[str, bool] = {}
    _check_corpus_lock(results)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"failed checks: {failed}"


def main() -> int:
    results: dict[str, bool] = {}
    _check_corpus_lock(results)
    all_pass = all(results.values())
    print("\n=== CORPUS_LOCK TEST ===")
    print(json.dumps(results, indent=2))
    print("CORPUS_LOCK_TEST_PASS" if all_pass else "CORPUS_LOCK_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

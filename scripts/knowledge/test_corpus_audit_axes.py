"""test_corpus_audit_axes — proves F-6 (corpus-lock-derived ``query_independent_corpus``) and F-7
(no-scrub open-corpus + descriptive ``answer_presence_expected`` + hard-failing
``no_injected_gold_context``) end to end through ``kb_batch_build.build_squtr_corpus_source`` and
directly through ``kb_batch_build.corpus_five_axis_audit`` (2026-07-13, ticket #38 items 1/2,
v4.2 doctoral review §3 F-6/F-7 + §6 P1_CORPUS_INDEPENDENCE).

Fully OFFLINE: ``squtr.load_full_corpus`` and ``corpus_lock.default_lock_path`` are monkeypatched
for every case below — nothing here touches the real 21GB squtr zip or the real
``docs/corpus.lock.json``; a tmp ``SPEECHRL_KB_DIR`` is used throughout (never the real KB store).
``embedder='omni-embed-nemotron'`` + ``precomputed_keys`` covering every doc means no real model is
ever loaded either.

Cases:
  1. golden / positive: a lock generated from corpus X, build against the SAME corpus X with
     non-empty overlapping ``eval_golds`` -> built successfully, ``query_independent_corpus``==PASS,
     ``answer_presence_expected``==PRESENT, ``leakage_gate_enforced``==False on the manifest, and —
     the literal ask — the PERSISTED VALUE for the gold-containing doc is UNCHANGED (byte-identical
     to the source text, i.e. NEVER scrubbed).
  2. ``eval_golds`` empty -> ``answer_presence_expected``==NOT_EVALUATED (unchanged rule).
  3. altered corpus (same doc COUNT, different CONTENT than what the lock was generated from) under
     corpus_mode='full' -> ``query_independent_corpus``==FAIL, ``overall`` reports the failing axis
     (this codebase's five-axis audit is a REPORTING mechanism, not a hard gate — see
     ``corpus_five_axis_audit``'s own "CORPUS_AUDIT_AXES" module comment; a downstream consumer
     gates on ``overall == 'CLEAN'``, mirroring ``leakage_verdict`` elsewhere in this repo).
  4. doc-count mismatch (one doc short of the frozen ``EXPECTED_DOC_COUNT`` anchor) -> HARD refusal:
     ``build_squtr_corpus_source`` raises ``ValueError`` BEFORE persisting anything (source dir never
     created) -- this IS the hard gate (distinct from case 3's soft axis report).
  5. ``corpus_five_axis_audit`` direct negative golden test: ``injected_gold_context_detected=True``
     -> ``no_injected_gold_context``==FAIL, ``overall`` reports it, REGARDLESS of every other axis
     being PASS.
  6. ``corpus_five_axis_audit`` direct positive golden test: ``injected_gold_context_detected=False``
     + every other axis PASS -> ``overall``=='CLEAN'.

Run:
    PYTHONPATH=src pytest scripts/knowledge/test_corpus_audit_axes.py -q
or:
    python -u scripts/knowledge/test_corpus_audit_axes.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(__file__)
LOADERS_DIR = os.path.join(os.path.dirname(HERE), "loaders")
for _p in (HERE, LOADERS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_lock as cl  # noqa: E402
import kb_batch_build as kbb  # noqa: E402


def check(label: str, cond: bool, results: dict) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}", flush=True)
    results[label] = bool(cond)


def _ensure_tmp_kb_dir() -> str:
    kb_dir = os.environ.get("SPEECHRL_KB_DIR")
    if not kb_dir:
        kb_dir = tempfile.mkdtemp(prefix="corpus_audit_axes_test_")
    resolved = os.path.abspath(kb_dir).replace("\\", "/").lower()
    assert "speechrl-knowledge" not in resolved, (
        f"refusing to run: SPEECHRL_KB_DIR looks like the real persistent KB ({kb_dir}). "
        "Point this at a tmp dir."
    )
    os.environ["SPEECHRL_KB_DIR"] = kb_dir
    return kb_dir


def _synthetic_corpus(n: int = 6, gold_text: str = "the answer is forty two") -> list[dict]:
    docs = [{"_id": f"doc{i}", "title": f"Title {i}", "text": f"Body paragraph {i}."} for i in range(n)]
    # doc0 naturally, legitimately contains the "gold" text -- an open evidence corpus doc that
    # happens to answer a real question, exactly the F-7 scenario (never a leak).
    docs[0] = dict(docs[0], text=f"Body paragraph 0. {gold_text}.")
    return docs


class _FakeSqutrModule(types.SimpleNamespace):
    pass


def _install_fake_squtr(corpus_docs: list[dict]):
    """Monkeypatch squtr.load_full_corpus (the real loaders/squtr module, already imported by
    kb_batch_build/corpus_lock lazily) -- returns corpus_docs verbatim, no zip/data-root touch."""
    import squtr

    orig = squtr.load_full_corpus
    squtr.load_full_corpus = lambda subset="fiqa": corpus_docs
    return orig


def _restore_squtr(orig) -> None:
    import squtr

    squtr.load_full_corpus = orig


def _write_synthetic_lock(path: str, subset: str, corpus_docs: list[dict]) -> None:
    lock = cl.compute_corpus_lock(subset, corpus_docs=corpus_docs, archive_member_bytes=b"synthetic")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_comment": "test fixture", "subsets": {subset: lock.to_dict()}}, fh)


def _check_corpus_audit_axes(results: dict) -> None:
    import kb_schema
    import squtr

    kb_dir = _ensure_tmp_kb_dir()
    print(f"[test_corpus_audit_axes] SPEECHRL_KB_DIR={kb_dir} (tmp)", flush=True)

    subset = "fiqa"
    corpus_docs = _synthetic_corpus(6)
    doc_ids = [d["_id"] for d in corpus_docs]
    precomputed_keys = {}
    import numpy as np

    for i, did in enumerate(doc_ids):
        v = np.random.RandomState(i).randn(8).astype("float32")
        precomputed_keys[did] = v / np.linalg.norm(v)

    orig_expected_count = dict(cl.EXPECTED_DOC_COUNT)
    orig_default_lock_path = cl.default_lock_path

    # --- shared fixture: EXPECTED_DOC_COUNT['fiqa'] monkeypatched to the tiny synthetic size, so
    # the hard doc-count assert in build_squtr_corpus_source is exercised against a fast synthetic
    # corpus rather than the real 57,638-doc one. ---
    cl.EXPECTED_DOC_COUNT["fiqa"] = len(corpus_docs)

    try:
        with tempfile.TemporaryDirectory(prefix="corpus_audit_axes_lock_") as tmpdir:
            lock_path = os.path.join(tmpdir, "corpus.lock.json")
            _write_synthetic_lock(lock_path, subset, corpus_docs)
            cl.default_lock_path = lambda: lock_path

            # === case 1: golden / positive — matching corpus, non-empty overlapping eval_golds ===
            orig_load = _install_fake_squtr(corpus_docs)
            try:
                result = kbb.build_squtr_corpus_source(
                    "omni-embed-nemotron", subset=subset, corpus_mode="full",
                    precomputed_keys=precomputed_keys,
                    eval_golds=["the answer is forty two"],
                    note="test fixture (case 1)",
                )
            finally:
                _restore_squtr(orig_load)

            check("case1: status == built", result.get("status") == "built", results)
            five_axis = result.get("five_axis_audit", {})
            check("case1: query_independent_corpus == PASS (lock-verified evidence)",
                  five_axis.get("query_independent_corpus") == "PASS", results)
            check("case1: corpus_lock_verification.verified is True",
                  result.get("corpus_lock_verification", {}).get("verified") is True, results)
            check("case1: answer_presence_expected == PRESENT",
                  five_axis.get("answer_presence_expected") == "PRESENT", results)
            check("case1: no_injected_gold_context == PASS (structurally not applicable)",
                  five_axis.get("no_injected_gold_context") == "PASS", results)
            manifest = result.get("manifest", {})
            check("case1: manifest.leakage_gate_enforced is False (descriptive-only, not a hard gate)",
                  manifest.get("leakage_gate_enforced") is False, results)
            check("case1: manifest.forced is False (nothing was 'forced' -- there was no gate to override)",
                  manifest.get("forced") is False, results)

            source_name = manifest["source"]
            values = kb_schema.load_values(source_name)
            doc0_value = next(v.value for v in values if v.provenance.get("from_item_id", "").endswith("|doc0"))
            check("case1: gold-containing doc's persisted VALUE is UNCHANGED (never scrubbed)",
                  doc0_value == corpus_docs[0]["title"] + " " + corpus_docs[0]["text"], results)
            check("case1: gold text is actually still present verbatim in the persisted value",
                  "the answer is forty two" in doc0_value, results)

            # === case 2: eval_golds empty -> NOT_EVALUATED ===
            orig_load = _install_fake_squtr(corpus_docs)
            try:
                result2 = kbb.build_squtr_corpus_source(
                    "omni-embed-nemotron", subset=subset, corpus_mode="full",
                    precomputed_keys=precomputed_keys, eval_golds=None,
                    note="test fixture (case 2)", supersede=True,
                )
            finally:
                _restore_squtr(orig_load)
            check("case2: answer_presence_expected == NOT_EVALUATED (0 golds checked)",
                  result2.get("five_axis_audit", {}).get("answer_presence_expected") == "NOT_EVALUATED",
                  results)

            # === case 3: altered corpus (same count, different content) -> query_independent_corpus FAIL ===
            altered_docs = _synthetic_corpus(6, gold_text="a totally different sentence")
            altered_docs = [dict(d, text=d["text"] + " ALTERED") for d in altered_docs]
            orig_load = _install_fake_squtr(altered_docs)
            try:
                result3 = kbb.build_squtr_corpus_source(
                    "omni-embed-nemotron", subset=subset, corpus_mode="full",
                    precomputed_keys=precomputed_keys, eval_golds=None,
                    note="test fixture (case 3, altered corpus)", supersede=True,
                )
            finally:
                _restore_squtr(orig_load)
            check("case3: altered corpus still builds (soft-fail reporting, not a crash)",
                  result3.get("status") == "built", results)
            check("case3: query_independent_corpus == FAIL (content diverges from the pinned lock)",
                  result3.get("five_axis_audit", {}).get("query_independent_corpus") == "FAIL", results)
            check("case3: corpus_lock_verification.verified is False",
                  result3.get("corpus_lock_verification", {}).get("verified") is False, results)
            check("case3: overall reports the failing axis",
                  "query_independent_corpus" in result3.get("five_axis_audit", {}).get("overall", ""),
                  results)

            # === case 4: doc-count mismatch -> HARD refusal, nothing persisted/overwritten ===
            manifest_path = (kb_schema.kb_root() / "knowledge_base" / source_name / "manifest.json")
            manifest_bytes_before = manifest_path.read_bytes()  # source_name from case1's build above

            short_docs = corpus_docs[:-1]  # one doc short of EXPECTED_DOC_COUNT['fiqa']
            short_keys = {k: v for k, v in precomputed_keys.items() if k in {d["_id"] for d in short_docs}}
            orig_load = _install_fake_squtr(short_docs)
            raised = False
            try:
                try:
                    kbb.build_squtr_corpus_source(
                        "omni-embed-nemotron", subset=subset, corpus_mode="full",
                        precomputed_keys=short_keys, eval_golds=None, supersede=True,
                        note="test fixture (case 4, short corpus) -- MUST NEVER PERSIST",
                    )
                except ValueError as e:
                    raised = "EXPECTED_DOC_COUNT" in str(e) or "incomplete" in str(e).lower()
            finally:
                _restore_squtr(orig_load)
            check("case4: build_squtr_corpus_source raises ValueError on doc-count mismatch", raised, results)
            check("case4: the existing source's manifest is byte-identical after the refused attempt "
                  "(refusal happens BEFORE kb_build.build_source is ever called -- nothing superseded)",
                  manifest_path.read_bytes() == manifest_bytes_before, results)
    finally:
        cl.EXPECTED_DOC_COUNT.clear()
        cl.EXPECTED_DOC_COUNT.update(orig_expected_count)
        cl.default_lock_path = orig_default_lock_path

    # === case 5/6: corpus_five_axis_audit direct golden tests (no build pipeline needed) ===
    base_manifest = {"content_hash": "x", "code_git_sha": "x", "embedder_token": "x",
                      "pool_split": "test", "build_hash": "x"}
    clv_pass = {"verified": True}

    axes_negative = kbb.corpus_five_axis_audit(
        object_correct=True, corpus_mode="full", label_independent_build=True,
        n_eval_golds_checked=1, answer_overlap_rate=0.0,
        injected_gold_context_detected=True,  # <-- the negative golden case
        corpus_lock_verification=clv_pass, manifest=base_manifest,
    )
    check("case5 negative golden: no_injected_gold_context == FAIL",
          axes_negative["no_injected_gold_context"] == "FAIL", results)
    check("case5 negative golden: overall reports FAIL even though every other axis is clean",
          axes_negative["overall"].startswith("FAIL") and "no_injected_gold_context" in axes_negative["overall"],
          results)

    axes_positive = kbb.corpus_five_axis_audit(
        object_correct=True, corpus_mode="full", label_independent_build=True,
        n_eval_golds_checked=1, answer_overlap_rate=0.5,
        injected_gold_context_detected=False,  # <-- the positive golden case
        corpus_lock_verification=clv_pass, manifest=base_manifest,
    )
    check("case6 positive golden: no_injected_gold_context == PASS", axes_positive["no_injected_gold_context"] == "PASS", results)
    check("case6 positive golden: answer_presence_expected == PRESENT (descriptive, not a fail)",
          axes_positive["answer_presence_expected"] == "PRESENT", results)
    check("case6 positive golden: overall == CLEAN", axes_positive["overall"] == "CLEAN", results)


def test_corpus_audit_axes_all_cases() -> None:
    results: dict[str, bool] = {}
    _check_corpus_audit_axes(results)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"failed checks: {failed}"


def main() -> int:
    results: dict[str, bool] = {}
    _check_corpus_audit_axes(results)
    all_pass = all(results.values())
    print("\n=== CORPUS_AUDIT_AXES TEST ===")
    print(json.dumps(results, indent=2))
    print("CORPUS_AUDIT_AXES_TEST_PASS" if all_pass else "CORPUS_AUDIT_AXES_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

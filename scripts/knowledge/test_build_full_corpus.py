"""scripts/knowledge/test_build_full_corpus.py — ticket #38 item 1 (F'-3 remediation): proves
``build_full_corpus.embed_full_corpus_checkpointed``'s resume/checkpoint mechanics for real, rather
than leaving the launcher unevaluated (the BLOCKER finding this file answers). Pure-python, fully
OFFLINE — ``kb_embed.embed_text`` is monkeypatched (no real glap/nemotron model load) and the
corpus is injected via ``corpus_override`` (no real squtr zip touched); NEVER calls
``kb_batch_build.build_squtr_corpus_source`` for real (the actual store-write persist is a
separate, deliberately-scheduled operator run — see ``build_full_corpus.sh``'s own docstring) —
that call is itself monkeypatched in the one test that exercises ``main()``'s wiring.

Run:
    python -u scripts/knowledge/test_build_full_corpus.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))              # scripts/knowledge
SCRIPTS = os.path.dirname(HERE)                                  # scripts
LOADERS_DIR = os.path.join(SCRIPTS, "loaders")
for _p in (HERE, LOADERS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _fake_corpus(n: int) -> list[dict]:
    return [{"_id": f"doc{i:03d}", "title": f"title {i}", "text": f"body text {i}"} for i in range(n)]


def _make_spy_embed_text(dim: int = 8):
    """A deterministic, dependency-free stand-in for ``kb_embed.embed_text`` that also records
    every batch it was called with (for asserting exactly which docs got (re-)embedded)."""
    import numpy as np

    calls: list[list[str]] = []

    def _spy(texts, embedder: str = "auto", source_lang: str = "eng_Latn", **kw):
        calls.append(list(texts))
        vecs = np.stack([
            np.random.RandomState(abs(hash(t)) % (2**31)).randn(dim).astype("float32")
            for t in texts
        ])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return f"fake:{embedder}", vecs, None

    return _spy, calls


def check(label: str, cond: bool, results: dict) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}", flush=True)
    results[label] = bool(cond)


# ---------------------------------------------------------------------------------------------
# test a: a fresh run embeds every doc, in batches, and checkpoints after each batch.
# ---------------------------------------------------------------------------------------------

def _check_fresh_run_embeds_all_docs_in_batches(results: dict) -> None:
    import build_full_corpus as bfc
    import kb_embed

    orig_embed_text = kb_embed.embed_text
    spy, calls = _make_spy_embed_text()
    kb_embed.embed_text = spy
    try:
        with tempfile.TemporaryDirectory(prefix="bfc_test_") as ckpt_dir:
            corpus = _fake_corpus(12)
            keys = bfc.embed_full_corpus_checkpointed(
                "glap", subset="fake", batch_size=5, checkpoint_dir=ckpt_dir,
                corpus_override=corpus, progress=False,
            )
            check("fresh run returns a vector for every corpus doc", len(keys) == 12, results)
            check("fresh run batches in ceil(12/5)=3 calls to embed_text",
                  len(calls) == 3, results)
            check("fresh run's batch sizes are [5, 5, 2]",
                  [len(c) for c in calls] == [5, 5, 2], results)

            path = bfc.checkpoint_path("glap", "fake", ckpt_dir)
            check("checkpoint file exists on disk after a fresh run", os.path.exists(path), results)

            reloaded = bfc.load_checkpoint(path)
            check("reloaded checkpoint has all 12 doc ids",
                  set(reloaded.keys()) == {d["_id"] for d in corpus}, results)
            import numpy as np

            check("reloaded checkpoint vectors match the returned vectors exactly",
                  all(np.allclose(reloaded[k], keys[k]) for k in keys), results)
    finally:
        kb_embed.embed_text = orig_embed_text


# ---------------------------------------------------------------------------------------------
# test b: re-running against the SAME checkpoint_dir with everything already done makes ZERO new
# embed_text calls -- the core resumability guarantee.
# ---------------------------------------------------------------------------------------------

def _check_rerun_after_full_checkpoint_makes_no_new_calls(results: dict) -> None:
    import build_full_corpus as bfc
    import kb_embed

    orig_embed_text = kb_embed.embed_text
    spy, calls = _make_spy_embed_text()
    kb_embed.embed_text = spy
    try:
        with tempfile.TemporaryDirectory(prefix="bfc_test_") as ckpt_dir:
            corpus = _fake_corpus(9)
            bfc.embed_full_corpus_checkpointed(
                "glap", subset="fake", batch_size=4, checkpoint_dir=ckpt_dir,
                corpus_override=corpus, progress=False,
            )
            n_calls_first_run = len(calls)
            calls.clear()

            keys2 = bfc.embed_full_corpus_checkpointed(
                "glap", subset="fake", batch_size=4, checkpoint_dir=ckpt_dir,
                corpus_override=corpus, progress=False,
            )
            check("first run made >=1 embed_text call", n_calls_first_run >= 1, results)
            check("re-running against a FULLY checkpointed corpus makes ZERO new embed_text calls",
                  len(calls) == 0, results)
            check("re-run still returns all 9 vectors (from checkpoint, not re-embedded)",
                  len(keys2) == 9, results)
    finally:
        kb_embed.embed_text = orig_embed_text


# ---------------------------------------------------------------------------------------------
# test c: a PARTIALLY checkpointed corpus (simulating a Ctrl-C / crash mid-run) only embeds the
# remaining docs on resume -- never re-embeds the ones already checkpointed.
# ---------------------------------------------------------------------------------------------

def _check_partial_checkpoint_resumes_only_remaining_docs(results: dict) -> None:
    import build_full_corpus as bfc
    import kb_embed

    orig_embed_text = kb_embed.embed_text
    spy, calls = _make_spy_embed_text()
    kb_embed.embed_text = spy
    try:
        with tempfile.TemporaryDirectory(prefix="bfc_test_") as ckpt_dir:
            corpus = _fake_corpus(10)
            path = bfc.checkpoint_path("glap", "fake", ckpt_dir)

            # simulate a crash after embedding only the first 7 docs.
            import numpy as np

            done = {d["_id"]: np.zeros(8, dtype="float32") + i for i, d in enumerate(corpus[:7])}
            bfc.save_checkpoint(path, done)
            calls.clear()

            keys = bfc.embed_full_corpus_checkpointed(
                "glap", subset="fake", batch_size=3, checkpoint_dir=ckpt_dir,
                corpus_override=corpus, progress=False,
            )
            embedded_ids = {t for batch in calls for t in batch}
            expected_remaining_texts = {
                f"{d.get('title', '')} {d.get('text', '')}".strip() for d in corpus[7:]
            }
            check("resume only re-embeds the 3 NOT-yet-checkpointed docs",
                  embedded_ids == expected_remaining_texts, results)
            check("resume's final mapping still covers all 10 docs",
                  len(keys) == 10, results)
            check("resume preserves the 7 already-checkpointed vectors byte-for-byte",
                  all(np.array_equal(keys[d["_id"]], done[d["_id"]]) for d in corpus[:7]),
                  results)
    finally:
        kb_embed.embed_text = orig_embed_text


# ---------------------------------------------------------------------------------------------
# test d: a stale checkpoint entry (doc id no longer present in the current corpus) is dropped,
# never silently carried forward as a ghost row in the returned mapping.
# ---------------------------------------------------------------------------------------------

def _check_stale_checkpoint_entries_are_dropped(results: dict) -> None:
    import build_full_corpus as bfc
    import kb_embed
    import numpy as np

    orig_embed_text = kb_embed.embed_text
    spy, calls = _make_spy_embed_text()
    kb_embed.embed_text = spy
    try:
        with tempfile.TemporaryDirectory(prefix="bfc_test_") as ckpt_dir:
            corpus = _fake_corpus(5)
            path = bfc.checkpoint_path("glap", "fake", ckpt_dir)
            stale = {d["_id"]: np.zeros(8, dtype="float32") for d in corpus}
            stale["doc999-no-longer-exists"] = np.ones(8, dtype="float32")
            bfc.save_checkpoint(path, stale)
            calls.clear()

            keys = bfc.embed_full_corpus_checkpointed(
                "glap", subset="fake", batch_size=10, checkpoint_dir=ckpt_dir,
                corpus_override=corpus, progress=False,
            )
            check("stale ghost doc id is absent from the returned mapping",
                  "doc999-no-longer-exists" not in keys, results)
            check("all 5 real docs are present, none re-embedded (all pre-checkpointed)",
                  set(keys.keys()) == {d["_id"] for d in corpus} and len(calls) == 0, results)
    finally:
        kb_embed.embed_text = orig_embed_text


# ---------------------------------------------------------------------------------------------
# test e: main()'s CLI wiring -- --embed-only skips the persist call; without it, the persist
# function IS invoked with the freshly-embedded precomputed_keys.
# ---------------------------------------------------------------------------------------------

def _check_main_cli_wiring(results: dict) -> None:
    import build_full_corpus as bfc
    import kb_embed
    import squtr

    orig_embed_text = kb_embed.embed_text
    orig_load_full_corpus = squtr.load_full_corpus
    spy, _calls = _make_spy_embed_text()
    kb_embed.embed_text = spy
    corpus = _fake_corpus(6)
    squtr.load_full_corpus = lambda subset="fiqa": corpus

    import kb_batch_build as kbb

    orig_build_squtr_corpus_source = kbb.build_squtr_corpus_source
    build_calls = []

    def _fake_build(*a, **kw):
        build_calls.append((a, kw))
        return {"status": "built", "n_corpus_docs": len(corpus), "five_axis_audit": {"overall": "CLEAN"}}

    kbb.build_squtr_corpus_source = _fake_build
    try:
        with tempfile.TemporaryDirectory(prefix="bfc_test_") as ckpt_dir:
            rc_embed_only = bfc.main([
                "--embedder", "glap", "--subset", "fake", "--batch-size", "3",
                "--checkpoint-dir", ckpt_dir, "--embed-only",
            ])
            check("main(--embed-only) returns exit code 0", rc_embed_only == 0, results)
            check("main(--embed-only) never calls build_squtr_corpus_source",
                  len(build_calls) == 0, results)

            rc_full = bfc.main([
                "--embedder", "glap", "--subset", "fake", "--batch-size", "3",
                "--checkpoint-dir", ckpt_dir,
            ])
            check("main() without --embed-only returns exit code 0 on a 'built' status",
                  rc_full == 0, results)
            check("main() without --embed-only calls build_squtr_corpus_source exactly once",
                  len(build_calls) == 1, results)
            passed_keys = build_calls[0][1].get("precomputed_keys")
            check("main() passes a complete precomputed_keys mapping (one vector per corpus doc)",
                  passed_keys is not None and set(passed_keys.keys()) == {d["_id"] for d in corpus},
                  results)
            check("main() passes corpus_mode='full'",
                  build_calls[0][1].get("corpus_mode") == "full", results)
    finally:
        kb_embed.embed_text = orig_embed_text
        squtr.load_full_corpus = orig_load_full_corpus
        kbb.build_squtr_corpus_source = orig_build_squtr_corpus_source


def test_fresh_run_embeds_all_docs_in_batches() -> None:
    results: dict[str, bool] = {}
    _check_fresh_run_embeds_all_docs_in_batches(results)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"failed checks: {failed}"


def test_rerun_after_full_checkpoint_makes_no_new_calls() -> None:
    results: dict[str, bool] = {}
    _check_rerun_after_full_checkpoint_makes_no_new_calls(results)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"failed checks: {failed}"


def test_partial_checkpoint_resumes_only_remaining_docs() -> None:
    results: dict[str, bool] = {}
    _check_partial_checkpoint_resumes_only_remaining_docs(results)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"failed checks: {failed}"


def test_stale_checkpoint_entries_are_dropped() -> None:
    results: dict[str, bool] = {}
    _check_stale_checkpoint_entries_are_dropped(results)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"failed checks: {failed}"


def test_main_cli_wiring() -> None:
    results: dict[str, bool] = {}
    _check_main_cli_wiring(results)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"failed checks: {failed}"


def main() -> int:
    results: dict[str, bool] = {}
    _check_fresh_run_embeds_all_docs_in_batches(results)
    _check_rerun_after_full_checkpoint_makes_no_new_calls(results)
    _check_partial_checkpoint_resumes_only_remaining_docs(results)
    _check_stale_checkpoint_entries_are_dropped(results)
    _check_main_cli_wiring(results)

    print("\n=== BUILD_FULL_CORPUS TEST ===")
    print(json.dumps(results, indent=2))
    all_pass = all(results.values())
    print("BUILD_FULL_CORPUS_TEST_PASS" if all_pass else "BUILD_FULL_CORPUS_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

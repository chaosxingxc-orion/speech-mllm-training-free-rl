"""scripts/knowledge/test_pseudo_question.py — fake-model checks for
``kb_batch_build``'s ``key_form='pseudo-question'`` path (续21-A① q2q keys, M1 engineering-base
item 3, 2026-07-13).

Fully OFFLINE and GPU-free: a tmp ``SPEECHRL_KB_DIR`` (never the real
``E:/speechrl-knowledge`` — same safety pattern as ``test_kb_gate.py``), ``kb_embed.embed_text``
monkeypatched to a deterministic FAKE embedder (no network/model download), and
``synthesize_pseudo_question_keys``'s ONE seam (``generate_fn``) fed a FAKE generator instead of a
real llama-server HTTP call — proving the build/persistence/provenance pipeline end to end without
touching a GPU. ``registry.LOADERS`` is temporarily patched with a synthetic dataset (restored in
``finally``), mirroring ``scripts/baselines/test_phase_a_e2e.py``'s established convention.

The REAL generation smoke (2 evidence values, real resident llama-server, brief GPU window) is a
separate, manually-run validation step — see the module docstring of
``kb_batch_build.build_pseudo_question_source`` and the M1 task report; not part of this file (this
file must stay runnable with no GPU / no data root).

Each check is a bare ``test_*()`` function, pytest-collectible; ``main()`` also runs every one
standalone with a PASS/FAIL summary (mirrors ``test_kb_gate.py``/``test_phase_a_e2e.py``):

    python -u scripts/knowledge/test_pseudo_question.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))                       # scripts/knowledge
SCRIPTS = os.path.dirname(HERE)                                          # scripts
LOADERS_DIR = os.path.join(SCRIPTS, "loaders")
for _p in (HERE, LOADERS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_tmp_kb_dir() -> str:
    kb_dir = os.environ.get("SPEECHRL_KB_DIR")
    if not kb_dir:
        kb_dir = tempfile.mkdtemp(prefix="pseudo_question_test_kb_")
    resolved = os.path.abspath(kb_dir).replace("\\", "/").lower()
    assert "speechrl-knowledge" not in resolved, (
        f"refusing to run: SPEECHRL_KB_DIR looks like the real persistent KB ({kb_dir}). "
        "Point this at a tmp dir."
    )
    os.environ["SPEECHRL_KB_DIR"] = kb_dir
    return kb_dir


# ---------------------------------------------------------------------------------------------
# _parse_pseudo_questions (pure, no I/O)
# ---------------------------------------------------------------------------------------------

def test_parse_numbered_list():
    import kb_batch_build as kbb

    raw = "1. What color is the sky?\n2. Why is water wet?\n3. When did this happen?"
    qs = kbb._parse_pseudo_questions(raw, k=3)
    assert qs == ["What color is the sky?", "Why is water wet?", "When did this happen?"]


def test_parse_bullets_and_blank_lines():
    import kb_batch_build as kbb

    raw = "- Q one?\n\n* Q two?\n\n\n• Q three?\n"
    qs = kbb._parse_pseudo_questions(raw, k=3)
    assert qs == ["Q one?", "Q two?", "Q three?"]


def test_parse_truncates_to_k_never_fabricates_more():
    import kb_batch_build as kbb

    raw = "\n".join(f"Q{i}?" for i in range(10))
    qs = kbb._parse_pseudo_questions(raw, k=3)
    assert len(qs) == 3


def test_parse_shortfall_not_padded():
    """Fewer parseable lines than k -- recorded honestly, never fabricated/duplicated to reach k."""
    import kb_batch_build as kbb

    raw = "Only one question?"
    qs = kbb._parse_pseudo_questions(raw, k=3)
    assert qs == ["Only one question?"]


def test_parse_dedupes_order_preserving():
    import kb_batch_build as kbb

    raw = "Same question?\nSame question?\nDifferent question?"
    qs = kbb._parse_pseudo_questions(raw, k=5)
    assert qs == ["Same question?", "Different question?"]


# ---------------------------------------------------------------------------------------------
# synthesize_pseudo_question_keys (fake generate_fn -- the one seam, no HTTP)
# ---------------------------------------------------------------------------------------------

def _fake_generate_fn(prompt: str, seed: int, temperature: float = 0.7, max_tokens: int = 200) -> str:
    """Deterministic FAKE generator: returns exactly 3 distinct questions tagged with `seed`, so a
    test can verify per-value seed threading (seed = seed_base + value_index) end to end."""
    return "\n".join(f"fake question {seed}-{i}?" for i in range(3))


def test_synthesize_pseudo_question_keys_shapes_and_seeds():
    import kb_batch_build as kbb

    evidence = [
        {"value": "passage A text", "from_item_id": "ev-0"},
        {"value": "passage B text", "from_item_id": "ev-1"},
    ]
    records, prov = kbb.synthesize_pseudo_question_keys(
        evidence, k=3, backbone="fake-backbone", seed=1000, generate_fn=_fake_generate_fn,
    )
    assert len(records) == 6  # 2 evidence values x 3 questions each
    assert all(r["value"] in ("passage A text", "passage B text") for r in records)
    assert records[0]["key_text"] == "fake question 1000-0?"
    assert records[0]["from_item_id"] == "ev-0|pq0"
    assert records[3]["key_text"] == "fake question 1001-0?"  # value index 1 -> seed 1000+1
    assert records[3]["from_item_id"] == "ev-1|pq0"

    assert prov["k_requested"] == 3
    assert prov["seed_base"] == 1000
    assert prov["n_values"] == 2 and prov["n_synthesized_keys"] == 6
    assert [pv["generation_seed"] for pv in prov["per_value_questions"]] == [1000, 1001]
    assert all(pv["n_generated"] == 3 for pv in prov["per_value_questions"])


def test_synthesize_pseudo_question_keys_honors_real_shortfall():
    import kb_batch_build as kbb

    def _short_generate_fn(prompt, seed, temperature=0.7, max_tokens=200):
        return "only one line"

    evidence = [{"value": "passage", "from_item_id": "ev-0"}]
    records, prov = kbb.synthesize_pseudo_question_keys(
        evidence, k=3, generate_fn=_short_generate_fn,
    )
    assert len(records) == 1  # only 1 question actually parsed, never padded to k=3
    assert prov["per_value_questions"][0]["n_generated"] == 1


# ---------------------------------------------------------------------------------------------
# fake embed_text (no network) -- mirrors test_phase_a_e2e.py's fake_embed_text convention.
# ---------------------------------------------------------------------------------------------

FAKE_DIM = 24


def _fake_vec(s: str):
    import hashlib

    import numpy as np

    seed = int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)
    v = rng.randn(FAKE_DIM).astype("float32")
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _fake_embed_text(texts, embedder: str = "auto", device: str = "cpu", source_lang: str = "eng_Latn"):
    import numpy as np

    return f"fake-text:{embedder}", np.stack([_fake_vec(t) for t in texts]), None


# ---------------------------------------------------------------------------------------------
# end-to-end: build_pseudo_question_source via corpus_records= (ticket #37 item 2 fix -- the OLD
# dataset_loader_key= generic-loader path is REMOVED; these fixtures are shaped like the CORPUS
# side build_squtr_corpus_source produces, never a query-shaped loader Row).
# ---------------------------------------------------------------------------------------------

def _fake_corpus_records(n: int, prefix: str = "row") -> list[dict]:
    """Pre-audited, in-memory CORPUS-DOCUMENT records -- the exact shape
    ``build_squtr_corpus_source`` produces (``{"value", "from_item_id"}``, no query text
    anywhere)."""
    return [
        {
            "value": f"evidence passage number {i} about topic {i % 3}",
            "from_item_id": f"{prefix}-{i}",
        }
        for i in range(n)
    ]


def test_build_pseudo_question_source_e2e_corpus_records(results: dict | None = None) -> None:
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb
    import kb_embed
    from kb_schema import load_manifest, load_values

    orig_embed_text = kb_embed.embed_text
    kb_embed.embed_text = _fake_embed_text
    fake_records = _fake_corpus_records(5, prefix="pq")
    try:
        result = kbb.build_pseudo_question_source(
            corpus_records=fake_records, dataset_label="fake_pq_ds", text_embedder="glap",
            pool_split="dev", k=3, backbone="fake-backbone", seed=20260713, n=5,
            generate_fn=_fake_generate_fn,
        )
        assert result["status"] == "built"
        assert result["n_evidence_values"] == 5
        assert result["n_synthesized_keys"] == 15  # 5 values x 3 questions
        assert result["corpus_source"] is None
        assert result["query_paths"] == ["audio", "text"]  # glap is audio-query-capable, no escape hatch needed

        manifest = result["manifest"]
        assert manifest["key_modality"] == "text"
        assert manifest["n_entries"] == 15
        assert manifest["source"] == "fake_pq_ds__glap__pseudo-question-k3__text-fact"
        assert manifest.get("content_hash")

        # provenance: exact prompt template + seeds + per-value question texts, IN the manifest note
        note = manifest["created_note"]
        assert "续21-A①" in note
        assert kbb.PSEUDO_QUESTION_PROMPT_TEMPLATE.split("{")[0] in note  # template prefix present
        assert "seed_base=20260713" in note
        assert '"n_generated": 3' in note
        assert "query_paths=['audio', 'text']" in note

        reloaded_manifest = load_manifest("fake_pq_ds__glap__pseudo-question-k3__text-fact")
        assert reloaded_manifest.content_hash == manifest["content_hash"]

        persisted = load_values("fake_pq_ds__glap__pseudo-question-k3__text-fact")
        assert len(persisted) == 15
        # each persisted row's key_text_ref is the exact synthesized question text (schema fix)
        assert all(v.key_text_ref is not None and v.key_text_ref.startswith("fake question ")
                   for v in persisted)
        # every row's value is one of the 5 parent evidence texts, repeated 3x each
        parent_values = {r["value"] for r in fake_records}
        assert all(v.value in parent_values for v in persisted)
        # from_item_id threading: "<parent>|pq<i>"
        assert all("|pq" in (v.provenance.get("from_item_id") or "") for v in persisted)
    finally:
        kb_embed.embed_text = orig_embed_text


def test_build_pseudo_question_source_eval_manifest_disjointness_enforced():
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb
    import kb_embed

    orig_embed_text = kb_embed.embed_text
    kb_embed.embed_text = _fake_embed_text
    fake_records = _fake_corpus_records(3, prefix="ev")
    try:
        raised = False
        try:
            kbb.build_pseudo_question_source(
                corpus_records=fake_records, dataset_label="fake_pq_ds2", text_embedder="glap",
                pool_split="dev", k=2, backbone="fake-backbone", seed=1, n=3,
                generate_fn=_fake_generate_fn, eval_manifest=["ev-0"],
            )
        except ValueError:
            raised = True
        assert raised
    finally:
        kb_embed.embed_text = orig_embed_text


def test_build_pseudo_question_source_rejects_non_text_capable_embedder():
    import kb_batch_build as kbb

    raised = False
    try:
        kbb.build_pseudo_question_source(text_embedder="clsp", n=1, generate_fn=_fake_generate_fn)
    except ValueError:
        raised = True
    assert raised  # 'clsp' has no text-embedding path (kb_embed.can_embed_text) -- caught BEFORE
    # even validating corpus_records/corpus_source, so neither needs to be supplied for this check.


# ---------------------------------------------------------------------------------------------
# ticket #37 item 2: removed dataset_loader_key= path + exactly-one-of corpus_records/corpus_source
# ---------------------------------------------------------------------------------------------

def test_build_pseudo_question_source_rejects_removed_dataset_loader_key_kwarg():
    import kb_batch_build as kbb

    raised_type_error = False
    try:
        kbb.build_pseudo_question_source(
            dataset_loader_key="squtr", text_embedder="glap", generate_fn=_fake_generate_fn,
        )
    except TypeError as e:
        raised_type_error = True
        assert "dataset_loader_key" in str(e) and "corpus_records" in str(e)
    assert raised_type_error


def test_build_pseudo_question_source_rejects_removed_value_spec_kwarg():
    import kb_batch_build as kbb

    raised = False
    try:
        kbb.build_pseudo_question_source(
            value_spec="text", text_embedder="glap", generate_fn=_fake_generate_fn,
        )
    except TypeError:
        raised = True
    assert raised


def test_build_pseudo_question_source_requires_exactly_one_of_corpus_records_or_corpus_source():
    import kb_batch_build as kbb

    raised_neither = False
    try:
        kbb.build_pseudo_question_source(text_embedder="glap", generate_fn=_fake_generate_fn)
    except ValueError:
        raised_neither = True
    assert raised_neither

    raised_both = False
    try:
        kbb.build_pseudo_question_source(
            text_embedder="glap", generate_fn=_fake_generate_fn,
            corpus_records=_fake_corpus_records(1), corpus_source="some_source",
        )
    except ValueError:
        raised_both = True
    assert raised_both


def test_build_pseudo_question_source_corpus_records_missing_value_key_raises():
    import kb_batch_build as kbb

    raised = False
    try:
        kbb.build_pseudo_question_source(
            corpus_records=[{"from_item_id": "x0"}],  # missing 'value'
            text_embedder="glap", generate_fn=_fake_generate_fn,
        )
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------------------------
# ticket #37 item 2: corpus_source= path (an EXISTING persisted, text-keyed, CLEAN KB source)
# ---------------------------------------------------------------------------------------------

def _build_fake_corpus_source(source_name: str, n: int = 4, dim: int = 16, seed: int = 0,
                               key_modality: str = "text", leaking: bool = False) -> dict:
    """Directly persists a tiny corpus-side source via kb_build.build_source (precomputed keys --
    no real embedder needed) -- stands in for a real build_squtr_corpus_source output."""
    import numpy as np

    import kb_build

    recs = []
    for i in range(n):
        vec = np.random.RandomState(seed + i).randn(dim).astype("float32")
        vec = vec / np.linalg.norm(vec)
        value = f"corpus doc {i} evidence text" + (" -- the answer is banana42" if leaking else "")
        key_field = "key_text" if key_modality == "text" else "key_audio_ref"
        recs.append({
            key_field: f"corpus-doc-{i}", "value": value,
            "from_item_id": f"{source_name}|corpus|{i}", "precomputed_key": vec,
        })
    audit_golds = ["banana42"] * n if leaking else []
    return kb_build.build_source(
        source_name, f"{source_name}-dataset", revision=None, records=recs,
        key_modality=key_modality, value_type="text-fact", embedder="glap",
        audit_golds=audit_golds, scrub=False, pool_split="test",
        note="test fixture -- fake corpus-side source (ticket #37 item 2)",
        force_persist=leaking,  # a LEAKAGE-verdict source refuses to persist otherwise -- this
        # helper needs one ON DISK (forced, stamped NOT-admissible) so build_pseudo_question_source
        # has something real to reject via corpus_source=.
    )


def test_build_pseudo_question_source_via_corpus_source_reference():
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb
    import kb_embed
    from kb_schema import load_values

    _build_fake_corpus_source("fake_corpus_src_ok", n=4)
    orig_embed_text = kb_embed.embed_text
    kb_embed.embed_text = _fake_embed_text
    try:
        result = kbb.build_pseudo_question_source(
            corpus_source="fake_corpus_src_ok", dataset_label="fake_corpus_src_ok_pq",
            text_embedder="glap", k=2, backbone="fake-backbone", seed=5,
            generate_fn=_fake_generate_fn,
        )
    finally:
        kb_embed.embed_text = orig_embed_text

    assert result["status"] == "built"
    assert result["corpus_source"] == "fake_corpus_src_ok"
    assert result["n_evidence_values"] == 4
    assert result["n_synthesized_keys"] == 8  # 4 values x 2 questions
    persisted = load_values(result["manifest"]["source"])
    assert all(v.value.startswith("corpus doc") for v in persisted)


def test_build_pseudo_question_source_corpus_source_must_be_text_keyed():
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb

    _build_fake_corpus_source("fake_corpus_src_audio", n=2, key_modality="audio")
    raised = False
    try:
        kbb.build_pseudo_question_source(
            corpus_source="fake_corpus_src_audio", text_embedder="glap", generate_fn=_fake_generate_fn,
        )
    except ValueError:
        raised = True
    assert raised


def test_build_pseudo_question_source_corpus_source_must_be_clean():
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb

    _build_fake_corpus_source("fake_corpus_src_leaking", n=2, leaking=True)  # scrub=False -> stays LEAKAGE
    raised = False
    try:
        kbb.build_pseudo_question_source(
            corpus_source="fake_corpus_src_leaking", text_embedder="glap", generate_fn=_fake_generate_fn,
        )
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------------------------
# ticket #37 item 5: no silent 'auto' embedder (audio-query-capable set / escape hatch)
# ---------------------------------------------------------------------------------------------

def test_build_pseudo_question_source_auto_embedder_hard_errors_without_escape_hatch():
    import kb_batch_build as kbb

    raised = False
    try:
        kbb.build_pseudo_question_source(
            corpus_records=_fake_corpus_records(2), text_embedder="auto",
            generate_fn=_fake_generate_fn,
        )
    except ValueError as e:
        raised = True
        assert "AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS" in str(e)
    assert raised


def test_build_pseudo_question_source_auto_embedder_allowed_with_escape_hatch():
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb
    import kb_embed

    orig_embed_text = kb_embed.embed_text
    kb_embed.embed_text = _fake_embed_text
    try:
        result = kbb.build_pseudo_question_source(
            corpus_records=_fake_corpus_records(2, prefix="autoesc"),
            dataset_label="fake_pq_auto_escape", text_embedder="auto", k=2,
            generate_fn=_fake_generate_fn, allow_text_only_query=True,
        )
    finally:
        kb_embed.embed_text = orig_embed_text

    assert result["status"] == "built"
    assert result["query_paths"] == ["text"]
    assert "query_paths=['text']" in result["manifest"]["created_note"]


def test_build_pseudo_question_source_glap_audio_query_capable_needs_no_escape_hatch():
    import kb_batch_build as kbb

    assert "glap" in kbb.AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS
    assert "omni-embed-nemotron" in kbb.AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS
    assert "lco-3b" in kbb.AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS
    assert "lco-7b" in kbb.AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS
    assert "qwen3-omni-own" in kbb.AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS
    assert "auto" not in kbb.AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS
    assert "minilm" not in kbb.AUDIO_QUERY_CAPABLE_TEXT_EMBEDDERS


# ---------------------------------------------------------------------------------------------
# ticket #37 item 2: REAL squtr qrels/corpus semantic test (offline, no GPU/network) -- proves
# every built VALUE is a genuine qrels-pointed CORPUS document, and that none of the sampled
# QUERIES' own text ever appears as a persisted value (the exact P0-1 wrong-object failure mode
# this item closes).
# ---------------------------------------------------------------------------------------------

def test_build_pseudo_question_source_real_squtr_corpus_only_semantic():
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb
    import kb_embed
    from kb_schema import load_values

    try:
        import squtr
        retrieval = squtr.load_squtr_retrieval(
            subset="fiqa", noise_level="clean", n=8, seed=20260705, n_distractors=0,
        )
    except FileNotFoundError as e:
        print(f"  [test_pseudo_question] SKIPPED (squtr data not available offline: {e})")
        return

    corpus_sample, qrels, queries = retrieval["corpus_sample"], retrieval["qrels"], retrieval["queries"]
    assert corpus_sample, "no squtr fiqa corpus sample -- see docs/datasets.lock.json"
    gold_docids = {docid for _, docid, _ in qrels}
    # n_distractors=0 -> build_mini_corpus returns ONLY the qrels-referenced gold docs
    assert all(d["_id"] in gold_docids for d in corpus_sample)

    doc_texts = [f"{d.get('title', '')} {d.get('text', '')}".strip() for d in corpus_sample]
    corpus_records = [
        {"value": t, "from_item_id": f"squtr-fiqa-clean|corpus|{d['_id']}"}
        for d, t in zip(corpus_sample, doc_texts)
    ]

    orig_embed_text = kb_embed.embed_text
    kb_embed.embed_text = _fake_embed_text
    try:
        result = kbb.build_pseudo_question_source(
            corpus_records=corpus_records, dataset_label="squtr-fiqa-clean-real-test",
            text_embedder="glap", k=2, backbone="fake-backbone", seed=999,
            generate_fn=_fake_generate_fn,
        )
    finally:
        kb_embed.embed_text = orig_embed_text

    assert result["status"] == "built"
    persisted = load_values(result["manifest"]["source"])
    # kb_build.build_source(scrub=True) UNCONDITIONALLY whitespace-collapses every persisted value
    # (kb_audit.scrub_golds's final ``re.sub(r"\s+", " ", s).strip()`` -- runs regardless of
    # whether any gold actually matched; pre-existing behavior, also true of
    # build_squtr_corpus_source's own persisted corpus sources) -- normalize doc_texts the SAME way
    # before the containment/equality checks below, or a real whitespace run in the raw corpus text
    # (fiqa passages have several) would spuriously look like a mismatch.
    import re as _re

    def _ws_collapse(s: str) -> str:
        return _re.sub(r"\s+", " ", s).strip()

    doc_text_set = {_ws_collapse(t) for t in doc_texts}
    query_texts = [q["text"] for q in queries]

    # every persisted value is a genuine qrels-pointed corpus document (never a query)
    assert persisted, "no rows persisted -- test fixture bug"
    assert all(v.value in doc_text_set for v in persisted)

    # none of the sampled queries' own text appears as (or is contained in) a persisted value --
    # the P0-1 wrong-object failure mode this item closes would have every value BE a query's text.
    for qt in query_texts:
        qt_ws = _ws_collapse(qt)
        assert qt_ws not in doc_text_set
        assert all(qt_ws not in v.value for v in persisted)


# ---------------------------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    results: dict[str, bool] = {}
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}", flush=True)
            results[name] = True
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            results[name] = False
    all_pass = all(results.values())
    print("\n=== PSEUDO_QUESTION TEST ===")
    print(json.dumps(results, indent=2))
    print("PSEUDO_QUESTION_TEST_PASS" if all_pass else "PSEUDO_QUESTION_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

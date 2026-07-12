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
# end-to-end: build_pseudo_question_source against a synthetic registry dataset
# ---------------------------------------------------------------------------------------------

def _fake_rows(n: int, prefix: str = "row") -> list[dict]:
    return [
        {
            "wav": f"unused-{prefix}-{i}.wav",  # pseudo-question keys are TEXT; wav is never read
            "gold": f"gold-{prefix}-{i}",
            "meta": {"item_id": f"{prefix}-{i}", "text": f"evidence passage number {i} about topic {i % 3}"},
        }
        for i in range(n)
    ]


def test_build_pseudo_question_source_e2e(results: dict | None = None) -> None:
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb
    import kb_embed
    import registry
    from kb_schema import load_manifest, load_values

    orig_embed_text = kb_embed.embed_text
    orig_loader = registry.LOADERS.get("fake_pq_ds")
    kb_embed.embed_text = _fake_embed_text
    fake_rows = _fake_rows(5, prefix="pq")
    registry.LOADERS["fake_pq_ds"] = lambda split="dev", n=None, seed=0: (
        fake_rows[:n] if n is not None else list(fake_rows)
    )
    try:
        result = kbb.build_pseudo_question_source(
            "fake_pq_ds", text_embedder="auto", pool_split="dev", value_spec="text",
            k=3, backbone="fake-backbone", seed=20260713, n=5, generate_fn=_fake_generate_fn,
        )
        assert result["status"] == "built"
        assert result["n_evidence_values"] == 5
        assert result["n_synthesized_keys"] == 15  # 5 values x 3 questions

        manifest = result["manifest"]
        assert manifest["key_modality"] == "text"
        assert manifest["n_entries"] == 15
        # value_spec="text" is not itself a kb_schema.VALUE_TYPES member -> falls back to
        # "text-fact" (build_one does the exact same fallback for the same reason).
        assert manifest["source"] == "fake_pq_ds__auto__pseudo-question-k3__text-fact"
        assert manifest.get("content_hash")

        # provenance: exact prompt template + seeds + per-value question texts, IN the manifest note
        note = manifest["created_note"]
        assert "续21-A①" in note
        assert kbb.PSEUDO_QUESTION_PROMPT_TEMPLATE.split("{")[0] in note  # template prefix present
        assert "seed_base=20260713" in note
        assert '"n_generated": 3' in note

        reloaded_manifest = load_manifest("fake_pq_ds__auto__pseudo-question-k3__text-fact")
        assert reloaded_manifest.content_hash == manifest["content_hash"]

        persisted = load_values("fake_pq_ds__auto__pseudo-question-k3__text-fact")
        assert len(persisted) == 15
        # each persisted row's key_text_ref is the exact synthesized question text (schema fix)
        assert all(v.key_text_ref is not None and v.key_text_ref.startswith("fake question ")
                   for v in persisted)
        # every row's value is one of the 5 parent evidence texts, repeated 3x each
        parent_values = {r["meta"]["text"] for r in fake_rows}
        assert all(v.value in parent_values for v in persisted)
        # from_item_id threading: "<parent>|pq<i>"
        assert all("|pq" in (v.provenance.get("from_item_id") or "") for v in persisted)
    finally:
        kb_embed.embed_text = orig_embed_text
        if orig_loader is not None:
            registry.LOADERS["fake_pq_ds"] = orig_loader
        else:
            registry.LOADERS.pop("fake_pq_ds", None)


def test_build_pseudo_question_source_eval_manifest_disjointness_enforced():
    _ensure_tmp_kb_dir()

    import kb_batch_build as kbb
    import kb_embed
    import registry

    orig_embed_text = kb_embed.embed_text
    orig_loader = registry.LOADERS.get("fake_pq_ds2")
    kb_embed.embed_text = _fake_embed_text
    fake_rows = _fake_rows(3, prefix="ev")
    registry.LOADERS["fake_pq_ds2"] = lambda split="dev", n=None, seed=0: (
        fake_rows[:n] if n is not None else list(fake_rows)
    )
    try:
        raised = False
        try:
            kbb.build_pseudo_question_source(
                "fake_pq_ds2", text_embedder="auto", pool_split="dev", value_spec="text",
                k=2, backbone="fake-backbone", seed=1, n=3, generate_fn=_fake_generate_fn,
                eval_manifest=["ev-0"],
            )
        except ValueError:
            raised = True
        assert raised
    finally:
        kb_embed.embed_text = orig_embed_text
        if orig_loader is not None:
            registry.LOADERS["fake_pq_ds2"] = orig_loader
        else:
            registry.LOADERS.pop("fake_pq_ds2", None)


def test_build_pseudo_question_source_rejects_non_text_capable_embedder():
    import kb_batch_build as kbb

    raised = False
    try:
        kbb.build_pseudo_question_source(
            "fake_pq_ds3", text_embedder="clsp", n=1, generate_fn=_fake_generate_fn,
        )
    except ValueError:
        raised = True
    assert raised  # 'clsp' has no text-embedding path (kb_embed.can_embed_text) -- caught BEFORE
    # even touching the registry, so no fake loader is needed for this check.


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

"""scripts/baselines/test_phase_a_e2e.py — ticket #25 Priority-4 (i)+(j): fake-model E2E over ALL
35 Phase-A arms against ONE tiny synthetic dataset, plus a CPU-real ('logmel-stats') smoke over 2
arms with a stub generator, plus (2026-07-12, Proposal-R prereg §5/§10 item 8, G2 gate) the 4
mechanism-control arms (no-retrieval/random-retrieval/oracle-retrieval/gold-transcript) -- 39 total
arms must go green.

Pure-python-reachable, fully OFFLINE and GPU-free:
  - a tmp ``SPEECHRL_KB_DIR`` (never the real ``E:/speechrl-knowledge`` -- same safety pattern as
    ``scripts/knowledge/test_kb_gate.py``, asserted below).
  - ``kb_embed.embed_audio``/``embed_text`` are monkeypatched to a deterministic, dependency-free
    FAKE embedder for the big 35-arm sweep (test i) -- no model download, no GPU, no network. The
    CPU-live smoke (test j) uses the REAL ``logmel-stats`` embedder instead (real librosa audio
    processing, still no network/model download/GPU) to prove the pipeline also works against a
    genuinely-computed embedding space, not just the fake one.
  - ``run_mock.generate_mock`` (and therefore ``_asr_transcribe``/``_hyde_generate``, which call it)
    is monkeypatched to a fake generator -- no HTTP call to a resident backbone server.
  - ``run_baseline.assert_gpu_session_held`` is monkeypatched to a no-op -- no GPU lock check.
  - test k (control arms) TEMPORARILY overrides ``registry.LOADERS["squtr"]``/``["heysquad"]`` with
    fake loaders (restored in ``finally``) and ``run_mock._squtr_corpus_text`` with an in-memory
    fake corpus, so ``control-oracle-retrieval``'s real per-dataset dispatch branches run without
    ever touching squtr's real (multi-GB) zip or heysquad's real parquet shards.

Run:
    python -u scripts/baselines/test_phase_a_e2e.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))              # scripts/baselines
SCRIPTS = os.path.dirname(HERE)                                  # scripts
KNOWLEDGE = os.path.join(SCRIPTS, "knowledge")
LOADERS_DIR = os.path.join(SCRIPTS, "loaders")
for p in (HERE, SCRIPTS, LOADERS_DIR, KNOWLEDGE):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------------------------
# tmp KB root -- NEVER the real E:/speechrl-knowledge default (mirrors test_kb_gate.py's guard).
# ---------------------------------------------------------------------------------------------

def _ensure_tmp_kb_dir() -> str:
    kb_dir = os.environ.get("SPEECHRL_KB_DIR")
    if not kb_dir:
        kb_dir = tempfile.mkdtemp(prefix="phase_a_e2e_kb_")
    resolved = os.path.abspath(kb_dir).replace("\\", "/").lower()
    assert "speechrl-knowledge" not in resolved, (
        f"refusing to run: SPEECHRL_KB_DIR looks like the real persistent KB ({kb_dir}). "
        "Point this at a tmp dir."
    )
    os.environ["SPEECHRL_KB_DIR"] = kb_dir
    return kb_dir


# ---------------------------------------------------------------------------------------------
# tiny synthetic wavs + a fake registry loader -- no dataset/network dependency.
# ---------------------------------------------------------------------------------------------

def _make_wav(path: str, freq: float = 220.0, sr: int = 16000, dur: float = 0.4) -> None:
    n = int(sr * dur)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            v = int(3000 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))


def _fake_rows(wavdir: str, prefix: str, n: int, freq0: float = 180.0) -> list[dict]:
    rows = []
    for i in range(n):
        wav = os.path.join(wavdir, f"{prefix}_{i}.wav")
        _make_wav(wav, freq=freq0 + 11 * i)
        rows.append({
            "wav": wav,
            "gold": f"answer-{prefix}-{i}",
            "meta": {"item_id": f"{prefix}-{i}", "text": f"spoken query {prefix} number {i} about topic {i % 3}"},
        })
    return rows


# ---------------------------------------------------------------------------------------------
# deterministic FAKE embedder (test i) -- hashlib-seeded, NOT Python's hash() (which is
# PYTHONHASHSEED-randomized per process for strings) so the same input always yields the same
# vector even across separate interpreter runs, not just within one.
# ---------------------------------------------------------------------------------------------

FAKE_DIM = 16


def _seed_from(s: str) -> int:
    return int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:8], 16)


def _fake_vec(s: str):
    import numpy as np

    rng = np.random.RandomState(_seed_from(s) % (2**31))
    v = rng.randn(FAKE_DIM).astype("float32")
    return v / max(float(np.linalg.norm(v)), 1e-9)


def fake_embed_audio(wav_paths, embedder: str = "auto", device: str = "cpu"):
    import numpy as np

    return f"fake-audio:{embedder}", np.stack([_fake_vec(p) for p in wav_paths])


def fake_embed_text(texts, embedder: str = "auto", device: str = "cpu"):
    import numpy as np

    return f"fake-text:{embedder}", np.stack([_fake_vec(t) for t in texts]), None


def fake_generate_mock(cfg, wav_path, payload, seed: int = 0) -> str:
    """Stands in for ``run_mock.generate_mock`` (and therefore ``_asr_transcribe``/
    ``_hyde_generate``, which call it) -- no HTTP call to a resident backbone server."""
    tag = payload if isinstance(payload, str) else f"<{len(payload)} messages>"
    return f"fake reply seed={seed} wav={os.path.basename(wav_path)} payload_len={len(str(tag))}"


# ---------------------------------------------------------------------------------------------
# test i: fake-model E2E over ALL 35 Phase-A arms x 1 tiny synthetic dataset.
# ---------------------------------------------------------------------------------------------

def test_e2e_all_arms(results: dict) -> None:
    import dataclasses

    import kb_batch_build as kbb
    import kb_embed
    import kb_retrieve
    import phase_a_cells as pac
    import registry
    import run_mock as rm
    import templates

    # --- monkeypatches: fake embedder, fake generator, no GPU-session check ---
    kb_embed.embed_audio = fake_embed_audio
    kb_embed.embed_text = fake_embed_text
    rm.generate_mock = fake_generate_mock
    rm.rb.assert_gpu_session_held = lambda *a, **kw: None

    # --- register the tiny synthetic dataset ---
    templates.DATASET_KTYPE["fake_e2e"] = "K8"  # free-text QA scoring (score_k8_qa), no MCQ opts

    wavdir = tempfile.mkdtemp(prefix="phase_a_e2e_wav_")
    eval_rows = _fake_rows(wavdir, "eval", 4, freq0=150.0)
    build_rows = _fake_rows(wavdir, "build", 6, freq0=300.0)
    eval_ids = [r["meta"]["item_id"] for r in eval_rows]

    def fake_loader(split: str = "dev", n: int | None = None, seed: int = 0):
        # test-only disambiguation: eval calls this with n=len(eval_rows), kb_batch_build's build
        # calls it with n=len(build_rows) -- deterministic, disjoint item_ids either way.
        return list(eval_rows) if n == len(eval_rows) else list(build_rows)

    registry.LOADERS["fake_e2e"] = fake_loader

    # --- pin REF_CONFIG's embedder to an omni-embed-family token for this test run, so the 3 arms
    # whose REAL implementation needs the cross-modal text-query bridge (asr-transcript-text-key /
    # hyde-single-shot query_construction, ircot-2hop's 2nd hop -- see run_mock._kb_retrieve_text's
    # docstring) have a compatible source to query. Every arm that varies a DIFFERENT dimension
    # keeps query_construction='audio-direct' (never touches the text bridge), so this is safe.
    orig_ref_config = rm.REF_CONFIG
    rm.REF_CONFIG = dataclasses.replace(orig_ref_config, embedder="omni-embed-nemotron")

    try:
        arms = pac.build_phase_a_arms()
        assert len(arms) == 35, f"expected 35 Phase-A arms, got {len(arms)}"

        needed = {(cfg.embedder, cfg.key_org, cfg.value_org) for _dim, _label, cfg in arms}
        built_sources = set()
        for embedder, key_org, value_org in sorted(needed):
            manifest = kbb.build_one(
                embedder, "fake_e2e", "dev", value_spec="text",
                key_org=key_org, value_org=value_org, n=len(build_rows), seed=1,
                eval_manifest=eval_ids,
            )
            built_sources.add(manifest["source"])

        # P1a round-trip: every arm's source_name_for(...) must resolve to something we actually built.
        for _dim, _label, cfg in arms:
            expected = rm.source_name_for("fake_e2e", cfg)
            assert expected in built_sources, (
                f"source naming round-trip FAILED: run_mock.source_name_for produced {expected!r}, "
                f"not among the {len(built_sources)} sources kb_batch_build.build_one actually built"
            )

        # P3g negative check: eval_manifest disjointness guard actually raises on a real overlap.
        raised = False
        try:
            kbb.build_one("omni-embed-nemotron", "fake_e2e", "dev", n=len(build_rows), seed=1,
                           eval_manifest=eval_ids + ["build-0"])
        except ValueError:
            raised = True
        results["P3g eval_manifest disjointness guard raises on overlap"] = raised

        # P3g own-item-exclusion + retrieved-passage-recording unit check (kb_schema.KnowledgeValue
        # instances directly, no KB build needed for this one -- isolates the two helpers exactly).
        from kb_schema import KnowledgeValue

        self_hit = {"value": KnowledgeValue(
            row=0, kid="src:self", source="src", key_modality="audio", value_type="text-fact",
            value="own item text", provenance={"from_item_id": "item-42"}), "sim": 0.99}
        other_hit = {"value": KnowledgeValue(
            row=1, kid="src:other", source="src", key_modality="audio", value_type="text-fact",
            value="other item text", provenance={"from_item_id": "item-99"}), "sim": 0.5}
        filtered = rm._exclude_own_item([self_hit, other_hit], "item-42")
        results["P3g own-item exclusion drops the current item's own hit"] = (
            filtered == [other_hit]
        )
        recorded = rm._record_retrieved([other_hit], "src")
        results["P3g retrieved-passage record carries source/from_item_id/sim/text"] = (
            recorded == [{"source": "src", "from_item_id": "item-99", "sim": 0.5, "text": "other item text"}]
        )

        # --- run every arm for real (fake embedder/generator, real retrieval/scoring logic) ---
        per_arm_status = {}
        for dim, label, cfg in arms:
            try:
                r = rm.run_one_mock("fake_e2e", cfg, "dev", len(eval_rows))
                errors = [it.get("error") for it in r["per_item"] if "error" in it]
                not_impl = [e for e in errors if e and "NotImplementedError" in e]
                has_retrieved_field = all("retrieved" in it for it in r["per_item"] if "error" not in it)
                ok = not errors and has_retrieved_field
                per_arm_status[(dim, label)] = {
                    "ok": ok, "errors": errors, "not_impl": not_impl,
                    "n_scored": r["aggregate"]["n_scored"], "n_total": r["aggregate"]["n_total"],
                }
            except Exception as e:  # noqa: BLE001 -- record, don't abort the sweep
                per_arm_status[(dim, label)] = {"ok": False, "errors": [f"{type(e).__name__}: {e}"],
                                                 "not_impl": [], "exception": True}

        n_fail = sum(0 if v["ok"] else 1 for v in per_arm_status.values())
        n_not_impl = sum(len(v["not_impl"]) for v in per_arm_status.values())
        results["all 35 arms produce a clean result (no per-item errors)"] = (n_fail == 0)
        results["zero NotImplementedError across all 35 arms"] = (n_not_impl == 0)

        print("\n-- per-arm fake-E2E status --")
        for (dim, label), v in per_arm_status.items():
            status = "PASS" if v["ok"] else "FAIL"
            extra = f" errors={v['errors'][:2]}" if not v["ok"] else ""
            print(f"  [{status}] {dim:20s} {label:30s}{extra}")

        # dimension-assertion negative check (P1c): a deliberately wrong-dim query must raise.
        ref_source = rm.source_name_for("fake_e2e", rm.REF_CONFIG)
        src_obj = kb_retrieve.load_source(ref_source)
        bad_embed_query = src_obj["embed_query"]
        src_obj["embed_query"] = lambda qs: [[0.0] * (FAKE_DIM + 1)]
        raised_dim = False
        try:
            kb_retrieve.retrieve(src_obj, [eval_rows[0]["wav"]], topk=2)
        except ValueError as e:
            raised_dim = "dim" in str(e)
        finally:
            src_obj["embed_query"] = bad_embed_query
        results["P1c dimension assertion raises on a mismatched query embedding"] = raised_dim

    finally:
        rm.REF_CONFIG = orig_ref_config


# ---------------------------------------------------------------------------------------------
# test j: CPU-real 'logmel-stats' smoke -- a genuine (non-faked) embedding pipeline, 5 entries,
# 2 arms, with a stub generator (still no GPU/network -- logmel-stats is pure librosa/numpy).
# ---------------------------------------------------------------------------------------------

def test_cpu_live_logmel_stats_smoke(results: dict) -> None:
    import kb_batch_build as kbb
    import kb_embed
    import registry
    import run_mock as rm
    import templates

    # logmel-stats is a real (pure-CPU, no model download) embedder kb_embed.embed_audio already
    # supports -- it just isn't in the kb_embed.EMBEDDERS *metadata* registry kb_batch_build's own
    # gate checks against. Extending that registry (test-only) is the same pattern this task's own
    # exploratory smoke already validated -- see this ticket's manual verification notes.
    kb_embed.EMBEDDERS.setdefault("logmel-stats", {"dim": 128, "license": "n/a", "domain": "CPU smoke-test key"})
    if "logmel-stats" not in rm.EMBEDDERS:
        rm.EMBEDDERS.append("logmel-stats")  # MockConfig.embedder's allowed-arm list (test-only)
    rm.generate_mock = fake_generate_mock
    rm.rb.assert_gpu_session_held = lambda *a, **kw: None
    templates.DATASET_KTYPE["fake_e2e_logmel"] = "K8"

    wavdir = tempfile.mkdtemp(prefix="phase_a_logmel_wav_")
    eval_rows = _fake_rows(wavdir, "logmel-eval", 2, freq0=210.0)
    build_rows = _fake_rows(wavdir, "logmel-build", 5, freq0=400.0)

    def fake_loader(split: str = "dev", n: int | None = None, seed: int = 0):
        return list(eval_rows) if n == len(eval_rows) else list(build_rows)

    registry.LOADERS["fake_e2e_logmel"] = fake_loader

    import dataclasses

    cfg_ref = dataclasses.replace(
        rm.REF_CONFIG, embedder="logmel-stats", key_org="single-utt", value_org="knowledge-passage",
    )
    cfg_topk5 = dataclasses.replace(
        cfg_ref, retrieval=dataclasses.replace(cfg_ref.retrieval, k=2),
    )

    manifest = kbb.build_one("logmel-stats", "fake_e2e_logmel", "dev", value_spec="text",
                              key_org="single-utt", value_org="knowledge-passage",
                              n=len(build_rows), seed=1)
    results["CPU-live: 5-entry logmel-stats KB builds"] = (manifest["n_entries"] == len(build_rows))
    results["CPU-live: manifest.embedder_token == 'logmel-stats'"] = (manifest["embedder_token"] == "logmel-stats")

    for label, cfg in (("ref-config (dense-topk k=3)", cfg_ref), ("dense-topk k=2", cfg_topk5)):
        r = rm.run_one_mock("fake_e2e_logmel", cfg, "dev", len(eval_rows))
        errors = [it.get("error") for it in r["per_item"] if "error" in it]
        ok = not errors and all("retrieved" in it for it in r["per_item"])
        results[f"CPU-live: arm '{label}' runs clean end-to-end"] = ok
        if not ok:
            print(f"  CPU-live arm {label!r} errors: {errors}")


# ---------------------------------------------------------------------------------------------
# fake rows shaped like squtr's/heysquad's real Row contracts -- just enough for
# run_mock._oracle_hits_squtr/_oracle_hits_heysquad's REAL dispatch branches to run, without ever
# touching squtr's real (multi-GB) zip or heysquad's real parquet shards.
# ---------------------------------------------------------------------------------------------

def _fake_squtr_rows(wavdir: str, prefix: str, n: int, freq0: float, docids: list[str]) -> list[dict]:
    rows = []
    for i in range(n):
        wav = os.path.join(wavdir, f"{prefix}_{i}.wav")
        _make_wav(wav, freq=freq0 + 11 * i)
        rows.append({
            "wav": wav,
            "gold": [{"corpus_id": docids[i % len(docids)], "score": 1}],  # score>0 == qrels-positive
            "meta": {"item_id": f"{prefix}-{i}", "subset": "fiqa",
                     "text": f"fake spoken squtr query {prefix} number {i}"},
        })
    return rows


def _fake_heysquad_rows(wavdir: str, prefix: str, n: int, freq0: float) -> list[dict]:
    """``context`` deliberately CONTAINS the gold answer verbatim -- mirrors heysquad's own real
    LEAKAGE WARNING (heysquad.py) so control-oracle-retrieval's scrub+CLEAN-audit path and
    control-gold-transcript's UNSCRUBBED path are both exercised for real, not against inert text.
    """
    rows = []
    for i in range(n):
        wav = os.path.join(wavdir, f"{prefix}_{i}.wav")
        _make_wav(wav, freq=freq0 + 11 * i)
        answer = f"answer{i}"
        context = f"This is a fake SQuAD passage about topic {i}. The correct answer is {answer}, right here."
        rows.append({
            "wav": wav,
            "gold": [answer],
            "meta": {"item_id": f"{prefix}-{i}", "context": context},
        })
    return rows


_FAKE_SQUTR_CORPUS = {
    "docA": "Fake corpus passage A: widgets are small mechanical parts used inside gadgets.",
    "docB": "Fake corpus passage B: gadgets are electronic devices that use widgets internally.",
}


def _fake_squtr_corpus_text(subset: str, docids: list) -> dict:
    return {d: _FAKE_SQUTR_CORPUS[d] for d in docids if d in _FAKE_SQUTR_CORPUS}


# ---------------------------------------------------------------------------------------------
# test k: fake-model E2E over the 4 Proposal-R prereg mechanism-control arms (wiki/2026-07-11-
# proposal-R-prereg-draft.md §5/§10 item 8) -- no-retrieval / random-retrieval / oracle-retrieval /
# gold-transcript. Each runs through the REAL run_mock.run_one_mock (not a bare unit call of the
# helper functions), mirroring test_e2e_all_arms's own style, reusing the SAME module-level fake
# embedder/generator/no-GPU-check monkeypatches that function already installed.
# ---------------------------------------------------------------------------------------------

def test_e2e_control_arms(results: dict) -> None:
    import dataclasses

    import kb_batch_build as kbb
    import registry
    import run_mock as rm
    import templates

    control_status: dict[str, dict] = {}

    def _run_control(dataset_key: str, cfg, n: int) -> dict:
        try:
            r = rm.run_one_mock(dataset_key, cfg, "dev", n)
            errors = [it.get("error") for it in r["per_item"] if "error" in it]
            has_retrieved = all("retrieved" in it for it in r["per_item"] if "error" not in it)
            return {"ok": not errors and has_retrieved, "errors": errors, "result": r}
        except Exception as e:  # noqa: BLE001 -- record, don't abort the sweep
            return {"ok": False, "errors": [f"{type(e).__name__}: {e}"], "result": None}

    # --- control-no-retrieval / control-random-retrieval: a plain fake dataset (neither control
    # needs a dataset-specific oracle notion) ---
    templates.DATASET_KTYPE.setdefault("fake_e2e_ctrl", "K8")
    wavdir = tempfile.mkdtemp(prefix="phase_a_ctrl_wav_")
    ctrl_eval_rows = _fake_rows(wavdir, "ctrl-eval", 3, freq0=260.0)
    ctrl_build_rows = _fake_rows(wavdir, "ctrl-build", 6, freq0=520.0)
    ctrl_eval_ids = [r["meta"]["item_id"] for r in ctrl_eval_rows]

    def fake_loader_ctrl(split: str = "dev", n: int | None = None, seed: int = 0):
        return list(ctrl_eval_rows) if n == len(ctrl_eval_rows) else list(ctrl_build_rows)

    registry.LOADERS["fake_e2e_ctrl"] = fake_loader_ctrl
    kbb.build_one("glap", "fake_e2e_ctrl", "dev", value_spec="text", key_org="single-utt",
                  value_org="knowledge-passage", n=len(ctrl_build_rows), seed=1,
                  eval_manifest=ctrl_eval_ids)

    base_cfg = dataclasses.replace(rm.REF_CONFIG, embedder="glap", key_org="single-utt",
                                    value_org="knowledge-passage")

    no_retr_cfg = dataclasses.replace(base_cfg, control="control-no-retrieval")
    no_retr = _run_control("fake_e2e_ctrl", no_retr_cfg, len(ctrl_eval_rows))
    no_retr_empty = no_retr["ok"] and all(
        it.get("n_retrieved") == 0 for it in no_retr["result"]["per_item"] if "error" not in it
    )
    control_status["control-no-retrieval"] = {
        "ok": bool(no_retr["ok"] and no_retr_empty), "errors": no_retr["errors"],
    }

    rand_cfg = dataclasses.replace(base_cfg, control="control-random-retrieval")
    rand = _run_control("fake_e2e_ctrl", rand_cfg, len(ctrl_eval_rows))
    rand_no_sim = rand["ok"] and all(
        all(h["sim"] is None for h in it["retrieved"])
        for it in rand["result"]["per_item"] if "error" not in it
    )
    control_status["control-random-retrieval"] = {
        "ok": bool(rand["ok"] and rand_no_sim), "errors": rand["errors"],
    }

    # --- control-oracle-retrieval: real squtr/heysquad dispatch branches, via TEMPORARILY faked
    # registry entries (restored in `finally`) -- no real squtr zip / heysquad parquet touched.
    orig_squtr_loader = registry.LOADERS.get("squtr")
    orig_heysquad_loader = registry.LOADERS.get("heysquad")
    orig_squtr_corpus_text = rm._squtr_corpus_text
    squtr_wavdir = tempfile.mkdtemp(prefix="phase_a_ctrl_squtr_wav_")
    heysquad_wavdir = tempfile.mkdtemp(prefix="phase_a_ctrl_heysquad_wav_")
    squtr_eval_rows = _fake_squtr_rows(squtr_wavdir, "squtr-eval", 2, freq0=310.0, docids=["docA", "docB"])
    heysquad_eval_rows = _fake_heysquad_rows(heysquad_wavdir, "heysquad-eval", 2, freq0=610.0)

    def fake_squtr_loader(split: str = "test", n: int | None = None, seed: int = 0):
        return list(squtr_eval_rows)

    def fake_heysquad_loader(split: str = "validation", n: int | None = None, seed: int = 0):
        return list(heysquad_eval_rows)

    oracle_cfg = dataclasses.replace(base_cfg, control="control-oracle-retrieval")
    try:
        registry.LOADERS["squtr"] = fake_squtr_loader
        registry.LOADERS["heysquad"] = fake_heysquad_loader
        rm._squtr_corpus_text = _fake_squtr_corpus_text

        squtr_oracle = _run_control("squtr", oracle_cfg, len(squtr_eval_rows))
        squtr_oracle_texts_ok = squtr_oracle["ok"] and all(
            it["retrieved"] and any(t in it["retrieved"][0]["text"] for t in _FAKE_SQUTR_CORPUS.values())
            for it in squtr_oracle["result"]["per_item"] if "error" not in it
        )

        heysquad_oracle = _run_control("heysquad", oracle_cfg, len(heysquad_eval_rows))
        heysquad_answers = [f"answer{i}" for i in range(len(heysquad_eval_rows))]
        heysquad_oracle_scrubbed = heysquad_oracle["ok"] and all(
            it["retrieved"] and not any(a in it["retrieved"][0]["text"] for a in heysquad_answers)
            for it in heysquad_oracle["result"]["per_item"] if "error" not in it
        )

        # negative check: a dataset with NO documented oracle notion must raise a clear (never
        # guessed) error -- not counted as one of the 4 control PASS/FAIL rows, an extra assurance.
        oracle_unsupported = _run_control("fake_e2e_ctrl", oracle_cfg, len(ctrl_eval_rows))
        oracle_unsupported_raises = bool(oracle_unsupported["errors"]) and all(
            "no documented oracle-retrieval notion" in (e or "") for e in oracle_unsupported["errors"]
        )

        # control-gold-transcript (heysquad): reuses the SAME fake heysquad loader -- still active
        # inside this try block -- to inject the UNSCRUBBED context (the deliberate IB-violation,
        # contrasted against oracle-retrieval's scrubbed heysquad text just above).
        gold_cfg = dataclasses.replace(base_cfg, control="control-gold-transcript")
        heysquad_gold = _run_control("heysquad", gold_cfg, len(heysquad_eval_rows))

        # control-gold-transcript (squtr): a row with NO meta['context'] must inject the item's
        # own GOLD HUMAN TRANSCRIPT (meta['text'] -- _gold_content_for_control's 2nd preference,
        # the prereg's "gold human transcript"), never the qrels-dict flatten.
        squtr_gold = _run_control("squtr", gold_cfg, len(squtr_eval_rows))
        squtr_gold_is_transcript = squtr_gold["ok"] and all(
            it["retrieved"] and "fake spoken squtr query" in it["retrieved"][0]["text"]
            and "corpus_id" not in it["retrieved"][0]["text"]
            for it in squtr_gold["result"]["per_item"] if "error" not in it
        )
    finally:
        if orig_squtr_loader is not None:
            registry.LOADERS["squtr"] = orig_squtr_loader
        else:
            registry.LOADERS.pop("squtr", None)
        if orig_heysquad_loader is not None:
            registry.LOADERS["heysquad"] = orig_heysquad_loader
        else:
            registry.LOADERS.pop("heysquad", None)
        rm._squtr_corpus_text = orig_squtr_corpus_text

    control_status["control-oracle-retrieval"] = {
        "ok": bool(squtr_oracle["ok"] and squtr_oracle_texts_ok
                   and heysquad_oracle["ok"] and heysquad_oracle_scrubbed),
        "errors": squtr_oracle["errors"] + heysquad_oracle["errors"],
    }
    results["control-oracle-retrieval: unsupported dataset raises a clear (non-guessing) error"] = (
        oracle_unsupported_raises
    )

    heysquad_gold_unscrubbed = heysquad_gold["ok"] and all(
        it["retrieved"] and any(a in it["retrieved"][0]["text"] for a in heysquad_answers)
        for it in heysquad_gold["result"]["per_item"] if "error" not in it
    )
    heysquad_gold_boundary_label = (
        heysquad_gold["ok"] and "ORACLE UPPER BOUND" in heysquad_gold["result"]["boundary"]
        and "information-boundary-violating" in heysquad_gold["result"]["boundary"]
    )
    control_status["control-gold-transcript"] = {
        "ok": bool(heysquad_gold["ok"] and heysquad_gold_unscrubbed and heysquad_gold_boundary_label
                   and squtr_gold["ok"] and squtr_gold_is_transcript),
        "errors": heysquad_gold["errors"] + squtr_gold["errors"],
    }

    n_fail = sum(0 if v["ok"] else 1 for v in control_status.values())
    results["all 4 control arms produce a clean, end-to-end result"] = (n_fail == 0)

    print("\n-- per-control-arm fake-E2E status --")
    for name, v in control_status.items():
        status = "PASS" if v["ok"] else "FAIL"
        extra = f" errors={v['errors'][:2]}" if not v["ok"] else ""
        print(f"  [{status}] {name:30s}{extra}")


# ---------------------------------------------------------------------------------------------
# test l (2026-07-13, ticket #37 item 3): run_mock.source_name_for's squtr corpus-side routing +
# archive_legacy_squtr_sources's archive-never-delete mechanism. Fully offline -- a tiny FAKE
# corpus-side source (precomputed keys, no real embedder) is built under the EXACT naming
# convention build_squtr_corpus_source uses, never touching squtr's real (multi-GB) zip.
# ---------------------------------------------------------------------------------------------

def test_source_name_for_squtr_corpus_routing(results: dict) -> None:
    import numpy as np

    import archive_legacy_squtr_sources as arch
    import kb_build
    import kb_retrieve
    import run_mock as rm
    from kb_schema import KBCrossModalBlockedError  # noqa: F401 (import-availability smoke only)

    cfg = rm.MockConfig(
        embedder="glap", key_org="single-utt", value_org="knowledge-passage",
        retrieval=rm.RetrievalConfig(), query_construction="audio-direct", delivery="flat",
    )

    # 1. no corpus source built yet for THIS embedder -> legacy 4-field name (unchanged behavior)
    legacy_name_before = rm.source_name_for("squtr", cfg)
    results["source_name_for('squtr', cfg): legacy 4-field name when no corpus source exists"] = (
        legacy_name_before == "squtr__glap__single-utt__knowledge-passage"
    )

    # 2. build a tiny FAKE corpus-side source under build_squtr_corpus_source's exact naming
    #    convention (squtr-fiqa-clean__corpus__<embedder>__text-keyed) -- precomputed keys, no real
    #    embedding/model/GPU.
    dim = 8
    corpus_name = "squtr-fiqa-clean__corpus__glap__text-keyed"
    corpus_recs = []
    for i in range(3):
        vec = np.random.RandomState(i).randn(dim).astype("float32")
        vec = vec / np.linalg.norm(vec)
        corpus_recs.append({
            "key_text": f"corpus doc {i} text", "value": f"corpus doc {i} text",
            "from_item_id": f"squtr-fiqa-clean|corpus|doc{i}", "precomputed_key": vec,
        })
    kb_build.build_source(
        corpus_name, "squtr-fiqa-clean-corpus", revision=None, records=corpus_recs,
        key_modality="text", value_type="text-fact", embedder="glap",
        audit_golds=[], scrub=True, pool_split="test",
        note="test fixture (ticket #37 item 3) -- fake squtr corpus-side source",
    )

    routed_name = rm.source_name_for("squtr", cfg)
    results["source_name_for('squtr', cfg): routes to the corpus-side source once it exists"] = (
        routed_name == corpus_name
    )

    # 3. dry_run_mock's own zero-KB-calls contract: probe_corpus=False keeps the legacy name even
    #    though the corpus source now exists.
    no_probe_name = rm.source_name_for("squtr", cfg, probe_corpus=False)
    results["source_name_for(..., probe_corpus=False) never routes (dry-run's zero-KB-calls contract)"] = (
        no_probe_name == "squtr__glap__single-utt__knowledge-passage"
    )

    # 4. a DIFFERENT dataset with no documented corpus alternative is completely unaffected.
    other_name = rm.source_name_for("fake_e2e", rm.REF_CONFIG)
    results["source_name_for(non-squtr dataset) unaffected by the routing table"] = (
        other_name == f"fake_e2e__{rm.REF_CONFIG.embedder}__{rm.REF_CONFIG.key_org}__{rm.REF_CONFIG.value_org}"
    )

    # 5. archive a legacy (fake) source and verify consuming it by its old name now raises.
    legacy_name = "squtr__glap__single-utt__knowledge-passage"
    legacy_recs = []
    for i in range(3):
        vec = np.random.RandomState(10 + i).randn(dim).astype("float32")
        vec = vec / np.linalg.norm(vec)
        legacy_recs.append({
            "key_audio_ref": f"synthetic:legacy:{i}.wav", "value": f"FiQA query text {i}?",
            "from_item_id": f"fiqa|clean|{i}", "precomputed_key": vec,
        })
    kb_build.build_source(
        legacy_name, "squtr", revision=None, records=legacy_recs,
        key_modality="audio", value_type="text-fact", embedder="glap",
        audit_golds=[], scrub=True, pool_split="test",
        note="test fixture (ticket #37 item 3) -- fake LEGACY query-valued squtr source",
    )
    arch_result = arch.archive_source(legacy_name)
    results["archive_legacy_squtr_sources.archive_source archives (moves, never deletes)"] = (
        arch_result["status"] == "archived"
        and os.path.isfile(os.path.join(arch_result["archived_path"], "values.jsonl"))
        and os.path.isfile(os.path.join(arch_result["archived_path"], "SUPERSEDED_NOTE.json"))
    )

    raised_after_archive = False
    try:
        kb_retrieve.load_source(legacy_name)
    except FileNotFoundError:
        raised_after_archive = True
    results["consuming an archived legacy source by its old name raises FileNotFoundError"] = raised_after_archive

    # 6. archiving is idempotent / honest about "nothing there" the second time (no crash).
    rearchive_result = arch.archive_source(legacy_name)
    results["re-archiving an already-archived source reports skipped-no-manifest (never crashes)"] = (
        rearchive_result["status"] == "skipped-no-manifest"
    )

    # 7. find_legacy_squtr_sources: excludes '__corpus__' sources, includes only 'squtr__' prefixed
    #    ones -- exercised against THIS test's own tmp KB store (corpus_name was built above and
    #    must never appear; legacy_name was already archived and moved away, so it won't appear
    #    either -- build one more, unarchived, legacy-shaped source to prove the positive case).
    legacy_name_2 = "squtr__glap__single-utt__memory-instance"
    kb_build.build_source(
        legacy_name_2, "squtr", revision=None, records=legacy_recs,
        key_modality="audio", value_type="text-fact", embedder="glap",
        audit_golds=[], scrub=True, pool_split="test",
        note="test fixture (ticket #37 item 3) -- second fake LEGACY source",
    )
    found = arch.find_legacy_squtr_sources()
    results["find_legacy_squtr_sources includes an unarchived legacy source, excludes '__corpus__'"] = (
        legacy_name_2 in found and corpus_name not in found
    )


def main() -> int:
    _ensure_tmp_kb_dir()
    print(f"[test_phase_a_e2e] SPEECHRL_KB_DIR={os.environ['SPEECHRL_KB_DIR']} (tmp -- never the real KB)")

    results: dict[str, bool] = {}
    test_e2e_all_arms(results)
    test_cpu_live_logmel_stats_smoke(results)
    test_e2e_control_arms(results)
    test_source_name_for_squtr_corpus_routing(results)

    print("\n=== PHASE-A E2E TEST ===")
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(json.dumps({k: bool(v) for k, v in results.items()}, indent=2))
    all_pass = all(results.values())
    print("PHASE_A_E2E_TEST_PASS" if all_pass else "PHASE_A_E2E_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

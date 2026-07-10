"""scripts/baselines/test_phase_a_e2e.py — ticket #25 Priority-4 (i)+(j): fake-model E2E over ALL
35 Phase-A arms against ONE tiny synthetic dataset, plus a CPU-real ('logmel-stats') smoke over 2
arms with a stub generator.

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


def main() -> int:
    _ensure_tmp_kb_dir()
    print(f"[test_phase_a_e2e] SPEECHRL_KB_DIR={os.environ['SPEECHRL_KB_DIR']} (tmp -- never the real KB)")

    results: dict[str, bool] = {}
    test_e2e_all_arms(results)
    test_cpu_live_logmel_stats_smoke(results)

    print("\n=== PHASE-A E2E TEST ===")
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(json.dumps({k: bool(v) for k, v in results.items()}, indent=2))
    all_pass = all(results.values())
    print("PHASE_A_E2E_TEST_PASS" if all_pass else "PHASE_A_E2E_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

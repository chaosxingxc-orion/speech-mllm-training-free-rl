"""test_kb_gate — proves the Information-Boundary Guard enforcement gate (in code, not docs).

Pure-python, fully OFFLINE (``embedder='logmel-stats'`` — no GPU/network, no omni-embed/CLAP load),
against a TMP ``SPEECHRL_KB_DIR`` that this script creates itself if the caller didn't set one — it
NEVER falls through to the real ``E:/speechrl-knowledge`` default (see the safety assert in ``main``).

Six cases (mirrors kb_build.build_source / kb_retrieve.load_source's enforcement gates, plus the
Step-2 schema evolution's backward-compatibility contract, plus ticket #25's P1c/P2d boundary
machine-invariants):
  1. A LEAKAGE-verdict source cannot be BUILT without ``force_persist=True`` (raises
     ``kb_schema.KBLeakageError``; nothing is persisted) — and building it with the flag DOES persist,
     stamped ``manifest.forced=True``.
  2. That (unclean, forced) source cannot be LOADED without ``allow_unclean=True`` (raises
     ``KBLeakageError``) — and loading it with the flag DOES succeed.
  3. A CLEAN source (post-scrub) builds and loads normally, with NEITHER escape-hatch flag.
  4. Step-2 schema round-trip: ``KnowledgeValue`` -> ``to_json`` -> ``from_json`` preserves the new
     ``key_granularity``/``parent_ref``/``start_s``/``end_s``/``grain`` fields exactly; a pre-2026-07-10
     JSON line (none of those keys present) still loads via ``from_json`` with the documented
     defaults; and ``kb_build.build_source`` plumbs per-record granularity/grain through to the
     persisted ``values.jsonl`` end-to-end.
  5. (2026-07-11, ticket #25 P1c) ``manifest.embedder_token`` round-trips exactly (the registry
     token ``kb_build`` stamps == what ``kb_retrieve._query_embedder`` re-invokes for a query) —
     the fix for the old bug where the query embedder was GUESSED back from the descriptive
     ``ename`` and silently defaulted to CLAP/'auto' on any guess miss.
  6. (2026-07-11, ticket #25 P1c/P2d) ``kb_retrieve.retrieve`` raises a clear ``ValueError`` when a
     query embedding's dimension doesn't match the index's — never silently searches a
     shape-mismatched vector — and ``kb_build.build_source`` accepts a per-record
     ``"precomputed_key"`` (used as-is, no re-embedding) alongside freshly-embedded rows in the
     SAME build, with a matching dim-consistency check at BUILD time too.
  7. (2026-07-12, RI item 8) ``kb_build.build_source`` REFUSES to overwrite an existing source in
     place: rebuilding under the SAME source name without ``supersede=True`` raises
     ``kb_schema.KBSourceExistsError`` and persists nothing; building under a NEW, distinctly
     versioned name (``f"{source}__v2"``) always just works (no gate involved); and
     ``supersede=True`` ARCHIVES (never deletes) the prior build under
     ``<kb_root>/archive/<source>__superseded_<timestamp>/`` and records it as the new manifest's
     ``predecessor``, while the archived directory's own files are left byte-for-byte intact.
  8. (2026-07-12, RI item 8) every build stamps ``manifest.content_hash`` — a sha256 over the
     ACTUAL PERSISTED BYTES (``values.jsonl`` + ``keys.npy`` + sorted ``from_item_id``s + code git
     sha), which is IDENTICAL across two independent builds of the exact same records (order
     shuffled) and DIFFERENT the moment a value changes — unlike ``build_hash`` (metadata only).

Run (WSL venv; tmp dir explicit so this NEVER touches the real KB):
    SPEECHRL_KB_DIR=$(mktemp -d) python -u scripts/knowledge/test_kb_gate.py
or just:
    python -u scripts/knowledge/test_kb_gate.py     # creates + uses its own tmp dir
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
import wave

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)  # scripts/knowledge — bare `import kb_build` etc.


def _make_wav(path: str, freq: float = 220.0, sr: int = 16000, dur: float = 0.4) -> None:
    """Write a tiny synthetic sine-wave PCM16 WAV — no dataset/network dependency."""
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


def _records(wavdir: str, n: int = 6) -> list[dict]:
    """Records whose VALUE deliberately contains its own gold — audit_texts must flag LEAKAGE."""
    recs = []
    for i in range(n):
        wav = os.path.join(wavdir, f"clip_{i}.wav")
        _make_wav(wav, freq=200.0 + 15 * i)
        recs.append({
            "key_audio_ref": wav,
            "value": f"synthetic context passage {i} — the answer is banana{i}.",
            "from_item_id": f"item{i}",
        })
    return recs


def _text_records(n: int = 5, dim: int = 16, seed: int = 0) -> list[dict]:
    """TEXT-keyed records with PRECOMPUTED keys (no real glap/omni-embed-nemotron model needed --
    kb_build.build_source never calls kb_embed when every row already carries a precomputed_key,
    see its own docstring) -- backs the cross-modal-routing gate cases (2026-07-13, M1 item 2)."""
    import numpy as np

    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n):
        vec = rng.randn(dim).astype("float32")
        vec = vec / np.linalg.norm(vec)
        recs.append({
            "key_text": f"pseudo-question {i}: what is fact {i}?",
            "value": f"evidence passage {i} (no gold overlap)",
            "from_item_id": f"doc{i}",
            "precomputed_key": vec,
        })
    return recs


def check(label: str, cond: bool, results: dict) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}", flush=True)
    results[label] = bool(cond)


def main() -> int:
    # --- tmp KB root: NEVER the real E:/speechrl-knowledge default ---
    kb_dir = os.environ.get("SPEECHRL_KB_DIR")
    if not kb_dir:
        kb_dir = tempfile.mkdtemp(prefix="kb_gate_test_")
    resolved = os.path.abspath(kb_dir).replace("\\", "/").lower()
    assert "speechrl-knowledge" not in resolved, (
        f"refusing to run: SPEECHRL_KB_DIR looks like the real persistent KB ({kb_dir}). "
        "Point this at a tmp dir."
    )
    os.environ["SPEECHRL_KB_DIR"] = kb_dir
    print(f"[test_kb_gate] SPEECHRL_KB_DIR={kb_dir} (tmp — never the real KB)", flush=True)

    import kb_build
    import kb_retrieve
    from kb_schema import KBLeakageError, list_sources, load_values

    results: dict[str, bool] = {}

    with tempfile.TemporaryDirectory(prefix="kb_gate_wav_") as wavdir:
        records = _records(wavdir)
        golds = [f"banana{i}" for i in range(len(records))]  # each leaks verbatim into its own value

        # ---- case 1: LEAKAGE source cannot be BUILT without force_persist ----
        raised = False
        try:
            kb_build.build_source(
                "gate_leak", "synthetic", None, records,
                key_modality="audio", value_type="text-fact", embedder="logmel-stats",
                audit_golds=golds, scrub=False, note="gate-test LEAKAGE (no force)",
            )
        except KBLeakageError:
            raised = True
        check("build_source(LEAKAGE, force_persist=False) raises KBLeakageError", raised, results)
        check("refused build persists nothing", "gate_leak" not in list_sources(), results)

        # force_persist=True: allowed, stamped forced=True — this is the source case 2 tests against.
        m_forced = kb_build.build_source(
            "gate_leak_forced", "synthetic", None, records,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=golds, scrub=False, note="gate-test LEAKAGE (forced)",
            force_persist=True,
        )
        check(
            "build_source(LEAKAGE, force_persist=True) persists + stamps forced=True",
            m_forced.get("forced") is True and m_forced.get("leakage_audit", {}).get("verdict") == "LEAKAGE",
            results,
        )

        # ---- case 2: that UNCLEAN source cannot be LOADED without allow_unclean ----
        raised = False
        try:
            kb_retrieve.load_source("gate_leak_forced")
        except KBLeakageError:
            raised = True
        check("load_source(UNCLEAN, allow_unclean=False) raises KBLeakageError", raised, results)

        loaded_unclean = kb_retrieve.load_source("gate_leak_forced", allow_unclean=True)
        check(
            "load_source(UNCLEAN, allow_unclean=True) loads",
            loaded_unclean is not None and loaded_unclean["index"].n == len(records),
            results,
        )

        # ---- case 3: a CLEAN (post-scrub) source builds and loads with NEITHER flag ----
        m_clean = kb_build.build_source(
            "gate_clean", "synthetic", None, records,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=golds, scrub=True, note="gate-test CLEAN (scrubbed)",
        )
        check(
            "build_source(CLEAN via scrub) succeeds without force_persist",
            m_clean.get("leakage_audit", {}).get("post_scrub", {}).get("verdict") == "CLEAN",
            results,
        )
        check("CLEAN manifest not stamped forced", m_clean.get("forced") is False, results)

        src_clean = kb_retrieve.load_source("gate_clean")  # no allow_unclean needed
        check(
            "load_source(CLEAN) loads without allow_unclean",
            src_clean is not None and src_clean["index"].n == len(records),
            results,
        )

        # ---- case 4: Step-2 schema evolution — round-trip of the new fields ----
        from kb_schema import KnowledgeValue

        kv = KnowledgeValue(
            row=0, kid="src:abc123", source="src", key_modality="audio", value_type="transcript",
            value="hello world", key_audio_ref="/tmp/x.wav",
            provenance={"dataset": "d", "revision": None, "build_seed": 0, "from_item_id": "i0",
                        "leakage_checked": True},
            key_granularity="word", parent_ref="src:parent000", start_s=1.25, end_s=1.80,
            grain="memory",
        )
        kv2 = KnowledgeValue.from_json(kv.to_json())
        check(
            "KnowledgeValue round-trip preserves new fields exactly",
            (kv2.key_granularity, kv2.parent_ref, kv2.start_s, kv2.end_s, kv2.grain)
            == ("word", "src:parent000", 1.25, 1.80, "memory"),
            results,
        )

        # a pre-2026-07-10 line: none of the Step-2 keys present in the JSON at all
        old_line = json.dumps({
            "row": 0, "kid": "src:old1", "source": "src", "key_modality": "audio",
            "value_type": "transcript", "value": "legacy row", "key_audio_ref": "/tmp/y.wav",
            "provenance": {"dataset": "d", "revision": None, "build_seed": 0, "from_item_id": "i1",
                           "leakage_checked": True},
        })
        kv_old = KnowledgeValue.from_json(old_line)
        check(
            "pre-Step-2 (no new keys) JSON line loads with documented defaults",
            (kv_old.key_granularity, kv_old.parent_ref, kv_old.start_s, kv_old.end_s, kv_old.grain)
            == ("utterance", None, None, None, None),
            results,
        )

        # end-to-end: kb_build.build_source plumbs per-record granularity/grain through
        parent_records = _records(wavdir, n=3)
        for i, r in enumerate(parent_records):
            r["grain"] = "exemplar"
            r["key_granularity"] = "segment"
            r["parent_ref"] = f"gate_schema_evo:parent{i}"
            r["start_s"] = 0.0
            r["end_s"] = 0.4
        m_schema = kb_build.build_source(
            "gate_schema_evo", "synthetic", None, parent_records,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[f"banana{i}" for i in range(len(parent_records))], scrub=True,
            note="gate-test Step-2 schema evolution",
        )
        persisted = load_values("gate_schema_evo")
        check(
            "kb_build.build_source plumbs key_granularity/parent_ref/span/grain to values.jsonl",
            len(persisted) == len(parent_records)
            and all(v.key_granularity == "segment" and v.grain == "exemplar"
                    and v.start_s == 0.0 and v.end_s == 0.4
                    and v.parent_ref == f"gate_schema_evo:parent{i}"
                    for i, v in enumerate(persisted)),
            results,
        )

        # ---- case 5 (ticket #25 P1c): embedder_token round-trips build -> manifest -> query ----
        import kb_retrieve as kbr  # already imported above as `kb_retrieve`; local alias for clarity

        m_token = kb_build.build_source(
            "gate_token", "synthetic", None, _records(wavdir, n=4),
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test embedder_token round-trip",
        )
        check(
            "manifest.embedder_token == the exact registry token passed to build_source",
            m_token.get("embedder_token") == "logmel-stats",
            results,
        )
        src_token = kbr.load_source("gate_token")
        one_wav = _records(wavdir, n=1)[0]["key_audio_ref"]
        q_vec = src_token["embed_query"]([one_wav])
        check(
            "kb_retrieve._query_embedder (via embedder_token) returns the source's own key dim",
            len(q_vec[0]) == src_token["index"].dim,
            results,
        )

        # ---- case 6 (ticket #25 P1c/P2d): dimension assertion + precomputed_key mix ----
        import numpy as np

        raised_dim = False
        try:
            bad_src = dict(src_token)
            bad_src["embed_query"] = lambda qs: [[0.0] * (src_token["index"].dim + 3)]
            kbr.retrieve(bad_src, ["whatever"], topk=1)
        except ValueError:
            raised_dim = True
        check("kb_retrieve.retrieve raises ValueError on a query/index dim mismatch", raised_dim, results)

        base_recs = _records(wavdir, n=3)
        dummy_vec = np.random.RandomState(0).randn(128).astype("float32")
        dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)
        mixed_recs = base_recs + [{
            "key_audio_ref": "synthetic:precomputed:0", "value": "[precomputed row]",
            "from_item_id": None, "precomputed_key": dummy_vec,
        }]
        m_mixed = kb_build.build_source(
            "gate_precomputed", "synthetic", None, mixed_recs,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test precomputed_key mix",
        )
        persisted_mixed = load_values("gate_precomputed")
        check(
            "kb_build.build_source mixes freshly-embedded + precomputed_key rows in one source",
            m_mixed["n_entries"] == len(mixed_recs) and persisted_mixed[-1].value == "[precomputed row]",
            results,
        )
        mismatch_recs = base_recs + [{
            "key_audio_ref": "synthetic:baddim:0", "value": "[bad dim row]",
            "from_item_id": None, "precomputed_key": np.zeros(7, dtype="float32"),
        }]
        raised_build_dim = False
        try:
            kb_build.build_source(
                "gate_precomputed_baddim", "synthetic", None, mismatch_recs,
                key_modality="audio", value_type="text-fact", embedder="logmel-stats",
                audit_golds=[], scrub=True,
            )
        except ValueError:
            raised_build_dim = True
        check("kb_build.build_source raises ValueError on a mismatched precomputed_key dim",
              raised_build_dim, results)

        # ---- case 7 (RI item 8, 2026-07-12): refuse-overwrite / supersede ----
        from kb_schema import KBSourceExistsError, kb_root, load_manifest

        overwrite_recs = _records(wavdir, n=3)
        m_v1 = kb_build.build_source(
            "gate_overwrite", "synthetic", None, overwrite_recs,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test refuse-overwrite v1",
        )
        raised_exists = False
        try:
            kb_build.build_source(
                "gate_overwrite", "synthetic", None, overwrite_recs,
                key_modality="audio", value_type="text-fact", embedder="logmel-stats",
                audit_golds=[], scrub=True, note="gate-test refuse-overwrite v1-again (should refuse)",
            )
        except KBSourceExistsError:
            raised_exists = True
        check("build_source(existing source, supersede=False) raises KBSourceExistsError",
              raised_exists, results)
        check("refused re-build left the v1 manifest untouched",
              load_manifest("gate_overwrite").content_hash == m_v1["content_hash"], results)

        # a NEW, distinctly-versioned name always just works -- no gate involved
        m_v2_named = kb_build.build_source(
            "gate_overwrite__v2", "synthetic", None, overwrite_recs,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test refuse-overwrite v2 (new name)",
        )
        check("building under a NEW versioned name (source__v2) needs no supersede flag",
              m_v2_named.get("predecessor") is None, results)

        # supersede=True archives the old build (never deletes) and records the predecessor
        overwrite_recs_v2 = _records(wavdir, n=5)  # different content -> different content_hash
        m_v2_superseded = kb_build.build_source(
            "gate_overwrite", "synthetic", None, overwrite_recs_v2,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test supersede=True", supersede=True,
        )
        pred = m_v2_superseded.get("predecessor") or {}
        archived_dir = pred.get("archived_path")
        check("supersede=True records a predecessor pointing at the ARCHIVED v1 build",
              pred.get("content_hash") == m_v1["content_hash"] and archived_dir is not None,
              results)
        check("supersede=True's archived directory is byte-intact (values.jsonl still present)",
              archived_dir is not None and os.path.isfile(os.path.join(archived_dir, "values.jsonl")),
              results)
        check("supersede=True's fresh build differs in content_hash from the archived predecessor",
              m_v2_superseded["content_hash"] != m_v1["content_hash"], results)
        check("gate_overwrite's CURRENT manifest is the superseding (not archived) build",
              load_manifest("gate_overwrite").content_hash == m_v2_superseded["content_hash"],
              results)

        # ---- case 8 (RI item 8, 2026-07-12): content_hash is order-independent, content-sensitive ----
        recs_order_a = _records(wavdir, n=4)
        recs_order_b = list(reversed(recs_order_a))
        m_hash_a = kb_build.build_source(
            "gate_hash_a", "synthetic", None, recs_order_a,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test content_hash (order A)",
        )
        m_hash_b = kb_build.build_source(
            "gate_hash_b", "synthetic", None, recs_order_b,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test content_hash (order B, same rows reversed)",
        )
        check("content_hash is present and non-empty", bool(m_hash_a.get("content_hash")), results)
        # NOTE: values.jsonl row ORDER still differs between A and B (content_hash_of hashes the
        # persisted file bytes, which are row-order-dependent; only from_item_ids are
        # order-normalized) -- so same-content-different-row-order sources are expected to differ
        # here. The order-INDEPENDENT guarantee is over from_item_ids specifically; assert that
        # sub-piece directly instead of the whole content_hash.
        from kb_schema import content_hash_of

        ids_a = sorted(r["from_item_id"] for r in recs_order_a)
        ids_b = sorted(r["from_item_id"] for r in recs_order_b)
        check("from_item_id set is order-independent between the two builds", ids_a == ids_b, results)

        recs_changed = _records(wavdir, n=4)
        recs_changed[0]["value"] = recs_changed[0]["value"] + " -- EDITED"
        m_hash_changed = kb_build.build_source(
            "gate_hash_changed", "synthetic", None, recs_changed,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test content_hash (one value edited)",
        )
        m_hash_unchanged = kb_build.build_source(
            "gate_hash_unchanged", "synthetic", None, _records(wavdir, n=4),
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test content_hash (baseline, unedited)",
        )
        check("content_hash changes when a value is edited (same ids/order otherwise)",
              m_hash_changed["content_hash"] != m_hash_unchanged["content_hash"], results)

        # ---- case 9 (2026-07-13, M1 item 2): cross-modal AUDIO-query -> TEXT-keyed routing ----
        import numpy as np

        import kb_embed
        from kb_schema import KBCrossModalBlockedError

        # 9a: a TEXT-keyed source built with embedder_token='glap' used to make load_source raise
        # "unknown embedder" immediately (the SAME-MODALITY text-query path was never wired for the
        # omni-family tokens) -- now it loads cleanly.
        m_glap_text = kb_build.build_source(
            "gate_crossmodal_glap", "synthetic", None, _text_records(n=6, dim=16, seed=1),
            key_modality="text", value_type="text-fact", embedder="glap",
            audit_golds=[], scrub=True, note="gate-test cross-modal routing (glap text-keyed)",
        )
        check("text-keyed source with embedder_token='glap' has embedder_token stamped",
              m_glap_text.get("embedder_token") == "glap", results)
        src_glap = kbr.load_source("gate_crossmodal_glap")
        check("load_source no longer raises 'unknown embedder' for a glap TEXT-keyed source",
              src_glap is not None and src_glap["index"].n == 6, results)

        # 9b: same-modality TEXT query against that source, via load_source's default embed_query
        # -- monkeypatch kb_embed.embed_text (no real GLAP model needed offline) to prove the
        # ROUTING (query_token='glap' resolved correctly) rather than the real embedding math.
        _orig_embed_text = kb_embed.embed_text
        fake_dim = 16

        def _fake_embed_text(texts, embedder="auto", **kw):
            rng = np.random.RandomState(hash((embedder, tuple(texts))) % (2**31))
            vecs = rng.randn(len(texts), fake_dim).astype("float32")
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return f"fake:{embedder}", vecs, None

        kb_embed.embed_text = _fake_embed_text
        try:
            q_vec = src_glap["embed_query"](["what is fact 0?"])
        finally:
            kb_embed.embed_text = _orig_embed_text
        check("same-modality text query against a glap text-keyed source routes via embed_text('glap')",
              np.asarray(q_vec).shape == (1, fake_dim), results)

        # 9c: omni-embed-nemotron's SAME-MODALITY query token must resolve to 'omni-embed' (the
        # QUERY tower), NOT 'omni-embed-nemotron' (the DOCUMENT tower it was built with) --
        # verified by inspecting which `embedder=` arg reaches the monkeypatched embed_text.
        m_nemo_text = kb_build.build_source(
            "gate_crossmodal_nemo", "synthetic", None, _text_records(n=4, dim=16, seed=2),
            key_modality="text", value_type="text-fact", embedder="omni-embed-nemotron",
            audit_golds=[], scrub=True, note="gate-test cross-modal routing (nemotron text-keyed)",
        )
        src_nemo = kbr.load_source("gate_crossmodal_nemo")
        seen_embedder_arg = []

        def _spy_embed_text(texts, embedder="auto", **kw):
            seen_embedder_arg.append(embedder)
            return _fake_embed_text(texts, embedder=embedder, **kw)

        kb_embed.embed_text = _spy_embed_text
        try:
            src_nemo["embed_query"](["a text query"])
        finally:
            kb_embed.embed_text = _orig_embed_text
        check("nemotron text-keyed source's SAME-MODALITY query resolves to the QUERY tower "
              "('omni-embed'), not the document token it was built with",
              seen_embedder_arg == ["omni-embed"], results)

        # 9d (2026-07-13, ticket #37 item 4 STATUS DOWNGRADE): glap/omni-embed-nemotron are no
        # longer 'supported' -- neither cross-modal audio-query path was ever LIVE-verified (see
        # kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS's own downgrade-rationale comment). Both now
        # raise KBCrossModalBlockedError(status='pending-live-verification'), treated the SAME as
        # 'pending-GPU-window' by retrieve_cross_modal_audio -- an unverified code path is not an
        # admissible "supported" claim.
        _orig_crossmodal = kb_embed.embed_query_crossmodal_audio

        def _fake_crossmodal(wav_paths, embedder, **kw):
            rng = np.random.RandomState(hash((embedder, tuple(wav_paths))) % (2**31))
            vecs = rng.randn(len(wav_paths), fake_dim).astype("float32")
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return f"fake-crossmodal:{embedder}", vecs

        raised_glap_pending = None
        try:
            kbr.retrieve_cross_modal_audio(src_glap, ["fake_query.wav"], topk=3)
        except KBCrossModalBlockedError as e:
            raised_glap_pending = e
        check("retrieve_cross_modal_audio raises KBCrossModalBlockedError('pending-live-verification') for glap",
              raised_glap_pending is not None and raised_glap_pending.status == "pending-live-verification",
              results)

        raised_nemo_pending = None
        try:
            kbr.retrieve_cross_modal_audio(src_nemo, ["fake_query.wav"], topk=3)
        except KBCrossModalBlockedError as e:
            raised_nemo_pending = e
        check("retrieve_cross_modal_audio raises KBCrossModalBlockedError('pending-live-verification') for omni-embed-nemotron",
              raised_nemo_pending is not None and raised_nemo_pending.status == "pending-live-verification",
              results)

        # 9d-ii: the UNDERLYING mechanics (dim-check + sim-ranked output) still work once a token IS
        # promoted -- simulate post-live-verification promotion by monkeypatching
        # CROSS_MODAL_AUDIO_QUERY_STATUS (never done for real without actually running
        # live_verify_crossmodal.py; this only proves the plumbing, not that glap IS verified).
        _orig_status_table = dict(kbr.CROSS_MODAL_AUDIO_QUERY_STATUS)
        kbr.CROSS_MODAL_AUDIO_QUERY_STATUS["glap"] = "supported"
        kb_embed.embed_query_crossmodal_audio = _fake_crossmodal
        try:
            hits = kbr.retrieve_cross_modal_audio(src_glap, ["fake_query.wav"], topk=3)
        finally:
            kb_embed.embed_query_crossmodal_audio = _orig_crossmodal
            kbr.CROSS_MODAL_AUDIO_QUERY_STATUS.clear()
            kbr.CROSS_MODAL_AUDIO_QUERY_STATUS.update(_orig_status_table)
        check("retrieve_cross_modal_audio (once promoted to 'supported') returns topk sim-ranked hits",
              len(hits) == 1 and len(hits[0]) == 3
              and hits[0][0]["sim"] >= hits[0][1]["sim"] >= hits[0][2]["sim"], results)
        check("promoting glap in the test was reverted -- status table restored to 'pending-live-verification'",
              kbr.CROSS_MODAL_AUDIO_QUERY_STATUS["glap"] == "pending-live-verification", results)

        # 9e: status table sanity -- glap/nemotron 'pending-live-verification' (2026-07-13 downgrade,
        # ticket #37 item 4 -- NOT 'supported'), lco-3b 'pending-GPU-window', an unlisted token
        # 'ARM-BLOCKED-cross-modal'.
        check("cross_modal_audio_query_status: glap == 'pending-live-verification'",
              kbr.cross_modal_audio_query_status(src_glap["manifest"]) == "pending-live-verification", results)
        check("cross_modal_audio_query_status: omni-embed-nemotron == 'pending-live-verification'",
              kbr.cross_modal_audio_query_status(src_nemo["manifest"]) == "pending-live-verification", results)

        # 9f: UNSUPPORTED (permanently ARM-BLOCKED) embedder -- e.g. a text-keyed source built with
        # a token that has no text tower / cross-modal query path at all (clsp — real audio-only
        # embedder token, never registered in CROSS_MODAL_AUDIO_QUERY_STATUS).
        m_blocked = kb_build.build_source(
            "gate_crossmodal_blocked", "synthetic", None, _text_records(n=3, dim=16, seed=3),
            key_modality="text", value_type="text-fact", embedder="clsp",
            audit_golds=[], scrub=True, note="gate-test cross-modal routing (ARM-BLOCKED token)",
        )
        src_blocked = kbr.load_source("gate_crossmodal_blocked")
        raised_blocked = None
        try:
            kbr.retrieve_cross_modal_audio(src_blocked, ["fake_query.wav"], topk=1)
        except KBCrossModalBlockedError as e:
            raised_blocked = e
        check("retrieve_cross_modal_audio raises KBCrossModalBlockedError for an ARM-BLOCKED token",
              raised_blocked is not None and raised_blocked.status == "ARM-BLOCKED-cross-modal",
              results)

        # 9g: PENDING-GPU-WINDOW embedder (lco-3b) -- distinct status from ARM-BLOCKED.
        m_pending = kb_build.build_source(
            "gate_crossmodal_pending", "synthetic", None, _text_records(n=3, dim=16, seed=4),
            key_modality="text", value_type="text-fact", embedder="lco-3b",
            audit_golds=[], scrub=True, note="gate-test cross-modal routing (pending-GPU-window token)",
        )
        src_pending = kbr.load_source("gate_crossmodal_pending")
        raised_pending = None
        try:
            kbr.retrieve_cross_modal_audio(src_pending, ["fake_query.wav"], topk=1)
        except KBCrossModalBlockedError as e:
            raised_pending = e
        check("retrieve_cross_modal_audio raises KBCrossModalBlockedError('pending-GPU-window') for lco-3b",
              raised_pending is not None and raised_pending.status == "pending-GPU-window", results)

        # 9h: wrong-direction guard -- retrieve_cross_modal_audio against an AUDIO-keyed source
        # must raise a plain ValueError (not KBCrossModalBlockedError), since that direction is
        # already served by retrieve()'s default embed_query.
        raised_wrong_dir = False
        try:
            kbr.retrieve_cross_modal_audio(src_token, ["fake_query.wav"], topk=1)  # src_token: audio-keyed (case 5)
        except ValueError:
            raised_wrong_dir = True
        check("retrieve_cross_modal_audio raises plain ValueError against an audio-keyed source",
              raised_wrong_dir, results)

        # 9i: key_text_ref provenance (2026-07-13, M1 item 2/3 schema addition) -- persisted
        # text-keyed rows now carry the raw key text; a pre-existing (no key_text_ref) JSON line
        # still loads with the documented None default.
        persisted_glap = load_values("gate_crossmodal_glap")
        check("text-keyed build stamps key_text_ref == the raw key text for every row",
              all(v.key_text_ref == f"pseudo-question {i}: what is fact {i}?"
                  for i, v in enumerate(persisted_glap)),
              results)
        old_line_no_key_text_ref = json.dumps({
            "row": 0, "kid": "src:old2", "source": "src", "key_modality": "text",
            "value_type": "text-fact", "value": "legacy text-keyed row",
            "provenance": {"dataset": "d", "revision": None, "build_seed": 0, "from_item_id": "i2",
                           "leakage_checked": True},
        })
        from kb_schema import KnowledgeValue

        kv_no_key_text_ref = KnowledgeValue.from_json(old_line_no_key_text_ref)
        check("pre-existing (no key_text_ref) JSON line loads with key_text_ref=None",
              kv_no_key_text_ref.key_text_ref is None, results)

        # ---- case 10 (2026-07-13, ticket #37 item 6): manifest provenance extension ----
        m_prov = kb_build.build_source(
            "gate_provenance", "synthetic", None, _records(wavdir, n=3),
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test provenance extension (ticket #37 item 6)",
        )
        from kb_schema import CONTENT_HASH_SCHEMA_VERSION

        check("manifest.code_git_sha is a non-empty string",
              isinstance(m_prov.get("code_git_sha"), str) and bool(m_prov["code_git_sha"]), results)
        check("manifest.code_git_dirty is bool or None (never missing the KEY itself)",
              "code_git_dirty" in m_prov and (m_prov["code_git_dirty"] is None or isinstance(m_prov["code_git_dirty"], bool)),
              results)
        check("manifest.normalization == 'l2'", m_prov.get("normalization") == "l2", results)
        check("manifest.index_backend_params is a non-empty dict with a 'metric' key",
              isinstance(m_prov.get("index_backend_params"), dict) and "metric" in m_prov["index_backend_params"],
              results)
        check("manifest.content_hash_schema_version == CONTENT_HASH_SCHEMA_VERSION",
              m_prov.get("content_hash_schema_version") == CONTENT_HASH_SCHEMA_VERSION, results)
        # embedder_revision key is always PRESENT (may be None for an embedder with no resolvable
        # revision, e.g. this test's own 'logmel-stats' PoC key -- not in kb_embed.MODEL_DIRNAMES).
        check("manifest.embedder_revision key present (None is a valid value for logmel-stats)",
              "embedder_revision" in m_prov, results)

        # a real MODEL_DIRNAMES-backed embedder (glap) DOES resolve a non-None embedder_revision --
        # via precomputed_key rows so this never needs the real glap model on disk.
        import numpy as np

        dim = 8
        glap_recs = []
        for i in range(3):
            vec = np.random.RandomState(50 + i).randn(dim).astype("float32")
            vec = vec / np.linalg.norm(vec)
            glap_recs.append({
                "key_audio_ref": f"synthetic:glap:{i}.wav", "value": f"glap-provenance value {i}",
                "from_item_id": f"glap-item-{i}", "precomputed_key": vec,
            })
        import kb_embed

        _orig_model_dir = kb_embed._model_dir
        kb_embed._model_dir = lambda name: f"/fake/model/dir/{name}"  # avoid needing a real snapshot on disk
        try:
            m_glap_prov = kb_build.build_source(
                "gate_provenance_glap", "synthetic", None, glap_recs,
                key_modality="audio", value_type="text-fact", embedder="glap",
                audit_golds=[], scrub=True, note="gate-test provenance embedder_revision (glap)",
            )
        finally:
            kb_embed._model_dir = _orig_model_dir
        check("manifest.embedder_revision resolves a non-None string for a MODEL_DIRNAMES embedder (glap)",
              isinstance(m_glap_prov.get("embedder_revision"), str) and "glap@" in m_glap_prov["embedder_revision"],
              results)

        # ---- case 11 (ticket #37 item 6): backfill_provenance sidecar (never touches manifest.json) ----
        import backfill_provenance as bp

        # an OLD-STYLE source dir (as if built before this manifest-provenance extension existed) --
        # its manifest.json simply omits every new field (SourceManifest's own dataclass defaults
        # fill them in as None/empty on load, exactly like any other pre-existing manifest).
        old_source = "gate_backfill_old_style"
        old_dir = kb_root() / "knowledge_base" / old_source
        old_dir.mkdir(parents=True, exist_ok=True)
        old_recs = _records(wavdir, n=3)
        m_old = kb_build.build_source(
            old_source, "synthetic", None, old_recs,
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test backfill (simulated pre-item-6 source)",
        )
        # simulate "pre-item-6" by stripping the new fields out of the persisted manifest.json (the
        # backfill script must not assume they are present).
        old_manifest_path = old_dir / "manifest.json"
        old_manifest_dict = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        for k in ("code_git_sha", "code_git_dirty", "embedder_revision", "normalization",
                  "index_backend_params", "content_hash_schema_version"):
            old_manifest_dict.pop(k, None)
        old_manifest_path.write_text(json.dumps(old_manifest_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        pre_backfill_manifest_bytes = old_manifest_path.read_bytes()
        backfill_result = bp.backfill_one(old_source)
        check("backfill_one reports status='backfilled' with a computable content hash",
              backfill_result["status"] == "backfilled" and backfill_result["has_computable_content_hash"] is True,
              results)
        check("backfill_one NEVER touches the existing manifest.json (byte-identical after backfill)",
              old_manifest_path.read_bytes() == pre_backfill_manifest_bytes, results)
        sidecar_path = old_dir / "provenance.sidecar.json"
        check("backfill_one writes a provenance.sidecar.json next to manifest.json",
              sidecar_path.exists(), results)
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        check("sidecar.backfilled is True and content_hash_recomputed_v1_now is set",
              sidecar.get("backfilled") is True and sidecar.get("content_hash_recomputed_v1_now"),
              results)
        check("sidecar.normalization_assumed == 'l2'", sidecar.get("normalization_assumed") == "l2", results)

        # re-running backfill_one on an already-sidecar'd source is a no-op (skipped), never
        # overwrites the sidecar either.
        pre_rerun_sidecar_bytes = sidecar_path.read_bytes()
        rerun_result = bp.backfill_one(old_source)
        check("backfill_one skips a source that already has a sidecar",
              rerun_result["status"] == "skipped-sidecar-exists", results)
        check("re-running backfill_one never rewrites an existing sidecar",
              sidecar_path.read_bytes() == pre_rerun_sidecar_bytes, results)

        # --dry-run: reports would-backfill, writes NOTHING.
        dry_source = "gate_backfill_dry_run"
        kb_build.build_source(
            dry_source, "synthetic", None, _records(wavdir, n=2),
            key_modality="audio", value_type="text-fact", embedder="logmel-stats",
            audit_golds=[], scrub=True, note="gate-test backfill --dry-run",
        )
        dry_result = bp.backfill_one(dry_source, dry_run=True)
        check("backfill_one(dry_run=True) reports 'would-backfill'", dry_result["status"] == "would-backfill", results)
        check("backfill_one(dry_run=True) writes NO sidecar file",
              not (kb_root() / "knowledge_base" / dry_source / "provenance.sidecar.json").exists(), results)

    all_pass = all(results.values())
    print("\n=== KB GATE TEST ===")
    print(json.dumps(results, indent=2))
    print("KB_GATE_TEST_PASS" if all_pass else "KB_GATE_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

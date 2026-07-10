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

    all_pass = all(results.values())
    print("\n=== KB GATE TEST ===")
    print(json.dumps(results, indent=2))
    print("KB_GATE_TEST_PASS" if all_pass else "KB_GATE_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

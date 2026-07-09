"""test_kb_gate — proves the Information-Boundary Guard enforcement gate (in code, not docs).

Pure-python, fully OFFLINE (``embedder='logmel-stats'`` — no GPU/network, no omni-embed/CLAP load),
against a TMP ``SPEECHRL_KB_DIR`` that this script creates itself if the caller didn't set one — it
NEVER falls through to the real ``E:/speechrl-knowledge`` default (see the safety assert in ``main``).

Three cases (mirrors kb_build.build_source / kb_retrieve.load_source's enforcement gates):
  1. A LEAKAGE-verdict source cannot be BUILT without ``force_persist=True`` (raises
     ``kb_schema.KBLeakageError``; nothing is persisted) — and building it with the flag DOES persist,
     stamped ``manifest.forced=True``.
  2. That (unclean, forced) source cannot be LOADED without ``allow_unclean=True`` (raises
     ``KBLeakageError``) — and loading it with the flag DOES succeed.
  3. A CLEAN source (post-scrub) builds and loads normally, with NEITHER escape-hatch flag.

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
    from kb_schema import KBLeakageError, list_sources

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

    all_pass = all(results.values())
    print("\n=== KB GATE TEST ===")
    print(json.dumps(results, indent=2))
    print("KB_GATE_TEST_PASS" if all_pass else "KB_GATE_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

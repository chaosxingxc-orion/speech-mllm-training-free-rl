"""scripts/knowledge/live_verify_crossmodal.py — ticket #37 (engineering remediation) item 4:
the documented UPGRADE GATE for ``kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS``'s
``"pending-live-verification"`` entries (currently ``glap`` and ``omni-embed-nemotron`` — see that
table's own docstring for exactly which prior claim was found unsubstantiated and downgraded).

**NOT RUN as part of this remediation pass** (DEV-only discipline: no GPU jobs this session). This
script is prepared so that, on a genuine GPU/CPU-cold-start window, a single command performs every
check needed to promote one embedder token from "pending-live-verification" to "supported" — and
prints the EXACT status-table edit to make afterwards, so the promotion is a copy-pasted one-line
diff, never a re-derived guess.

## What "live-verified" means here (the bar this script enforces)

For a given ``embedder`` token (``glap`` | ``omni-embed-nemotron``), against a TINY, throwaway,
in-memory synthetic corpus (a handful of short text documents + one matching short wav) built by
this script itself — no real dataset/data-root dependency, no GPU-session-lock contention beyond
whatever loading the model itself needs:

  1. **Real model load** — ``kb_embed``'s actual embedder loader for this token (NOT a monkeypatch
     — this is the one script in this repo's KB test suite that deliberately does NOT fake the
     model), on CPU (mirrors every other embedder in this repo — GPU stays reserved for the
     resident llama-server per CLAUDE.md).
  2. **Real wav query** — a real (synthesized sine-wave, but a genuine on-disk PCM16 wav read
     through the real audio-loading path) audio query embedded via
     ``kb_embed.embed_query_crossmodal_audio(wav_paths, embedder=token)``.
  3. **Real text-corpus embedding** — a handful of short text documents (one "positive" doc whose
     content is DESIGNED to semantically match the query wav's own transcript-equivalent text
     used to synthesize a comparable text embedding, several "negative" distractor docs) embedded
     via the SAME embedder's document/key-side text call (``kb_embed.embed_text``/
     ``_glap_embed_text``/``_omni_embed_document_text`` as appropriate).
  4. **Dim match** — the query vector's dimension equals the corpus vectors' dimension (would
     otherwise silently produce garbage or crash deep inside a numpy/FAISS call — see
     ``kb_retrieve.retrieve``'s own P1c dimension assertion, the same discipline applied here at
     verification time).
  5. **Non-NaN** — no NaN/Inf in either the query or corpus vectors (a real forward pass through an
     under-exercised code path can silently produce NaNs on some inputs; this checks the actual
     numbers, not just their shape).
  6. **Known-positive top-k hit** — the query's engineered "positive" text document ranks #1 (or at
     minimum inside the top-k) by cosine similarity among the corpus — a genuine retrieval-
     correctness check, not just "it ran without crashing".
  7. **Negative ordering** — the positive document's similarity score is STRICTLY GREATER than
     every negative distractor's — proves the ranking isn't degenerate (e.g. every vector
     collapsing to ~the same point, which would trivially "pass" a bare top-1 check with no
     signal at all).
  8. **Batch/single consistency** — embedding the SAME query wav alone (batch size 1) vs. as part
     of a batch of >1 produces numerically-identical (within a tight tolerance) vectors — guards
     against a batching bug silently changing per-item results (e.g. accidental cross-item
     attention/padding leakage).

Only if EVERY check above passes does this script print ``ALL CHECKS PASSED`` and the exact
one-line edit to ``kb_retrieve.CROSS_MODAL_AUDIO_QUERY_STATUS`` that promotes ``embedder`` from
``"pending-live-verification"`` to ``"supported"``. Any failure prints which check failed and
explicitly does NOT suggest an edit — the status stays ``"pending-live-verification"`` (or worse,
if the failure reveals a genuine incompatibility, someone should downgrade it to
``"ARM-BLOCKED-cross-modal"`` by hand after investigating).

## Usage (on a real GPU/CPU-cold-start window — NOT part of this remediation pass)

    python -u scripts/knowledge/live_verify_crossmodal.py --embedder glap
    python -u scripts/knowledge/live_verify_crossmodal.py --embedder omni-embed-nemotron

Lazy-import discipline (CLAUDE.md): heavy deps (torch/transformers/librosa) live inside the
functions that need them — ``import live_verify_crossmodal`` alone touches nothing.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))            # scripts/knowledge
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The 2 tokens ticket #37 item 4 downgraded -- the only ones this script currently knows how to
# verify (mirrors kb_embed.embed_query_crossmodal_audio's own dispatch, which only wires these two).
VERIFIABLE_EMBEDDERS = ("glap", "omni-embed-nemotron")

TOLERANCE = 1e-4  # batch/single consistency + non-NaN comparison tolerance


def _make_wav(path: str, freq: float = 220.0, sr: int = 16000, dur: float = 0.6) -> None:
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


def _document_embed(embedder: str, texts: list[str]):
    """The DOCUMENT/key-side text embed call for ``embedder`` -- mirrors
    ``kb_batch_build.build_squtr_corpus_source``'s own per-embedder dispatch (glap: encode_text via
    embed_text(embedder='glap'); omni-embed-nemotron: encode_document via
    embed_text(embedder='omni-embed-nemotron'))."""
    import kb_embed

    if embedder == "glap":
        _name, vecs = kb_embed._glap_embed_text(texts, source_lang="eng_Latn")
        return vecs
    if embedder == "omni-embed-nemotron":
        _name, vecs = kb_embed._omni_embed_document_text(texts)
        return vecs
    raise ValueError(f"live_verify_crossmodal: no document-text embed path wired for {embedder!r}")


def run_live_verification(embedder: str) -> dict:
    """Runs every check documented in this module's docstring for ``embedder``. Returns
    ``{"embedder", "checks": {name: bool}, "all_pass": bool, "detail": {...}}`` -- never raises for
    an ordinary check FAILURE (a failed check is recorded, not an exception); DOES let a genuine
    infrastructure error (e.g. the model truly cannot load at all) propagate, since that itself is
    diagnostic information for whoever is running this live.
    """
    import numpy as np

    import kb_embed

    if embedder not in VERIFIABLE_EMBEDDERS:
        raise ValueError(
            f"live_verify_crossmodal: {embedder!r} not in VERIFIABLE_EMBEDDERS={VERIFIABLE_EMBEDDERS} "
            "-- this script only knows how to verify the 2 tokens ticket #37 item 4 downgraded."
        )

    checks: dict[str, bool] = {}
    detail: dict = {}

    with tempfile.TemporaryDirectory(prefix="live_verify_crossmodal_") as wavdir:
        # a real on-disk wav (synthesized sine, but read through the real audio-loading path) --
        # the "positive" query.
        query_wav = os.path.join(wavdir, "query.wav")
        _make_wav(query_wav, freq=440.0)
        query_wav_2 = os.path.join(wavdir, "query2.wav")  # a DIFFERENT wav, for the batch check
        _make_wav(query_wav_2, freq=880.0)

        # --- 1/2: real model load + real wav query ---
        try:
            _name, q_vecs = kb_embed.embed_query_crossmodal_audio([query_wav], embedder=embedder)
            checks["1_real_model_load"] = True
            checks["2_real_wav_query"] = True
        except Exception as e:  # noqa: BLE001 -- record and stop; every downstream check is moot
            checks["1_real_model_load"] = False
            checks["2_real_wav_query"] = False
            detail["load_or_query_error"] = f"{type(e).__name__}: {e}"
            return {"embedder": embedder, "checks": checks, "all_pass": False, "detail": detail}

        q_vec = np.asarray(q_vecs, dtype="float32")[0]

        # --- 3: real text-corpus embedding (1 positive + 3 negatives) ---
        positive_text = "a synthetic 440 Hz sine tone test recording"
        negative_texts = [
            "quarterly financial results for the fiscal year",
            "instructions for assembling a wooden bookshelf",
            "a recipe for baking sourdough bread",
        ]
        corpus_texts = [positive_text] + negative_texts
        corpus_vecs = np.asarray(_document_embed(embedder, corpus_texts), dtype="float32")
        checks["3_real_text_corpus_embedding"] = corpus_vecs.shape[0] == len(corpus_texts)

        # --- 4: dim match ---
        dim_match = q_vec.shape[0] == corpus_vecs.shape[1]
        checks["4_dim_match"] = bool(dim_match)
        detail["query_dim"] = int(q_vec.shape[0])
        detail["corpus_dim"] = int(corpus_vecs.shape[1])
        if not dim_match:
            checks["all_pass_early_exit"] = "dim mismatch -- skipping downstream checks"
            return {"embedder": embedder, "checks": checks, "all_pass": False, "detail": detail}

        # --- 5: non-NaN ---
        checks["5_non_nan"] = bool(np.isfinite(q_vec).all() and np.isfinite(corpus_vecs).all())

        # --- 6/7: known-positive top-k hit + negative ordering ---
        sims = corpus_vecs @ q_vec / (
            np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-9
        )
        detail["sims"] = {t: float(s) for t, s in zip(corpus_texts, sims)}
        best_idx = int(np.argmax(sims))
        checks["6_known_positive_top1_hit"] = (best_idx == 0)
        checks["7_negative_ordering_strict"] = bool(
            all(sims[0] > sims[i] for i in range(1, len(sims)))
        )

        # --- 8: batch/single consistency ---
        _name_single, q_vec_single = kb_embed.embed_query_crossmodal_audio([query_wav], embedder=embedder)
        _name_batch, q_vecs_batch = kb_embed.embed_query_crossmodal_audio(
            [query_wav, query_wav_2], embedder=embedder,
        )
        q_vec_single = np.asarray(q_vec_single, dtype="float32")[0]
        q_vec_in_batch = np.asarray(q_vecs_batch, dtype="float32")[0]
        max_abs_diff = float(np.max(np.abs(q_vec_single - q_vec_in_batch)))
        detail["batch_single_max_abs_diff"] = max_abs_diff
        checks["8_batch_single_consistency"] = max_abs_diff <= TOLERANCE

    all_pass = all(v is True for v in checks.values())
    return {"embedder": embedder, "checks": checks, "all_pass": all_pass, "detail": detail}


def print_upgrade_edit(embedder: str) -> None:
    print(
        "\nALL CHECKS PASSED -- to promote this token, edit "
        "scripts/knowledge/kb_retrieve.py's CROSS_MODAL_AUDIO_QUERY_STATUS:\n\n"
        f'    "{embedder}": "supported",   # LIVE-VERIFIED {__import__("datetime").datetime.now().date()} '
        "via scripts/knowledge/live_verify_crossmodal.py -- see its printed check results for the run\n\n"
        "(replacing the current \"pending-live-verification\" entry for this token). Also update "
        "test_kb_gate.py's cases exercising this token's expected status accordingly."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embedder", required=True, choices=VERIFIABLE_EMBEDDERS)
    args = ap.parse_args()

    print(f"[live_verify_crossmodal] running LIVE verification for embedder={args.embedder!r} "
          "(real model load, CPU, no monkeypatching) ...", flush=True)
    result = run_live_verification(args.embedder)

    print("\n-- per-check results --")
    for name, ok in result["checks"].items():
        status = "PASS" if ok is True else ("SKIP" if not isinstance(ok, bool) else "FAIL")
        print(f"  [{status}] {name}" + (f": {ok}" if not isinstance(ok, bool) else ""))

    import json

    print("\n-- detail --")
    print(json.dumps(result["detail"], indent=2, ensure_ascii=False, default=str))

    if result["all_pass"]:
        print_upgrade_edit(args.embedder)
        return 0
    print(
        f"\nNOT all checks passed for {args.embedder!r} -- status stays "
        "'pending-live-verification' (or investigate further; do not hand-edit to 'supported').",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

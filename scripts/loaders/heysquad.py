"""heysquad — HeySQuAD spoken-SQuAD QA loader (task B4-recovery).

Layout on disk: ``datasets/heysquad/data/{train,validation}-*-of-*.parquet`` (HF parquet export).
This loader targets the **validation** shards only (2 shards, 4158 rows total, verified
2026-07-09) — HeySQuAD, like the original SQuAD it wraps, ships train/validation only (no public
"test" split with golds), so ``load_heysquad``'s ``split`` default is ``"validation"``, not
``"test"`` (same documented divergence as ``speech_massive.py``'s ``"validation"``-default note).

Each row carries: ``audio`` (spoken rendering of ``question``, ``{"bytes","path"}`` — ``path`` is
always non-informative here, this loader always decodes ``bytes``), ``question`` (text the audio
renders), ``context`` (the SQuAD passage the question was drawn from), ``answers`` (SQuAD-style —
either ``{"text": [...], "answer_start": [...]}`` or a list of ``{"text", "answer_start"}`` dicts;
empty when unanswerable), ``is_impossible`` (bool, SQuAD-2.0-style "no answer exists" flag —
1657/2079 rows in one validation shard, i.e. the MAJORITY, are unanswerable), ``id`` (unique SQuAD
question id), ``transcription`` (a noisy ASR transcript of the spoken ``question`` audio, shipped
by the dataset itself — NOT this loader's own ASR, kept in meta for provenance only).

**Default-slice filtering**: rows with ``is_impossible=True`` are DROPPED (no gold answer exists
to score against) — the default slice keeps only rows with a non-empty ``answers`` list (1002/4158
validation rows, verified 2026-07-09). A caller wanting the unanswerable rows for a refusal-style
probe would need a separate accessor; out of scope here.

*** LEAKAGE WARNING (read before building any KB from this loader's ``context`` field) ***
Every row's ``context`` passage is the SQuAD passage the question's ``answers`` were extracted
FROM — i.e. **the context contains the gold answer verbatim, by construction** (T7 incident, see
``_repro/t7_errata.md`` / ``scripts/t7_leakage_audit.py`` / ``scripts/knowledge/kb_audit.py``):
T7 built a RAG knowledge base directly out of heysquad ``context`` passages and measured
``answer_in_own_KB ≈ 1.0`` — its "RAG gain" (+0.517) was answer-lookup, not knowledge utilization
(boundary-clean rerun T8: ``clean_H0 = -0.066``, null). **Any KB built from this loader's
``context`` field is inadmissible for a knowledge-utilization claim until it passes
``kb_audit.audit_texts`` / ``kb_audit.scrub_golds`` with ``leakage_audit.verdict == "CLEAN"``**
(``kb_build.build_source`` already enforces this gate — refuses a LEAKAGE verdict without
``force_persist``, see ``scripts/knowledge/kb_registry.py``). Do not inject raw ``context`` into a
generation prompt as "retrieved knowledge" without scrubbing first.

Row shape:
    wav  -> materialized wav path (``audio["bytes"]`` decoded via ``_common.wav_from_bytes``)
    gold -> list[str] (SQuAD-style acceptable answer texts; NEVER a single str — a question can
            have multiple annotator-provided acceptable answers)
    meta -> {"dataset": "heysquad", "item_id": id, "id": id, "split": "validation", "task": "qa",
             "context": context, "is_impossible": False, "transcription": transcription}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/loaders
from _common import SLICE_SEED, datasets_dir, freeze_ids, wav_cache_dir, wav_from_bytes  # noqa: E402


def _extract_answer_texts(answers) -> list[str]:
    """SQuAD-style ``answers`` -> list[str] gold texts (mirrors t7_rag_gate_probe.py's ``_golds``).

    Handles both shapes seen in the wild: a single ``{"text": [...], "answer_start": [...]}`` dict
    (columnar SQuAD export) and a list of ``{"text": ..., "answer_start": ...}`` dicts (row-wise).
    """
    if answers is None:
        return []
    vals: list = []
    if isinstance(answers, dict):
        t = answers.get("text")
        vals = list(t) if isinstance(t, (list, tuple)) else ([t] if t else [])
    elif isinstance(answers, (list, tuple)):
        for el in answers:
            if isinstance(el, dict) and el.get("text"):
                vals.append(el["text"])
            elif isinstance(el, str):
                vals.append(el)
    return [str(x) for x in vals if str(x).strip()]


def load_heysquad(split: str = "validation", n: int | None = None, seed: int = SLICE_SEED) -> list:
    """Load heysquad's validation slice, dropping unanswerable (``is_impossible``) rows.

    See module docstring for the LEAKAGE WARNING on ``meta["context"]`` before building any KB
    from this loader's output.
    """
    import numpy as np
    import pyarrow.parquet as pq

    if split != "validation":
        raise NotImplementedError(
            f"heysquad: split={split!r} not wired up -- only 'validation' ships a public-gold "
            "eval slice (train-*.parquet also exists on disk but is out of scope for this "
            "loader's recipe; see module docstring)."
        )

    data_dir = datasets_dir() / "heysquad" / "data"
    files = sorted(data_dir.glob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError(f"heysquad: no validation-*.parquet under {data_dir}")

    cols = ["id", "audio", "question", "context", "answers", "is_impossible", "transcription"]
    rows = []
    for f in files:
        rows += pq.read_table(f, columns=cols).to_pylist()
    rows.sort(key=lambda r: r["id"])  # stable enumeration order before sampling (contract step 2)

    kept = []
    for r in rows:
        if r.get("is_impossible"):
            continue
        golds = _extract_answer_texts(r.get("answers"))
        if golds:
            kept.append((r, golds))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(kept)) if n is not None else np.arange(len(kept))
    idx = idx[:n] if n is not None else idx

    cache_dir = wav_cache_dir("heysquad")
    out, ids = [], []
    for j in idx:
        r, golds = kept[int(j)]
        item_id = r["id"]
        wav = wav_from_bytes(r["audio"]["bytes"], f"heysquad_{item_id}", cache_dir)
        meta = {
            "dataset": "heysquad", "item_id": item_id, "id": item_id, "split": split, "task": "qa",
            "context": r["context"], "is_impossible": bool(r.get("is_impossible")),
            "transcription": r.get("transcription"),
        }
        out.append({"wav": wav, "gold": golds, "meta": meta})
        ids.append(item_id)

    freeze_ids("heysquad", ids, dataset="heysquad", seed=seed, extra={"split": split, "n_requested": n})
    return out


if __name__ == "__main__":
    try:
        rows = load_heysquad(n=3)
        keys = sorted(rows[0].keys()) if rows else []
        print(f"heysquad: n={len(rows)} keys={keys} gold0={repr(rows[0]['gold'])[:80] if rows else None}")
    except Exception as e:
        print(f"heysquad: FAILED {type(e).__name__}: {e}")

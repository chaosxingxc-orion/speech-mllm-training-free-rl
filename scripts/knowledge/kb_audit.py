"""kb_audit — the Information-Boundary Guard as a MACHINE guardrail (not human discipline).

Owner's standing failure mode (``wiki/Information-Boundary-Guard.md``): levers repeatedly
"worked" by feeding the model information the deployment lacks — most sharply the test item's
gold answer. The T7 RAG track is the in-thread instance: the heysquad KB is a pool of SQuAD
``context`` passages, and each passage CONTAINS its own item's answer, so
``answer_in_own_KB = 1.0`` — the +0.517 "RAG gain" was answer-lookup, not knowledge reasoning
(``t7_leakage_audit.py``; boundary-clean rerun T8 -> NULL).

This module runs that audit automatically at build time. Any source whose entries overlap the
eval golds is flagged LEAKAGE (or SUSPECT), and ``scrub_golds`` produces the boundary-clean
variant (mirrors ``t8_clean_rag_rerun.scrub``). No source is admissible for a knowledge-utilization
claim until its ``leakage_audit.verdict == "CLEAN"``.
"""
from __future__ import annotations

import re
import unicodedata


def norm(s) -> str:
    """NFKC-lower-alnum normalization — identical to ``p2_baselines.norm`` so audit == reward space."""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return "".join(c for c in s if c.isalnum() or c.isspace()).strip()


def ngram_containment(golds: list[str], entry_texts: list[str], n: int = 8) -> dict:
    """AUXILIARY containment score: does the KB (as a whole) contain gold-length-``n`` WORD
    n-grams? (2026-07-11, ticket #25 P3h). A word-n-gram check (the standard corpus-contamination-
    check convention, e.g. GPT-3's) catches a leaked gold even when the exact FULL-STRING match
    ``audit_texts`` looks for is broken up by minor reformatting on either side (e.g. extra
    whitespace/punctuation splitting) that still shares a long contiguous word run with the KB.
    Golds shorter than ``n`` words fall back to a single whole-gold "gram" (containment = 1 iff
    that short gold's normalized text is itself a WORD-run inside the KB).

    AUXILIARY ONLY — returns its own ``aux_verdict``, never the hard gate (``kb_build``'s
    enforcement gate keys off ``audit_texts``' exact-substring ``verdict`` exclusively; see this
    module's docstring + ``kb_schema.leakage_verdict``).
    """
    ng = [norm(g) for g in golds if norm(g)]

    def _grams(s: str, k: int) -> set[str]:
        words = s.split()
        if not words:
            return set()
        if len(words) < k:
            return {s}
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

    joined_grams = _grams(" ".join(norm(t) for t in entry_texts), n)
    per_gold = []
    n_contained = 0
    for g in ng:
        gg = _grams(g, n)
        overlap = (len(gg & joined_grams) / len(gg)) if gg else 0.0
        per_gold.append(round(overlap, 3))
        if overlap > 0:
            n_contained += 1
    rate = n_contained / len(ng) if ng else 0.0
    return {
        "n": n,
        "n_golds": len(ng),
        "n_golds_ngram_contained": n_contained,
        "ngram_containment_rate": round(rate, 3),
        "per_gold_overlap": per_gold,
        "aux_verdict": "LEAKAGE" if rate >= 0.5 else ("SUSPECT" if rate > 0 else "CLEAN"),
    }


def embedding_similarity_audit(golds: list[str], entry_texts: list[str], embedder: str = "auto",
                                threshold: float = 0.9) -> dict:
    """AUXILIARY audit: cosine similarity of gold-text embeddings vs. value-text embeddings, CPU,
    when a text embedder is actually available (2026-07-11, ticket #25 P3h). Catches PARAPHRASED
    leakage (same answer, different surface form) that neither the exact-substring nor the n-gram
    check would flag. Best-effort: if no text embedder can be loaded (no network for MiniLM, no
    sklearn for the TF-IDF fallback, ...), returns ``{"available": False, ...}`` rather than
    raising — this is a diagnostic extra, never something a build should hard-fail on.

    AUXILIARY ONLY — ``aux_verdict`` here NEVER feeds ``kb_build``'s enforcement gate (that gate is
    ``audit_texts``' exact-substring ``verdict`` exclusively).
    """
    ng = [g for g in golds if g and str(g).strip()]
    if not ng or not entry_texts:
        return {"available": True, "n_golds": len(ng), "n_entries": len(entry_texts),
                "threshold": threshold, "max_sim_per_gold": [], "rate_above_threshold": 0.0,
                "aux_verdict": "CLEAN"}
    try:
        import numpy as np

        import kb_embed

        _n1, gvecs, _f1 = kb_embed.embed_text(ng, embedder=embedder)
        _n2, evecs, _f2 = kb_embed.embed_text(list(entry_texts), embedder=embedder)
        sims = np.asarray(gvecs, dtype="float32") @ np.asarray(evecs, dtype="float32").T
        max_per_gold = sims.max(axis=1)
    except Exception as e:  # noqa: BLE001 -- best-effort diagnostic, never fatal to a build
        return {"available": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    rate = float((max_per_gold >= threshold).mean())
    return {
        "available": True,
        "n_golds": len(ng),
        "n_entries": len(entry_texts),
        "threshold": threshold,
        "max_sim_per_gold": [round(float(x), 3) for x in max_per_gold],
        "rate_above_threshold": round(rate, 3),
        "aux_verdict": "LEAKAGE" if rate >= 0.5 else ("SUSPECT" if rate > 0 else "CLEAN"),
    }


def audit_texts(entry_texts: list[str], golds: list[str], aux: bool = False,
                 aux_embedder: str = "auto") -> dict:
    """Answer-overlap audit: does the KB (as a whole) contain the eval golds verbatim?

    Returns a summary dict; ``verdict`` is LEAKAGE (>=0.5 of golds appear in the KB),
    SUSPECT (>0), or CLEAN (0). Also returns per-entry indices that contain some gold, so the
    build step can scrub or exclude them (provenance firewall).

    ``aux=True`` (2026-07-11, ticket #25 P3h) additionally runs ``ngram_containment`` (cheap, pure
    Python, always attempted) and ``embedding_similarity_audit`` (best-effort, CPU) and folds their
    results in under ``"ngram"``/``"embedding_similarity"`` keys — BOTH auxiliary; the top-level
    ``"verdict"`` this function returns (the hard gate every existing caller already keyed off) is
    computed EXACTLY as before, unaffected by ``aux``.
    """
    ng = [norm(g) for g in golds]
    ng = [g for g in ng if g]
    ne = [norm(t) for t in entry_texts]
    joined = "\n".join(ne)
    golds_leaked = [g for g in ng if g in joined]
    leaking_entry_idx = sorted(
        {i for i, t in enumerate(ne) for g in ng if g and g in t}
    )
    rate = len(golds_leaked) / len(ng) if ng else 0.0
    verdict = "LEAKAGE" if rate >= 0.5 else ("SUSPECT" if rate > 0 else "CLEAN")
    out = {
        "n_golds": len(ng),
        "n_entries": len(ne),
        "answer_overlap_rate": round(rate, 3),
        "n_golds_leaked": len(golds_leaked),
        "n_leaking_entries": len(leaking_entry_idx),
        "leaking_entry_idx": leaking_entry_idx,
        "verdict": verdict,
    }
    if aux:
        out["ngram"] = ngram_containment(golds, entry_texts)
        out["embedding_similarity"] = embedding_similarity_audit(golds, entry_texts, embedder=aux_embedder)
    return out


def scrub_golds(entry_texts: list[str], golds: list[str]) -> list[str]:
    """Remove gold answer spans from entries -> boundary-clean KB (mirrors t8_clean_rag_rerun.scrub).

    Case-insensitive removal of each gold surface form. Leaves the passage otherwise intact so the
    model can still be tested on genuine knowledge/reasoning rather than answer-copying.
    """
    surfaces = sorted({g for g in golds if g and str(g).strip()}, key=len, reverse=True)
    out = []
    for t in entry_texts:
        s = t
        for g in surfaces:
            s = re.sub(re.escape(str(g)), " ", s, flags=re.IGNORECASE)
        out.append(re.sub(r"\s+", " ", s).strip())
    return out


def audit_source(source: str, golds: list[str], aux: bool = False, aux_embedder: str = "auto") -> dict:
    """Load a persisted source's values (``values.jsonl``, one ``KnowledgeValue`` per row) and audit
    them against ``golds``. ``aux``/``aux_embedder`` forward to ``audit_texts`` (2026-07-11, ticket
    #25 P3h) -- see its docstring."""
    from kb_schema import load_values

    texts = [v.value or "" for v in load_values(source)]
    return audit_texts(texts, golds, aux=aux, aux_embedder=aux_embedder)

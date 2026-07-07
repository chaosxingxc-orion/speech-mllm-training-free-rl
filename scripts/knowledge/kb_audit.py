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


def audit_texts(entry_texts: list[str], golds: list[str]) -> dict:
    """Answer-overlap audit: does the KB (as a whole) contain the eval golds verbatim?

    Returns a summary dict; ``verdict`` is LEAKAGE (>=0.5 of golds appear in the KB),
    SUSPECT (>0), or CLEAN (0). Also returns per-entry indices that contain some gold, so the
    build step can scrub or exclude them (provenance firewall).
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
    return {
        "n_golds": len(ng),
        "n_entries": len(ne),
        "answer_overlap_rate": round(rate, 3),
        "n_golds_leaked": len(golds_leaked),
        "n_leaking_entries": len(leaking_entry_idx),
        "leaking_entry_idx": leaking_entry_idx,
        "verdict": verdict,
    }


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


def audit_source(source: str, golds: list[str]) -> dict:
    """Load a persisted source's entries and audit them against ``golds``."""
    from kb_schema import KnowledgeEntry, source_dir

    p = source_dir(source) / "entries.jsonl"
    texts = [
        (KnowledgeEntry.from_json(line).content_text or "")
        for line in open(p, encoding="utf-8")
        if line.strip()
    ]
    return audit_texts(texts, golds)

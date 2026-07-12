"""scripts/knowledge/knowledge_card.py — versioned Knowledge Card schema v1 (续21-A③, M1
engineering-base item 5, 2026-07-13).

## Why this exists

Owner ruling 续21-A②/③ ("使用段收敛为规范"): the retrieved-knowledge DELIVERY step converges to
ONE standardized, versioned "knowledge card" shape, with Stage-2's comparison axis simplified to
**standard card vs. flat** (rather than each delivery mode inventing its own ad hoc rendering).
Before this module, ``run_mock.py``'s ``structured-ref`` delivery mode rendered a bare
``"[grain] passage text"`` line per hit (``_grain_marker``/``_structured_ref_prompt``) — no
provenance (which source built this), no relevance signal (how confident was the retrieval?), and
no explicit instruction to the model on how much to trust the card. This module fixes that gap
with a schema-versioned, four-field card:

    content            -- the retrieved VALUE text itself (``KnowledgeValue.value`` verbatim).
    source_tag          -- WHERE this came from: source name + key_modality + value_type + grain —
                            traceability for a human/audit trail, never itself fed to the model as
                            a truth claim.
    relevance_signal     -- the retrieval-time evidence of relevance: cosine similarity + rank
                            among the delivered set + a FIXED similarity band (see
                            ``RELEVANCE_BANDS``) — "how confident was retrieval", not a model
                            judgment.
    usage_directive       -- an explicit, model-facing instruction on how much to trust this card,
                            chosen as a FIXED (never learned/adaptive) function of
                            ``relevance_signal``'s band — matches ``run_mock.py``'s own
                            no-adaptive-logic discipline for the retrieval/delivery stages (see
                            that module's ``assert_no_adaptive_logic``): the band->directive table
                            is a static lookup, not a per-item decision.

## Versioning

``SCHEMA_VERSION`` is stamped into every rendered card and every ``KnowledgeCard.schema_version``.
Bump it (and extend ``render_card``'s dispatch, never silently reinterpreting an old version's
fields) whenever the field set changes — the rendered text always says which version produced it,
so a downstream consumer (or a human reading a transcript) can tell which schema's contract it's
reading.

## Wired into run_mock's delivery arm

``run_mock.py``'s ``structured-ref`` delivery mode (``_structured_ref_prompt``) now renders via
``render_cards_block`` below instead of its old ad hoc grain-marker line format — see that
function's docstring for the exact call site. ``render_flat`` is this module's OWN flat-delivery
counterpart (deliberately NOT importing ``run_mock`` — ``scripts/knowledge`` must never depend on
``scripts/baselines``, the reverse of the existing dependency direction, per
``kb_batch_build.py``'s own layering note), so the "standard card vs flat" comparison pair is
fully self-contained and testable from this module alone.

Lazy-import discipline (CLAUDE.md): this module has NO heavy imports at all — pure stdlib
dataclasses/string formatting, so ``import knowledge_card`` is always cheap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------------------------
# relevance banding -- a FIXED (non-adaptive) function of the retrieval-time cosine similarity.
# Thresholds are deliberately coarse/documented defaults (not tuned against any eval data --
# tuning these against held-out results would itself be an adaptive-logic violation of the kind
# run_mock.assert_no_adaptive_logic guards against for the retrieval/delivery stages); a caller
# with domain-specific calibration may pass its own bands via `bands=` to make_card/render_*.
# ---------------------------------------------------------------------------------------------
RELEVANCE_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.85, "high",
     "High-confidence match. If this card directly answers the question, you may rely on it; "
     "still verify it is consistent with the rest of the context before using it."),
    (0.60, "medium",
     "Medium-confidence match. May be only partially relevant -- cross-check against the "
     "question before using it as evidence."),
    (0.0, "low",
     "Low-confidence match. May be irrelevant or only tangentially related -- verify carefully "
     "before using it, and prefer your own knowledge if this card conflicts with it."),
)


def relevance_band(sim: float, bands: tuple = RELEVANCE_BANDS) -> tuple[str, str]:
    """``sim`` -> ``(band_name, usage_directive)`` via the FIRST band whose threshold ``sim`` meets
    (bands must be given threshold-descending, the default's own order). Always returns a value —
    the last band's threshold is ``0.0``, so every real similarity (even negative, for a
    pathological embedder) matches it."""
    for threshold, band, directive in bands:
        if sim >= threshold:
            return band, directive
    return bands[-1][1], bands[-1][2]


@dataclass(frozen=True)
class KnowledgeCard:
    """One rendered unit of delivered knowledge — see module docstring for what each field means."""

    content: str
    source_tag: str
    relevance_signal: dict = field(default_factory=dict)  # {"sim": float, "rank": int, "band": str}
    usage_directive: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def source_tag_for(value) -> str:
    """``value``: a ``kb_schema.KnowledgeValue`` (duck-typed — only ``.source``/``.key_modality``/
    ``.value_type``/``.grain`` are read, so a plain object/dict-like stand-in works too, which is
    what ``test_knowledge_card.py``'s golden fixture uses instead of a real KB build)."""
    grain = getattr(value, "grain", None) or "untagged"
    return (f"source={getattr(value, 'source', '?')} "
            f"key_modality={getattr(value, 'key_modality', '?')} "
            f"value_type={getattr(value, 'value_type', '?')} grain={grain}")


def make_card(hit: dict, rank: int, bands: tuple = RELEVANCE_BANDS) -> KnowledgeCard:
    """``hit``: ``{"value": <KnowledgeValue-like>, "sim": float}`` — ``kb_retrieve.retrieve()``'s
    own per-item hit shape (row 0 of its return list). ``rank`` is the hit's 0-based position in
    the DELIVERED set (caller-supplied — this function does not re-sort; pass hits in the order
    they will actually be delivered, e.g. after ``run_mock._apply_relevance_edge_reorder``)."""
    value = hit["value"]
    sim = float(hit.get("sim", 0.0))
    band, directive = relevance_band(sim, bands)
    return KnowledgeCard(
        content=getattr(value, "value", str(value)),
        source_tag=source_tag_for(value),
        relevance_signal={"sim": sim, "rank": rank, "band": band},
        usage_directive=directive,
    )


def render_card(card: KnowledgeCard) -> str:
    """ONE card -> a single human/model-readable text block, schema-version-tagged."""
    rel = card.relevance_signal
    sim = rel.get("sim")
    sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else str(sim)
    header = (f"[knowledge-card v{card.schema_version} | {card.source_tag} | "
              f"relevance={rel.get('band')} sim={sim_str} rank={rel.get('rank')}]")
    return f"{header}\n{card.content}\nusage_directive: {card.usage_directive}"


def render_cards_block(hits: list[dict], base_instruction: str, bands: tuple = RELEVANCE_BANDS) -> str:
    """The STANDARD-CARD delivery: every hit rendered as its own versioned card, joined, prepended
    to ``base_instruction``. Empty ``hits`` -> ``base_instruction`` verbatim (no empty "Reference
    material:" header when nothing was retrieved)."""
    if not hits:
        return base_instruction
    cards = [make_card(h, rank=i, bands=bands) for i, h in enumerate(hits)]
    block = "\n---\n".join(render_card(c) for c in cards)
    return (
        f"Reference material (retrieved, standardized knowledge-card v{SCHEMA_VERSION} format -- "
        "each card's usage_directive tells you how much to trust it; the set as a whole may or "
        f"may not be relevant):\n{block}\n\n{base_instruction}"
    )


def render_flat(hits: list[dict], base_instruction: str) -> str:
    """The FLAT counterpart of ``render_cards_block`` (the "S2 收敛为标准卡 vs flat" comparison
    pair's other arm) — content only, no source-tag/relevance-signal/usage-directive, matching
    ``run_mock._flat_prompt``'s existing shape exactly (duplicated here, not imported, per this
    module's layering note) so the pair is comparable and self-contained in ONE module."""
    if not hits:
        return base_instruction
    passages = [getattr(h["value"], "value", str(h["value"])) for h in hits]
    block = "\n---\n".join(passages)
    return f"Reference material (retrieved, may or may not be relevant):\n{block}\n\n{base_instruction}"

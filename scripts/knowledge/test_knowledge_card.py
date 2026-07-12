"""scripts/knowledge/test_knowledge_card.py — golden-fixture + unit checks for
``knowledge_card.py`` (Knowledge Card schema v1, 续21-A③, M1 engineering-base item 5, 2026-07-13),
plus an integration check that ``run_mock.py``'s ``structured-ref`` delivery mode actually renders
through it (not just a parallel, unused module).

Pure-python, no GPU/network/data-root. Each check is a bare ``test_*()`` function,
pytest-collectible; ``main()`` also runs every one standalone with a PASS/FAIL summary:

    python -u scripts/knowledge/test_knowledge_card.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                       # scripts/knowledge
SCRIPTS = os.path.dirname(HERE)                                           # scripts
BASELINES_DIR = os.path.join(SCRIPTS, "baselines")
for _p in (HERE, BASELINES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import knowledge_card as kc  # noqa: E402


class _FakeValue:
    """Duck-typed stand-in for kb_schema.KnowledgeValue -- only the 4 attributes
    make_card/source_tag_for read (value/source/key_modality/value_type/grain)."""

    def __init__(self, value, source, key_modality="audio", value_type="text-fact", grain=None):
        self.value = value
        self.source = source
        self.key_modality = key_modality
        self.value_type = value_type
        self.grain = grain


def _hit(value, sim):
    return {"value": value, "sim": sim}


# ---------------------------------------------------------------------------------------------
# relevance_band
# ---------------------------------------------------------------------------------------------

def test_relevance_band_boundaries():
    assert kc.relevance_band(0.95)[0] == "high"
    assert kc.relevance_band(0.85)[0] == "high"       # exactly at the "high" threshold
    assert kc.relevance_band(0.849999)[0] == "medium"
    assert kc.relevance_band(0.60)[0] == "medium"       # exactly at the "medium" threshold
    assert kc.relevance_band(0.59999)[0] == "low"
    assert kc.relevance_band(0.0)[0] == "low"
    assert kc.relevance_band(-1.0)[0] == "low"           # pathological negative sim -> still "low", never crashes


def test_relevance_band_custom_bands_override():
    custom = ((0.5, "top", "TOP DIRECTIVE"), (0.0, "bottom", "BOTTOM DIRECTIVE"))
    band, directive = kc.relevance_band(0.6, bands=custom)
    assert band == "top" and directive == "TOP DIRECTIVE"
    band, directive = kc.relevance_band(0.1, bands=custom)
    assert band == "bottom" and directive == "BOTTOM DIRECTIVE"


# ---------------------------------------------------------------------------------------------
# make_card / source_tag_for
# ---------------------------------------------------------------------------------------------

def test_make_card_fields():
    v = _FakeValue("Paris is the capital of France.", source="heysquad__glap__single-utt__knowledge-passage",
                    key_modality="audio", value_type="text-fact", grain="knowledge")
    card = kc.make_card(_hit(v, 0.91), rank=0)
    assert card.content == "Paris is the capital of France."
    assert card.schema_version == kc.SCHEMA_VERSION == 1
    assert card.relevance_signal == {"sim": 0.91, "rank": 0, "band": "high"}
    assert "heysquad__glap__single-utt__knowledge-passage" in card.source_tag
    assert "key_modality=audio" in card.source_tag
    assert "value_type=text-fact" in card.source_tag
    assert "grain=knowledge" in card.source_tag
    assert card.usage_directive == kc.RELEVANCE_BANDS[0][2]


def test_make_card_untagged_grain_defaults_to_untagged_string():
    v = _FakeValue("some content", source="src", grain=None)
    card = kc.make_card(_hit(v, 0.3), rank=2)
    assert "grain=untagged" in card.source_tag
    assert card.relevance_signal["band"] == "low"


def test_card_to_dict_json_serializable():
    v = _FakeValue("x", source="s")
    card = kc.make_card(_hit(v, 0.7), rank=0)
    d = card.to_dict()
    json.dumps(d)  # must not raise
    assert d["schema_version"] == 1


# ---------------------------------------------------------------------------------------------
# render_card / render_cards_block -- GOLDEN FIXTURES (exact string match; any format drift must
# be a DELIBERATE, reviewed change to this test, not an accidental byproduct of some other edit).
# ---------------------------------------------------------------------------------------------

def test_render_card_golden_fixture():
    v = _FakeValue("The mitochondria is the powerhouse of the cell.",
                    source="bio-facts__glap__single-utt__knowledge-passage",
                    key_modality="audio", value_type="text-fact", grain="knowledge")
    card = kc.make_card(_hit(v, 0.912345), rank=0)
    rendered = kc.render_card(card)
    expected = (
        "[knowledge-card v1 | source=bio-facts__glap__single-utt__knowledge-passage "
        "key_modality=audio value_type=text-fact grain=knowledge | relevance=high sim=0.912 rank=0]\n"
        "The mitochondria is the powerhouse of the cell.\n"
        "usage_directive: High-confidence match. If this card directly answers the question, you "
        "may rely on it; still verify it is consistent with the rest of the context before using it."
    )
    assert rendered == expected


def test_render_cards_block_golden_fixture_two_hits():
    v1 = _FakeValue("Card one content.", source="src-a", grain="knowledge")
    v2 = _FakeValue("Card two content.", source="src-b", grain="memory")
    block = kc.render_cards_block([_hit(v1, 0.9), _hit(v2, 0.4)], "Answer the question.")
    expected = (
        "Reference material (retrieved, standardized knowledge-card v1 format -- each card's "
        "usage_directive tells you how much to trust it; the set as a whole may or may not be "
        "relevant):\n"
        "[knowledge-card v1 | source=src-a key_modality=audio value_type=text-fact grain=knowledge | "
        "relevance=high sim=0.900 rank=0]\n"
        "Card one content.\n"
        "usage_directive: High-confidence match. If this card directly answers the question, you "
        "may rely on it; still verify it is consistent with the rest of the context before using it.\n"
        "---\n"
        "[knowledge-card v1 | source=src-b key_modality=audio value_type=text-fact grain=memory | "
        "relevance=low sim=0.400 rank=1]\n"
        "Card two content.\n"
        "usage_directive: Low-confidence match. May be irrelevant or only tangentially related -- "
        "verify carefully before using it, and prefer your own knowledge if this card conflicts "
        "with it.\n\n"
        "Answer the question."
    )
    assert block == expected


def test_render_cards_block_empty_hits_returns_base_instruction_verbatim():
    assert kc.render_cards_block([], "Answer the question.") == "Answer the question."


def test_render_flat_golden_fixture():
    v1 = _FakeValue("Flat content one.", source="src-a")
    v2 = _FakeValue("Flat content two.", source="src-b")
    flat = kc.render_flat([_hit(v1, 0.9), _hit(v2, 0.4)], "Answer the question.")
    expected = (
        "Reference material (retrieved, may or may not be relevant):\n"
        "Flat content one.\n---\nFlat content two.\n\nAnswer the question."
    )
    assert flat == expected


def test_render_flat_empty_hits_returns_base_instruction_verbatim():
    assert kc.render_flat([], "Answer the question.") == "Answer the question."


def test_standard_card_and_flat_diverge_in_information_content():
    """The whole point of the "standard card vs flat" comparison pair: card carries strictly MORE
    (source-tag/relevance/usage-directive), flat carries strictly the content only."""
    v = _FakeValue("Shared content.", source="src-x", grain="knowledge")
    card_block = kc.render_cards_block([_hit(v, 0.7)], "Q?")
    flat_block = kc.render_flat([_hit(v, 0.7)], "Q?")
    assert "usage_directive" in card_block and "usage_directive" not in flat_block
    assert "source=src-x" in card_block and "source=src-x" not in flat_block
    assert "Shared content." in card_block and "Shared content." in flat_block


# ---------------------------------------------------------------------------------------------
# integration: run_mock.py's structured-ref delivery mode actually renders THROUGH this module.
# ---------------------------------------------------------------------------------------------

def test_run_mock_structured_ref_delivery_uses_knowledge_card():
    import run_mock as rm  # scripts/baselines/run_mock.py

    cfg = rm.MockConfig(
        embedder="glap", key_org="single-utt", value_org="knowledge-passage",
        retrieval=rm.RetrievalConfig(), query_construction="audio-direct", delivery="structured-ref",
    )
    v = _FakeValue("Some retrieved fact.", source="ds__glap__single-utt__knowledge-passage",
                    grain="knowledge")
    rendered = rm.render_delivery(cfg, [_hit(v, 0.88)], "What is the fact?")
    assert isinstance(rendered, str)
    assert f"knowledge-card v{kc.SCHEMA_VERSION}" in rendered
    assert "usage_directive:" in rendered
    assert "Some retrieved fact." in rendered
    assert "What is the fact?" in rendered
    # matches knowledge_card.render_cards_block's own output for the identical input, byte for byte
    assert rendered == kc.render_cards_block([_hit(v, 0.88)], "What is the fact?")


def test_run_mock_structured_ref_empty_hits():
    import run_mock as rm

    cfg = rm.MockConfig(
        embedder="glap", key_org="single-utt", value_org="knowledge-passage",
        retrieval=rm.RetrievalConfig(), query_construction="audio-direct", delivery="structured-ref",
    )
    assert rm.render_delivery(cfg, [], "Just the question.") == "Just the question."


# ---------------------------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    results: dict[str, bool] = {}
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}", flush=True)
            results[name] = True
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            results[name] = False
    all_pass = all(results.values())
    print("\n=== KNOWLEDGE_CARD TEST ===")
    print(json.dumps(results, indent=2))
    print("KNOWLEDGE_CARD_TEST_PASS" if all_pass else "KNOWLEDGE_CARD_TEST_FAIL", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

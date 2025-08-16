from scripts.api import extract_entities, _strip_think, _extract_final_short


def test_extract_entities_basic():
    q = 'Where is "Random House Tower" located near 888 7th Avenue?'
    ents = extract_entities(q)
    s = {e.lower() for e in ents}
    assert "random house tower" in s
    assert any("888" in e for e in s)


def test_strip_think_and_short_extractor():
    raw = "<think>chain of thought</think> Final answer: Lisbon, Portugal."
    assert _strip_think(raw) == "Final answer: Lisbon, Portugal."
    short = _extract_final_short(raw, max_words=3)
    assert short.lower().startswith("lisbon")


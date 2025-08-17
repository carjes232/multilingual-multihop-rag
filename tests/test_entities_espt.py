from scripts.api import extract_entities


def test_extract_entities_detects_accented_proper_nouns():
    q = "Did São Paulo host events featuring José Martí?"
    ents = {e.lower() for e in extract_entities(q)}
    
    # Check for both "São Paulo" and "José Martí"
    assert "são paulo" in ents
    assert "josé martí" in ents
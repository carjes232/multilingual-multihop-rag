from scripts.api import SearchHit, build_prompt, compute_citations, is_yesno_question


def make_hit(idx: int, text: str, title: str = "T") -> SearchHit:
    return SearchHit(id=f"c{idx}", doc_id=f"d{idx}", title=title, text=text, score=1.0)


def test_is_yesno_question_examples():
    assert is_yesno_question("Is it raining?") is True
    assert is_yesno_question("ARE we there yet?") is True
    assert is_yesno_question("What is the capital?") is False


def test_compute_citations_simple_matches():
    ans = "Lisbon"
    hits = [
        make_hit(1, "The capital of Portugal is Lisbon."),
        make_hit(2, "Porto is a city in Portugal."),
        make_hit(3, "Lisbon, Portugal is coastal."),
    ]
    idxs = compute_citations(ans, hits)
    # Should cite 1 and 3 (1-based indexing), max 3 cites
    assert idxs[:2] == [1, 3]


def test_build_prompt_yesno_and_short_flags():
    q1 = "Is water wet?"
    q2 = "Name the capital of Portugal."
    hits = [make_hit(1, "dummy text", title="A Title")]
    p1 = build_prompt(q1, hits)
    p2 = build_prompt(q2, hits)
    assert "Answer (yes/no only):" in p1
    assert "Answer (short):" in p2

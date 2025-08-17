from scripts.api import SearchHit, compute_citations


def make_hit(idx: int, text: str, title: str = "T") -> SearchHit:
    return SearchHit(id=f"c{idx}", doc_id=f"d{idx}", title=title, text=text, score=1.0)


def test_compute_citations_handles_diacritics_and_case():
    # São Paulo should match Sao Paulo (no diacritics) and vice versa
    hits = [
        make_hit(1, "The city of São Paulo is in Brazil."),
        make_hit(2, "Another text"),
    ]
    idxs = compute_citations("sao paulo", hits)
    assert 1 in idxs and len(idxs) >= 1


def test_compute_citations_ignores_too_short_answers():
    hits = [make_hit(1, "of and in are stopwords"), make_hit(2, "random text")]
    # very short tokens like 'of' should not produce citations
    assert compute_citations("of", hits) == []

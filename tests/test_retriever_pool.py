from scripts.retriever import Hit, pool_and_rerank


def test_pool_and_rerank_keeps_best_score():
    a1 = [Hit(id="c1", doc_id="d1", title="T1", text="x", score=0.4),
          Hit(id="c2", doc_id="d2", title="T2", text="y", score=0.2)]
    a2 = [Hit(id="c1", doc_id="d1", title="T1", text="x", score=0.9),
          Hit(id="c3", doc_id="d3", title="T3", text="z", score=0.3)]
    out = pool_and_rerank([a1, a2], top_k=2, max_pool=10)
    # c1 should have score 0.9 and appear first
    assert out[0].id == "c1" and out[0].score == 0.9
    # Next should be c3 (0.3) over c2 (0.2)
    assert out[1].id == "c3"


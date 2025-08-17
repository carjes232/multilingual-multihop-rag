from scripts.eval_multilingual_retrieval import jaccard


def test_jaccard_basic_cases():
    assert jaccard([], []) == 1.0
    assert jaccard(["a"], []) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3

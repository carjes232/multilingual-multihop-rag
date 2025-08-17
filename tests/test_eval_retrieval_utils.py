import numpy as np
from scripts.eval_retrieval import (
    gold_titles_from_example,
    parse_k_list,
    vec_to_pgvector,
)


def test_parse_k_list_happy_and_errors():
    assert parse_k_list("1,5,10", 3) == [1, 5, 10]
    assert parse_k_list("1, 1, 2", 3) == [1, 2]
    assert parse_k_list("", 7) == [7]
    try:
        parse_k_list("a,2", 3)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_vec_to_pgvector_formatting():
    v = np.array([0.1, -0.2, 0.0], dtype=np.float32)
    s = vec_to_pgvector(v)
    assert s == "[0.100000,-0.200000,0.000000]"


def test_gold_titles_from_example():
    ex = {"context": {"title": ["A", "B"]}}
    assert gold_titles_from_example(ex) == {"A", "B"}
    assert gold_titles_from_example({"context": {}}) == set()

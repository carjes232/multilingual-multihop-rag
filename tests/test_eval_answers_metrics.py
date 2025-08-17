from scripts.eval_answers import exact_match, f1_score, normalize_text, percentile


def test_normalize_text_and_exact_match():
    a = "The Random–House Tower!"
    b = "random-house tower"
    assert normalize_text(a) == "random-house tower"
    assert exact_match(a, b) is True


def test_f1_score_basic():
    assert f1_score("Lisbon, Portugal", "Lisbon, Portugal") == 1.0
    assert f1_score("", "Lisbon") == 0.0
    assert f1_score("Lisbon", "") == 0.0
    assert f1_score("Lisbon", "Porto") == 0.0


def test_percentile_selection():
    xs = [10, 20, 30, 40]
    assert percentile(xs, 0) == 10
    assert percentile(xs, 50) == 30  # nearest-rank rounding
    assert percentile(xs, 100) == 40

from scripts.search import fmt_snippet


def test_fmt_snippet_truncates_and_replaces_newlines():
    txt = "line1\nline2\nline3"
    s = fmt_snippet(txt, width=8)
    # "line1 line2 line3" -> truncated to width 8 with ellipsis
    assert s.endswith("…")
    assert "\n" not in s

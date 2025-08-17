import importlib
import os


def test_has_gold_titles_and_headers_env(monkeypatch):
    mod = importlib.import_module("scripts.make_multilingual_eval_set")

    # has_gold_titles
    assert mod.has_gold_titles({"context": {"title": ["T1"]}}) is True
    assert mod.has_gold_titles({"context": {"title": []}}) is False
    assert mod.has_gold_titles({"context": {}}) is False

    # _headers requires API key; simulate missing and present
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    # Update module-level constant used by _headers
    mod.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        mod._headers()
        assert False, "Expected RuntimeError when API key missing"
    except RuntimeError:
        pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    mod.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    h = mod._headers()
    assert h.get("Authorization", "").startswith("Bearer ")

from unittest.mock import patch

from fastapi.testclient import TestClient
from scripts.api import _TITLE_SPAN, SearchHit, app, token_bearer

client = TestClient(app)


# This function will override the token_bearer dependency, effectively disabling auth for tests
async def override_dependency() -> None:
    return None


app.dependency_overrides[token_bearer] = override_dependency


def test_answer_endpoint():
    params = {
        "q": "What is the capital of France?",
        "k": 4,
        "model": "qwen3:8b",
        "temperature": 0.2,
        "max_tokens": 80,
    }
    # Mock downstream network calls to isolate the API endpoint logic
    mock_hits = [SearchHit(id='c1', doc_id='d1', title='France', text='The capital of France is Paris.', score=0.9)]
    with patch('scripts.api.list_ollama_models', return_value=["qwen3:8b"]), \
         patch('scripts.api.search_multi', return_value=mock_hits), \
         patch('scripts.api.call_ollama', return_value="Paris"):

        response = client.post("/answer", params=params)

    assert response.status_code == 200
    assert "answer" in response.json()
    assert "hits" in response.json()


def test_title_span_basic():
    input_text = "This Is A Test"
    expected_output = ["This Is A Test"]
    assert _TITLE_SPAN.findall(input_text) == expected_output


def test_title_span_multiple():
    input_text = "This Is A Test. Another Example."
    expected_output = ["This Is A Test", "Another Example"]
    assert _TITLE_SPAN.findall(input_text) == expected_output


def test_title_span_single_word():
    input_text = "Test"
    expected_output = []
    assert _TITLE_SPAN.findall(input_text) == expected_output


def test_title_span_special_characters():
    input_text = "This Is A Test, with special characters!"
    expected_output = ["This Is A Test"]
    assert _TITLE_SPAN.findall(input_text) == expected_output


def test_title_span_different_languages():
    input_text = "São Paulo"
    expected_output = ["São Paulo"]  # Corrected expectation
    assert _TITLE_SPAN.findall(input_text) == expected_output
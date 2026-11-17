import pytest
from fastapi.testclient import TestClient

import step5_api


class FakeLlama:
    def __init__(self, raise_value_error=False):
        self.raise_value_error = raise_value_error
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.raise_value_error:
            raise ValueError("Requested tokens exceed context window")
        return {"usage": {"completion_tokens": 5}, "choices": [{"text": "  hello world \n"}]}


@pytest.fixture()
def client():
    # No `with` block, so the lifespan (which loads a real model) is not run.
    return TestClient(step5_api.app)


def test_generate_returns_metrics(client, monkeypatch):
    fake = FakeLlama()
    monkeypatch.setattr(step5_api, "llm", fake)
    r = client.post("/generate", json={"prompt": "hi", "max_tokens": 5, "temperature": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "hello world"
    assert body["tokens_generated"] == 5
    assert body["queue_wait_ms"] >= 0 and body["inference_ms"] >= 0
    assert fake.calls[0][1]["max_tokens"] == 5


def test_generate_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(step5_api, "llm", None)
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 503


def test_generate_422_when_prompt_too_long_for_context(client, monkeypatch):
    monkeypatch.setattr(step5_api, "llm", FakeLlama(raise_value_error=True))
    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == 422 and "context window" in r.json()["detail"]


@pytest.mark.parametrize("payload", [{"prompt": ""}, {"prompt": "x", "max_tokens": 0}, {"prompt": "x", "temperature": 5}])
def test_request_validation(client, monkeypatch, payload):
    monkeypatch.setattr(step5_api, "llm", FakeLlama())
    assert client.post("/generate", json=payload).status_code == 422


def test_health_reports_state(client, monkeypatch):
    monkeypatch.setattr(step5_api, "llm", None)
    assert client.get("/health").json()["status"] == "model_not_loaded"
    monkeypatch.setattr(step5_api, "llm", FakeLlama())
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["process_rss_mb"] > 0

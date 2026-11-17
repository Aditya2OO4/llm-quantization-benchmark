import json

import pytest
from fastapi.testclient import TestClient

import compare_checks as cc
import step8_compare as sc


# ---------- answer checkers ----------
def test_contains_check():
    assert cc.check_answer("capital", "The capital of France is Paris.")["passed"] is True
    assert cc.check_answer("capital", "It is Lyon.")["passed"] is False
    assert cc.check_answer("math_add", "17 + 25 = 42")["passed"] is True
    assert cc.check_answer("math_add", "The answer is 420")["passed"] is False  # word boundary


def test_no_check_for_free_form_or_unknown_preset():
    assert cc.check_answer("explain", "whatever") is None
    assert cc.check_answer(None, "whatever") is None
    assert cc.check_answer("nope", "whatever") is None


GOOD = "Sure!\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\nHope that helps."
WRONG = "```python\ndef reverse_string(s):\n    return s\n```"


def test_python_check_pass_and_fail():
    assert cc.check_answer("code_reverse", GOOD)["passed"] is True
    res = cc.check_answer("code_reverse", WRONG)
    assert res["passed"] is False and "reverse_string('hello')" in res["detail"]


def test_python_check_handles_unclosed_fence_and_missing_fence():
    assert cc.check_answer("code_reverse", "```python\ndef reverse_string(s):\n    return s[::-1]")["passed"] is True
    assert cc.check_answer("code_reverse", "Here: def reverse_string(s):\n    return s[::-1]\n")["passed"] is True


def test_python_check_reports_syntax_error_and_missing_function():
    assert "did not run" in cc.check_answer("code_reverse", "```python\ndef reverse_string(s) return s\n```")["detail"]
    assert cc.check_answer("code_reverse", "```python\ndef other(s):\n    return s\n```")["passed"] is False


@pytest.mark.parametrize("code", [
    "import os\ndef reverse_string(s):\n    return s[::-1]",
    "def reverse_string(s):\n    open('x.txt','w').write('hi')\n    return s[::-1]",
    "def reverse_string(s):\n    return eval('s[::-1]')",
    "def reverse_string(s):\n    return s.__class__ and s[::-1]",
])
def test_python_check_refuses_risky_code_without_running_it(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = cc.check_answer("code_reverse", f"```python\n{code}\n```")
    assert res["passed"] is False and res["detail"].startswith("not run")
    assert not (tmp_path / "x.txt").exists()


def test_python_check_times_out_on_infinite_loop():
    res = cc.check_answer("code_reverse", "```python\ndef reverse_string(s):\n    while True:\n        pass\n```", timeout=1.0)
    assert res["passed"] is False and "timed out" in res["detail"]


def test_public_presets_hide_the_answers():
    for p in cc.public_presets():
        assert set(p) == {"id", "title", "question", "max_tokens"}


# ---------- streaming ----------
class FakeLlama:
    def __init__(self, tokens, fail=False):
        self.tokens, self.fail, self.closed = tokens, fail, False

    def __call__(self, prompt, **kw):
        assert kw["stream"] and kw["temperature"] == 0.0 and "<|user|>" in prompt
        if self.fail:
            raise RuntimeError("boom")
        for t in self.tokens:
            yield {"choices": [{"text": t, "finish_reason": None}]}
        yield {"choices": [{"text": "", "finish_reason": "stop"}]}  # marker, must not be counted

    def close(self):
        self.closed = True


@pytest.fixture()
def two_models(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    for name in ("tinyllama-f16.gguf", "tinyllama-q4km.gguf"):
        (tmp_path / name).write_bytes(b"x" * 2048)


def _events(client, **body):
    body.setdefault("question", "What is the capital of France? Answer in one sentence.")
    with client.stream("POST", "/api/compare", json=body) as r:
        assert r.status_code == 200
        return [json.loads(line) for line in r.iter_lines() if line]


def test_compare_streams_models_in_order_with_stats(two_models, monkeypatch):
    made = []
    monkeypatch.setattr(sc, "load_model", lambda p, t: made.append(FakeLlama(["Paris", " is", " it"])) or made[-1])
    ev = _events(TestClient(sc.app), preset_id="capital")
    types = [(e["type"], e.get("label")) for e in ev]
    assert types[0][0] == "run_start" and [m["label"] for m in ev[0]["models"]] == ["F16", "Q4_K_M"]
    assert types[1:4] == [("model_start", "F16"), ("loaded", "F16"), ("token", "F16")]
    assert types[-1] == ("run_done", None)
    done = [e for e in ev if e["type"] == "model_done"]
    assert [d["tokens"] for d in done] == [3, 3]  # finish marker not counted
    assert all(d["check"]["passed"] for d in done)
    assert "".join(e["text"] for e in ev if e["type"] == "token" and e["label"] == "F16") == "Paris is it"
    assert all(m.closed for m in made)  # each model freed before the next


def test_compare_respects_label_selection_and_reports_load_errors(two_models, monkeypatch):
    def loader(path, threads):
        if "q4km" in path.name:
            raise OSError("out of memory")
        return FakeLlama(["a"])

    monkeypatch.setattr(sc, "load_model", loader)
    ev = _events(TestClient(sc.app), labels=["Q4_K_M", "F16"])
    err = next(e for e in ev if e["type"] == "error")
    assert err["label"] == "Q4_K_M" and "out of memory" in err["message"]
    assert any(e["type"] == "model_done" and e["label"] == "F16" for e in ev)  # run continued
    assert ev[-1]["type"] == "run_done"


def test_compare_generation_error_is_isolated(two_models, monkeypatch):
    llms = iter([FakeLlama([], fail=True), FakeLlama(["ok"])])
    monkeypatch.setattr(sc, "load_model", lambda p, t: next(llms))
    ev = _events(TestClient(sc.app))
    assert [e["type"] for e in ev if e["type"] in ("error", "model_done")] == ["error", "model_done"]


def test_second_run_is_rejected_while_one_is_active(two_models, monkeypatch):
    monkeypatch.setattr(sc, "load_model", lambda p, t: FakeLlama(["x"]))
    assert sc._run_lock.acquire(blocking=False)
    try:
        ev = _events(TestClient(sc.app))
    finally:
        sc._run_lock.release()
    assert ev == [{"type": "error", "label": None, "message": "Another comparison is already running."}]
    assert len(_events(TestClient(sc.app))) > 1  # lock was released, so a normal run works again


def test_no_models_gives_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    ev = _events(TestClient(sc.app))
    assert ev[0]["type"] == "error" and "step 3" in ev[0]["message"]


def test_static_page_and_helper_endpoints(two_models):
    c = TestClient(sc.app)
    assert "Quantization side-by-side" in c.get("/").text
    assert [m["label"] for m in c.get("/api/models").json()] == ["F16", "Q4_K_M"]
    assert any(p["id"] == "code_reverse" for p in c.get("/api/presets").json())
    assert c.post("/api/compare", json={"question": ""}).status_code == 422

"""Unit tests for OllamaClient -- streaming/parsing logic with the HTTP call faked.
The real streamed generation is checked manually against Ollama.
"""
import json

from rag.config import settings
from rag.llm.ollama_client import OllamaClient, build_options
from rag.models import Message


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


class _FakeStreamCtx:
    """Mimics the context manager returned by httpx.stream(...)."""
    def __init__(self, lines):
        self._response = _FakeStreamResponse(lines)

    def __enter__(self):
        return self._response

    def __exit__(self, *args):
        return False


def test_chat_stream_yields_content_deltas_and_captures_stats(monkeypatch):
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "To "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "roll "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "back."}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True,
                    "prompt_eval_count": 42, "eval_count": 3}),
    ]
    captured = {}

    def fake_stream(method, url, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = json
        return _FakeStreamCtx(lines)

    monkeypatch.setattr("rag.llm.ollama_client.httpx.stream", fake_stream)

    client = OllamaClient()
    out = list(client.chat_stream([Message(role="user", content="how?")], {"num_ctx": 6000}))

    assert out == ["To ", "roll ", "back."]                  # deltas only; empty final skipped
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["messages"] == [{"role": "user", "content": "how?"}]
    assert captured["payload"]["options"] == {"num_ctx": 6000}
    # stats from the final done object are captured for budget verification
    assert client.last_prompt_eval_count == 42
    assert client.last_eval_count == 3


def test_build_options_sets_the_6k_guard_params():
    options = build_options(system_tokens=123)
    assert options["num_ctx"] == settings.num_ctx
    assert options["num_predict"] == settings.num_predict
    assert options["num_keep"] == 123                        # = system_tokens, pins the system prompt
    assert options["temperature"] == settings.temperature

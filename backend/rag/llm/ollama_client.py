"""LLMClient -- Ollama /api/chat, streaming.

Interface seam: orchestration depends only on LLMClient; tests inject a FakeLLM.
Streaming: /api/chat returns a sequence of JSON objects, each carrying a message.content
delta; the final object has done:true and includes prompt_eval_count / eval_count (used
to verify the 6K token estimate).
"""

import json
from typing import Iterator, Protocol

import httpx

from rag.config import settings
from rag.models import Message


class LLMClient(Protocol):
    def chat_stream(self, messages: list[Message], options: dict) -> Iterator[str]: ...


def build_options(system_tokens: int) -> dict:
    """Build the /api/chat options -- the three parameters of the 6K guard."""
    return {
        "num_ctx": settings.num_ctx,          # 6000, lock the window explicitly
        "num_predict": settings.num_predict,  # reserve output space
        "num_keep": system_tokens,            # pin the system prompt against front-truncation
        "temperature": settings.temperature,
    }


class OllamaClient:
    """Production implementation: forward Ollama's stream."""

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = host or settings.ollama_host
        self.model = model or settings.llm_model
        # Filled in from the final streamed object -- lets the caller verify the token
        # estimate against what Ollama actually counted.
        self.last_prompt_eval_count: int | None = None   # tokens Ollama actually read (input)
        self.last_eval_count: int | None = None          # tokens generated (output)

    def chat_stream(self, messages: list[Message], options: dict) -> Iterator[str]:
        """POST /api/chat with stream=True and yield each content delta as it arrives.

        Ollama streams newline-delimited JSON objects: each has message.content (a piece of
        the answer); the final one has done=true plus prompt_eval_count / eval_count.
        """
        message_dicts = []
        for message in messages:
            message_dicts.append({"role": message.role, "content": message.content})

        payload = {
            "model": self.model,
            "messages": message_dicts,
            "stream": True,
            "options": options,
        }

        with httpx.stream("POST", f"{self.host}/api/chat", json=payload, timeout=120.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)

                content = obj.get("message", {}).get("content", "")
                if content:
                    yield content

                if obj.get("done"):
                    self.last_prompt_eval_count = obj.get("prompt_eval_count")
                    self.last_eval_count = obj.get("eval_count")

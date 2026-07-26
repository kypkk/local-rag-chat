"""Test fakes -- fake implementations of the interfaces.

Inject these -> pipeline / orchestration tests need no Ollama, are fast, and are reproducible.
"""

from typing import Iterator

import numpy as np

from rag.models import Chunk, Hit, Message


class FakeEmbedder:
    """Returns fixed-dimension fake vectors (contents don't matter; for testing orchestration)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        # deterministic vector seeded by length, normalized
        v = np.ones(self.dim) * (len(text) % 7 + 1)
        return v / np.linalg.norm(v)


class FakeStore:
    """Returns pre-seeded Hits regardless of the query (to test whether the right chunks
    end up in the messages)."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self._hits = hits or []

    def add(self, chunks: list[Chunk]) -> None:
        self._hits = [Hit(chunk=c, score=1.0) for c in chunks]

    def search(self, query_vec, k: int, threshold: float) -> list[Hit]:
        return self._hits[:k]

    def clear(self) -> None:
        self._hits = []

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def __len__(self) -> int:
        return len(self._hits)


class FakeLLM:
    """Records the messages it receives (for assertions) and echoes a fixed token stream."""

    def __init__(self) -> None:
        self.last_messages: list[Message] | None = None
        self.last_options: dict | None = None

    def chat_stream(self, messages: list[Message], options: dict) -> Iterator[str]:
        self.last_messages = messages
        self.last_options = options
        yield "ok"

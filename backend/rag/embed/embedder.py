"""Embedder -- text -> vectors.

Interface seam: other modules depend only on the Embedder Protocol, not on Ollama.
Correctness:
  - documents get the search_document: prefix, queries get search_query:, same model on both sides
  - the prefix is encapsulated here so callers never hand-assemble strings
    (which would silently degrade retrieval quality)
  - vectors are returned L2-normalized -> cosine = dot product downstream
"""

from typing import Protocol

import httpx
import numpy as np

from rag.config import settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Divide by L2 norm -> unit sphere."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class OllamaEmbedder:
    """Production implementation: via Ollama /api/embed (nomic-embed-text)."""

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = host or settings.ollama_host
        self.model = model or settings.embed_model

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Prefix each text with search_document:, embed, return normalized vectors."""
        prefixed = []
        for text in texts:
            prefixed.append(settings.embed_prefix_document + text)
        return self._embed(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """Prefix the query with search_query:, embed, return its normalized vector."""
        prefixed = settings.embed_prefix_query + text
        return self._embed([prefixed])[0]

    def _embed(self, inputs: list[str]) -> list[np.ndarray]:
        """POST the inputs to Ollama /api/embed and return one L2-normalized vector each.

        The response shape is {"embeddings": [[...], ...]} -- one vector per input.
        """
        response = httpx.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": inputs},
            timeout=60.0,   # generous: the very first call may need to load the model
        )
        response.raise_for_status()
        raw_vectors = response.json()["embeddings"]

        vectors: list[np.ndarray] = []
        for raw in raw_vectors:
            vectors.append(l2_normalize(np.array(raw, dtype=float)))
        return vectors

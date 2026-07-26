"""Unit tests for OllamaEmbedder -- the prefix + normalization logic, with the HTTP call
faked (no Ollama needed). The real end-to-end embed is checked manually against Ollama.
"""
import numpy as np

from rag.embed.embedder import OllamaEmbedder, l2_normalize


class _FakeResponse:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def raise_for_status(self):
        pass

    def json(self):
        return {"embeddings": self._embeddings}


def _fake_post_returning(embeddings, captured):
    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["input"] = json["input"]
        return _FakeResponse(embeddings)
    return fake_post


def test_embed_documents_adds_document_prefix_and_normalizes(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "rag.embed.embedder.httpx.post",
        _fake_post_returning([[3.0, 4.0], [0.0, 5.0]], captured),
    )

    vectors = OllamaEmbedder().embed_documents(["hello", "world"])

    # each input got the document prefix
    assert captured["input"] == ["search_document: hello", "search_document: world"]
    assert captured["url"].endswith("/api/embed")
    # [3, 4] normalizes to [0.6, 0.8] (norm 5); every vector is unit length
    assert np.allclose(vectors[0], [0.6, 0.8])
    assert all(np.isclose(np.linalg.norm(v), 1.0) for v in vectors)


def test_embed_query_adds_query_prefix_and_returns_single_vector(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "rag.embed.embedder.httpx.post",
        _fake_post_returning([[1.0, 0.0]], captured),
    )

    vec = OllamaEmbedder().embed_query("how to roll back")

    assert captured["input"] == ["search_query: how to roll back"]   # single, prefixed, sent as a list
    assert vec.shape == (2,)                                          # unwrapped from the list
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_l2_normalize_handles_zero_vector():
    # a zero vector has no direction -> return it unchanged instead of dividing by zero
    zero = np.zeros(3)
    assert np.array_equal(l2_normalize(zero), zero)

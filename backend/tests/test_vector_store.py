"""Unit tests for NumpyStore -- pure math, tested with fake normalized vectors (no Ollama)."""
import numpy as np
import pytest

from rag.models import Chunk
from rag.store.vector_store import NumpyStore


def _chunk(chunk_id: str, vec: list[float]) -> Chunk:
    v = np.array(vec, dtype=float)
    v = v / np.linalg.norm(v)                       # the store assumes normalized vectors
    return Chunk(id=chunk_id, text=chunk_id, source="d.md", embedding=v)


def _query(vec: list[float]) -> np.ndarray:
    v = np.array(vec, dtype=float)
    return v / np.linalg.norm(v)


def test_search_returns_nearest_first():
    store = NumpyStore()
    store.add([
        _chunk("exact", [1, 0, 0]),
        _chunk("orthogonal", [0, 1, 0]),
        _chunk("close", [0.9, 0.1, 0]),
    ])

    hits = store.search(_query([1, 0, 0]), k=3, threshold=-1.0)

    assert [h.chunk.id for h in hits] == ["exact", "close", "orthogonal"]
    assert hits[0].score >= hits[1].score >= hits[2].score   # descending


def test_threshold_filters_low_scores():
    store = NumpyStore()
    store.add([_chunk("near", [1, 0, 0]), _chunk("far", [0, 0, 1])])

    hits = store.search(_query([1, 0, 0]), k=5, threshold=0.5)

    assert [h.chunk.id for h in hits] == ["near"]            # "far" (cosine ~0) dropped


def test_top_k_caps_the_number_of_hits():
    store = NumpyStore()
    store.add([_chunk(str(i), [1, i * 0.01, 0]) for i in range(10)])

    hits = store.search(_query([1, 0, 0]), k=3, threshold=-1.0)

    assert len(hits) == 3


def test_save_load_roundtrip(tmp_path):
    store = NumpyStore()
    store.add([_chunk("a", [1, 0, 0]), _chunk("b", [0, 1, 0])])
    path = str(tmp_path / "store.pkl")
    store.save(path)

    reloaded = NumpyStore()
    reloaded.load(path)

    assert len(reloaded) == 2
    top = reloaded.search(_query([1, 0, 0]), k=1, threshold=-1.0)
    assert top[0].chunk.id == "a"


def test_add_requires_embeddings():
    store = NumpyStore()
    with pytest.raises(ValueError):
        store.add([Chunk(id="x", text="x", source="d.md")])   # no embedding set


def test_empty_store_search_returns_nothing():
    assert NumpyStore().search(_query([1, 0, 0]), k=3, threshold=-1.0) == []

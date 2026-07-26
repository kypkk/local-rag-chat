"""VectorStore -- storage layer + similarity search.

Interface seam: this is a vector *store* (storage layer), not a vector *database*
(standalone service). NumpyStore uses exact/flat search -- the very same computation a
vector DB does internally at small scale, just without the extra service. To scale, swap this implementation for an ANN backend (FAISS/HNSW) and
the retriever/orchestration do not change.
"""

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np

from rag.models import Chunk, Hit


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query_vec: np.ndarray, k: int, threshold: float) -> list[Hit]: ...
    def clear(self) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def __len__(self) -> int: ...


class NumpyStore:
    """Production implementation: stack all chunk vectors into an (N x dim) matrix and
    do a single matmul per query."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None   # (N x dim), already normalized

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks (which must already carry L2-normalized embeddings) and rebuild
        the matrix of stacked vectors."""
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.id} has no embedding; embed it before adding")
            self._chunks.append(chunk)
        self._matrix = np.vstack([chunk.embedding for chunk in self._chunks])

    def clear(self) -> None:
        """Drop all chunks -- so a re-ingest replaces the index instead of appending."""
        self._chunks = []
        self._matrix = None

    def search(self, query_vec: np.ndarray, k: int, threshold: float) -> list[Hit]:
        """Brute-force exact cosine: one matmul scores every chunk, then take the k
        highest and drop any that fall below the threshold -- so fewer than k hits come
        back when the query matches little. Assumes query_vec and the stored vectors are
        L2-normalized, so the dot product is the cosine similarity.
        """
        if self._matrix is None or not self._chunks:
            return []

        scores = self._matrix @ query_vec              # (N,) cosine similarity, all at once
        top_indices = np.argsort(scores)[::-1][:k]     # indices of the k highest, descending

        hits: list[Hit] = []
        for index in top_indices:
            score = float(scores[index])
            if score < threshold:
                continue                               # below threshold -> "not relevant enough"
            hits.append(Hit(chunk=self._chunks[index], score=score))
        return hits

    def save(self, path: str) -> None:
        """Persist the chunks + matrix to disk. Pickle is fine here: we only ever load an
        index this project wrote itself, never untrusted input."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump({"chunks": self._chunks, "matrix": self._matrix}, file)

    def load(self, path: str) -> None:
        """Load chunks + matrix previously written by save()."""
        with open(path, "rb") as file:
            data = pickle.load(file)
        self._chunks = data["chunks"]
        self._matrix = data["matrix"]

    def __len__(self) -> int:
        return len(self._chunks)

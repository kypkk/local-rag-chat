"""Retriever -- turn a question into top-k chunks.

A thin orchestration layer: embed query -> store.search -> return Hits.
Depends on the Embedder / VectorStore interfaces, not on concrete implementations.
"""

from rag.config import settings
from rag.embed.embedder import Embedder
from rag.models import Hit
from rag.store.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int | None = None,
                 threshold: float | None = None) -> list[Hit]:
        """Embed the query (with the search_query: prefix) and return the top-k Hits.

        k / threshold fall back to config. We use `is not None` (not `or`) so an explicit
        0 is respected rather than treated as "unset".
        """
        query_vec = self.embedder.embed_query(query)
        return self.store.search(
            query_vec,
            k if k is not None else settings.top_k,
            threshold if threshold is not None else settings.score_threshold,
        )

"""Shared data types -- the things passed between pipeline stages.

Deliberately stdlib dataclasses: zero dependencies, readable.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Document:
    """A raw document read by the loader (upstream)."""
    source: str            # file name / relative path
    text: str              # plain-text content


@dataclass
class Chunk:
    """The unit of retrieval -- simultaneously the unit of embedding, retrieval,
    and context.

    The metadata (source + headers) forms the citation string in the answer,
    e.g. "[1] (deploy.md · Rollback)".
    """
    id: str                        # stable unique id, e.g. "deploy.md::Rollback::1"
    text: str                      # clean body -- injected into the prompt and displayed
    source: str                    # source file name
    headers: list[str] = field(default_factory=list)   # header path, e.g. ["Deploy Guide", "Rollback"]
    chunk_idx: int = 0             # index when an oversized section is split into several chunks
    n_chunks: int = 1
    # Char span of this chunk in the source document. Provenance metadata, kept so a client
    # could later highlight the exact passage; the current UI cites by breadcrumb instead.
    # Best-effort: the splitter strips headers and reflows whitespace, so the span is found
    # by a whitespace-tolerant search; -1 means it could not be located.
    start_char: int = -1
    end_char: int = -1
    n_tokens: int = 0              # tokens of `text` (for the 6K budget) -- excludes the breadcrumb
    kind: Literal["prose", "table", "code"] = "prose"  # atomic-unit marker
    # What the embedder actually embeds: the header breadcrumb + body, so a chunk whose
    # body doesn't repeat its section's topic words is still findable.
    # Decoupled from `text` so the breadcrumb helps retrieval without costing prompt budget.
    embed_text: str = ""
    embedding: np.ndarray | None = None                # L2-normalized vector; filled in by the embedder

    @property
    def citation(self) -> str:
        """Build the citation string, e.g. 'deploy.md · Rollback'."""
        path = " · ".join([self.source, *self.headers]) if self.headers else self.source
        return path


@dataclass
class Hit:
    """A single retrieval result."""
    chunk: Chunk
    score: float           # cosine similarity


@dataclass
class Message:
    """One turn of conversation history; history stores raw Q/A only,
    never the assembled prompt."""
    role: Literal["user", "assistant", "system"]
    content: str

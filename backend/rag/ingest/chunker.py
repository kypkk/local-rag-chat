"""Chunker -- Document -> list[Chunk], built on langchain-text-splitters.

Text splitting is a well-solved problem with many fiddly edge cases (nested
headers, code fences, whitespace, separator hierarchy), so we use a maintained
library rather than hand-rolling it.

Strategy:
  1. MarkdownHeaderTextSplitter -- split on #/##/### and keep the header path as
     metadata. That header path becomes the citation (e.g. "deploy.md · Rollback").
  2. RecursiveCharacterTextSplitter (markdown-aware, measured in tokens) -- split any
     oversized section into ~chunk_target_tokens pieces with overlap.
"""

import re

from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.config import settings
from rag.models import Chunk, Document
from rag.chat.prompt_builder import estimate_tokens  # share the same token estimator

# markdown heading levels we split on -> metadata keys, in nesting order
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_HEADER_KEYS = [key for _, key in HEADERS_TO_SPLIT_ON]


# ===========================================================================
# Public API
# ===========================================================================

def chunk_document(doc: Document) -> list[Chunk]:
    """Split one Document into a list of Chunks."""
    sections = _split_into_sections(doc.text)
    splitter = _make_recursive_splitter()

    chunks: list[Chunk] = []
    for header_path, body in sections:
        body = body.strip()
        if not body:
            continue

        # RecursiveCharacterTextSplitter.split_text(body) -> list[str]
        # (plain strings; one piece if the body is small, several if oversized)
        pieces = splitter.split_text(body)
        for index, piece in enumerate(pieces):
            start_char, end_char = _locate(doc.text, piece)
            chunk = Chunk(
                id=_make_chunk_id(doc.source, header_path, index),
                text=piece,
                source=doc.source,
                headers=header_path,
                chunk_idx=index,
                n_chunks=len(pieces),
                start_char=start_char,
                end_char=end_char,
                n_tokens=estimate_tokens(piece),
                kind=_classify_kind(piece),
                embed_text=_embed_text(header_path, piece),
            )
            chunks.append(chunk)

    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Batch: chunk each document and flatten."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks


# ===========================================================================
# Step 1: split into (header_path, body) sections
# ===========================================================================

def _split_into_sections(text: str) -> list[tuple[list[str], str]]:
    """Split on markdown headers; return (header_path, body) per section.

    Text with no headers comes back as a single section with an empty header path.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,  # the header text lives in metadata, not the body
    )

    # MarkdownHeaderTextSplitter.split_text(text) -> list[langchain Document]
    # (NOT our Document). Each has: .page_content (str, the body) and
    # .metadata (dict, e.g. {"h1": "Guide", "h2": "Rollback"}).
    sections: list[tuple[list[str], str]] = []
    for section in header_splitter.split_text(text):
        header_path = _header_path(section.metadata)
        sections.append((header_path, section.page_content))
    return sections


def _header_path(metadata: dict) -> list[str]:
    """Turn the splitter's metadata ({'h1': ..., 'h2': ...}) into an ordered path."""
    path: list[str] = []
    for key in _HEADER_KEYS:
        if key in metadata:
            path.append(metadata[key])
    return path


# ===========================================================================
# Step 2: split an oversized section into token-sized pieces (with overlap)
# ===========================================================================

def _make_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """A markdown-aware recursive splitter whose sizes are measured in tokens.

    length_function=estimate_tokens makes chunk_size/chunk_overlap count tokens
    (not characters), so they line up with the 6K budget in config.
    """
    overlap_tokens = int(settings.chunk_target_tokens * settings.chunk_overlap_ratio)
    return RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=settings.chunk_target_tokens,
        chunk_overlap=overlap_tokens,
        length_function=estimate_tokens,
    )


# ===========================================================================
# Small helpers
# ===========================================================================

def _make_chunk_id(source: str, header_path: list[str], index: int) -> str:
    """A stable, unique id, e.g. 'deploy.md::Deployment Guide::Rollback::0'."""
    if header_path:
        slug = "::".join(header_path)   # ["Guide", "Rollback"] -> "Guide::Rollback"
    else:
        slug = "root"                   # a section with no heading
    return f"{source}::{slug}::{index}"


def _locate(doc_text: str, chunk_text: str) -> tuple[int, int]:
    """Best-effort (start_char, end_char) of chunk_text within the original doc_text.

    The splitter strips headers and reflows whitespace, so chunk_text is NOT a verbatim
    slice of doc_text. We therefore match its words in order, tolerating any whitespace
    between them, and return the char span of the first match. Returns (-1, -1) if it
    cannot be located.
    """
    words = chunk_text.split()
    if not words:
        return (-1, -1)

    # Build a regex like  roll\s+back  that matches the words in order, with any
    # whitespace between them:
    #   - re.escape(word) makes punctuation literal (so "a.b" isn't treated as a regex)
    #   - "\s+" between the words matches any run of whitespace (spaces, newlines)
    escaped_words = []
    for word in words:
        escaped_words.append(re.escape(word))
    pattern = re.compile(r"\s+".join(escaped_words))

    match = pattern.search(doc_text)
    if match is None:
        return (-1, -1)
    return (match.start(), match.end())


def _embed_text(header_path: list[str], body: str) -> str:
    """The text to embed: the header breadcrumb prepended to the body, so the
    section's topic words are part of the vector even if the body omits them.

    Kept separate from the chunk's `text` (the clean body) so it improves retrieval
    without being injected into the prompt or costing budget.
    """
    if not header_path:
        return body
    breadcrumb = " > ".join(header_path)   # ["Guide", "Rollback"] -> "Guide > Rollback"
    return f"{breadcrumb}\n\n{body}"


def _classify_kind(text: str) -> str:
    """Tag the chunk as code / table / prose (best-effort, just metadata)."""
    if text.lstrip().startswith("```"):
        return "code"

    # A markdown table separator row "| --- | --- |" becomes "|---|---|" once spaces are
    # removed, which contains the tell-tale "-|-" at a column boundary.
    text_without_spaces = text.replace(" ", "")
    if "|" in text and "-|-" in text_without_spaces:
        return "table"

    return "prose"

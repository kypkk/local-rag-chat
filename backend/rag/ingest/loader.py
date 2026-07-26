"""Loader -- .md/.txt -> Document (upstream).

Responsibility: read files into plain text, keeping the source metadata (needed
later to cite sources). Deterministic and dependency-free -> good to write and
unit-test first.
"""

from pathlib import Path

from rag.models import Document

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(docs_dir: str) -> list[Document]:
    """Scan a folder recursively and read each .md/.txt file into a Document.

    Files are returned in a stable (sorted) order so downstream chunk ids and
    tests are reproducible.
    """
    base = Path(docs_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"docs_dir does not exist or is not a directory: {docs_dir}"
        )

    files = []
    for path in base.rglob("*"):  # walks every file/dir at any depth
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    files.sort()  # deterministic, reproducible order

    documents = []
    for path in files:
        documents.append(load_file(path, docs_dir))
    return documents


def load_file(path: Path, docs_dir: str) -> Document:
    """Read a single file into a Document (helper for load_documents).

    The source is the path relative to docs_dir, using POSIX separators so the
    citation string is stable across operating systems.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Tolerate the occasional non-UTF-8 byte rather than failing the whole ingest.
        text = path.read_text(encoding="utf-8", errors="replace")

    source = path.relative_to(docs_dir).as_posix()
    return Document(source=source, text=text)

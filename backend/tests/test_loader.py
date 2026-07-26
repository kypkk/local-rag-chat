"""Unit tests for the loader -- deterministic, no external services."""
from pathlib import Path

import pytest

from rag.ingest.loader import load_documents
from rag.models import Document

SAMPLE_DIR = str(Path(__file__).resolve().parent.parent / "sample_docs")


def test_loads_sample_docs():
    docs = load_documents(SAMPLE_DIR)
    sources = {d.source for d in docs}
    # the known docs must load; tolerate extra files being added to the corpus
    assert {"architecture.md", "usage.md"} <= sources
    assert all(isinstance(d, Document) for d in docs)
    # content is actually read, not just the filename
    usage = next(d for d in docs if d.source == "usage.md")
    assert "Endpoints" in usage.text


def test_stable_sorted_order(tmp_path):
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.md").write_text("a")
    docs = load_documents(str(tmp_path))
    assert [d.source for d in docs] == ["a.md", "b.md"]


def test_ignores_unsupported_and_recurses(tmp_path):
    (tmp_path / "keep.txt").write_text("hi")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.md").write_text("deep")
    docs = load_documents(str(tmp_path))
    assert {d.source for d in docs} == {"keep.txt", "nested/deep.md"}


def test_suffix_match_is_case_insensitive(tmp_path):
    # the brief literally says ".MD and .TXT files"
    (tmp_path / "UPPER.MD").write_text("x")
    docs = load_documents(str(tmp_path))
    assert [d.source for d in docs] == ["UPPER.MD"]


def test_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_documents("definitely_not_a_real_dir_xyz")

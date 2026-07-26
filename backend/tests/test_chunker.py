"""Unit tests for the chunker (langchain-text-splitters based) -- deterministic, no LLM."""
from rag.chat.prompt_builder import estimate_tokens
from rag.ingest.chunker import chunk_document, _split_into_sections
from rag.models import Document


def test_markdown_headers_become_metadata():
    doc = Document(
        source="deploy.md",
        text="# Guide\nintro line\n\n## Rollback\nto roll back, redeploy the last release.\n",
    )
    chunks = chunk_document(doc)

    # the nested header path is captured (used later for citations)
    assert any(c.headers == ["Guide", "Rollback"] for c in chunks)
    assert all(c.source == "deploy.md" for c in chunks)
    assert all(c.id.startswith("deploy.md::") for c in chunks)


def test_txt_without_headers_still_chunks():
    doc = Document(source="notes.txt", text="first paragraph.\n\nsecond paragraph.")
    chunks = chunk_document(doc)

    assert len(chunks) >= 1
    assert chunks[0].headers == []           # no headings -> empty header path
    assert "first paragraph." in chunks[0].text


def test_oversized_section_is_split_with_overlap():
    # 40 distinct, sizable paragraphs -> the section far exceeds the token target
    paragraphs = [f"Paragraph {i}: " + "lorem ipsum dolor sit amet " * 8 for i in range(40)]
    doc = Document(source="big.txt", text="# Big\n\n" + "\n\n".join(paragraphs))

    chunks = chunk_document(doc)

    assert len(chunks) > 1                    # it was actually split
    # overlap: the tail of one chunk reappears in the next
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.text[-40:] in later.text


def test_small_code_block_stays_whole():
    # a code block that fits under the token target is not broken up
    code_block = "```python\nx = 1\ny = 2\nz = x + y\n```"
    doc = Document(source="c.md", text="# Code\n\n" + code_block)

    chunks = chunk_document(doc)

    code_chunks = [c for c in chunks if "```" in c.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.count("```") == 2   # fences balanced, not cut in half
    assert code_chunks[0].kind == "code"


def test_embed_text_prepends_header_breadcrumb():
    doc = Document(
        source="deploy.md",
        text="# Guide\n\n## Rollback\n\n### Emergency rollback\npage the on-call engineer.\n",
    )
    chunk = next(c for c in chunk_document(doc) if c.headers == ["Guide", "Rollback", "Emergency rollback"])

    # the embedded text carries the topic words even though the body omits them
    assert chunk.embed_text.startswith("Guide > Rollback > Emergency rollback")
    assert "page the on-call engineer." in chunk.embed_text
    # the clean body (injected into the prompt) does NOT carry the breadcrumb
    assert chunk.text == "page the on-call engineer."
    # the breadcrumb does not inflate the budget count
    assert chunk.n_tokens == estimate_tokens(chunk.text)


def test_char_offsets_locate_chunk_in_source():
    doc = Document(
        source="deploy.md",
        text="# Guide\n\n## Rollback\nto roll back, redeploy the last release.\n",
    )
    chunk = next(c for c in chunk_document(doc) if c.headers == ["Guide", "Rollback"])

    assert chunk.start_char >= 0 and chunk.end_char > chunk.start_char
    # the span pulled from the ORIGINAL text matches the chunk body (whitespace-normalized)
    span = doc.text[chunk.start_char:chunk.end_char]
    assert " ".join(span.split()) == " ".join(chunk.text.split())


def test_header_inside_code_block_is_not_treated_as_heading():
    # a '#' comment inside a fenced code block must not start a new section
    sections = _split_into_sections("# Real Heading\n\n```py\n# not a heading\nx = 1\n```\n")

    header_paths = [headers for headers, _ in sections]
    assert ["Real Heading"] in header_paths
    assert ["not a heading"] not in header_paths

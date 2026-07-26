"""Integration tests for RagService -- orchestration wired with FAKE components (no Ollama)."""
from rag.config import settings
from rag.models import Chunk, Hit
from rag.service import RagService
from rag.store.vector_store import NumpyStore
from tests.fakes import FakeEmbedder, FakeLLM, FakeStore


def test_answer_retrieves_and_puts_the_chunk_in_the_prompt():
    chunk = Chunk(id="c1", text="To roll back, re-deploy the previous release.",
                  source="deploy.md", headers=["Rollback"])
    store = FakeStore([Hit(chunk=chunk, score=0.9)])
    llm = FakeLLM()
    service = RagService(embedder=FakeEmbedder(), store=store, llm=llm)

    events = list(service.answer("how do I roll back?", []))

    # answer() now yields typed events: token deltas + a sources bibliography
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["ok"]                                # FakeLLM echoes one token
    sources = [e for e in events if e["type"] == "sources"]
    assert len(sources) == 1                               # exactly one sources event, up front
    assert events[0]["type"] == "sources"                  # ...and it comes before the tokens
    # the retrieved chunk is offered as a resolvable source (breadcrumb from its metadata)
    assert any("deploy.md" in item["label"] for item in sources[0]["items"])
    # the retrieved chunk reached the final user message, and system is first
    assert llm.last_messages[0].role == "system"
    assert llm.last_messages[-1].role == "user"
    assert "To roll back, re-deploy the previous release." in llm.last_messages[-1].content
    # the 6K options were passed through (track config, don't hard-code the value)
    assert llm.last_options["num_ctx"] == settings.num_ctx


def test_ingest_loads_chunks_and_fills_the_store(tmp_path):
    (tmp_path / "doc.md").write_text("# Guide\n\n## Rollback\nre-deploy the previous release.\n")
    store = FakeStore()
    service = RagService(embedder=FakeEmbedder(), store=store, llm=FakeLLM())

    n = service.ingest(str(tmp_path))

    assert n >= 1
    assert len(store) == n


def test_reingest_replaces_and_does_not_append(tmp_path, monkeypatch):
    # a real NumpyStore (whose add() APPENDS) would double on re-ingest without clear()
    monkeypatch.setattr(NumpyStore, "save", lambda self, path: None)   # don't touch the real index file
    (tmp_path / "doc.md").write_text("# Guide\n\n## Rollback\nre-deploy the previous release.\n")
    store = NumpyStore()
    service = RagService(embedder=FakeEmbedder(), store=store, llm=FakeLLM())

    first = service.ingest(str(tmp_path))
    second = service.ingest(str(tmp_path))

    assert first == second == len(store)   # re-ingest replaced the index, didn't double it

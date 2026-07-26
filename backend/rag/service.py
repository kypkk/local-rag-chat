"""RagService -- the orchestration layer.

Wires the components into the two pipelines. This is the home of answer() / ingest(),
i.e. the "heart". Endpoints (api/server.py) only call in here; business
logic never lives in a route. Components are injected via interfaces -> inject real ones
in production, fakes in tests.
"""

import re
from typing import Iterator

from rag.config import settings
from rag.embed.embedder import Embedder
from rag.llm.ollama_client import LLMClient, build_options
from rag.models import Chunk, Message
from rag.retrieve.retriever import Retriever
from rag.store.vector_store import VectorStore
from rag.chat import history as history_mod
from rag.chat import prompt_builder
from rag.ingest import loader, chunker


class RagService:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLMClient) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.retriever = Retriever(embedder, store)

    # -- ingestion pipeline (offline) -----------------------------
    def ingest(self, docs_dir: str | None = None) -> int:
        """Read files -> chunk -> embed -> add to the store, then persist. Returns chunk count.

        Clears the store first so a re-ingest REPLACES the index (rather than appending),
        making it safe to call repeatedly, e.g. via POST /ingest after editing the docs.
        """
        self.store.clear()
        documents = loader.load_documents(docs_dir or settings.docs_dir)
        chunks = chunker.chunk_documents(documents)

        # embed the breadcrumb+body (embed_text), then attach each vector to its chunk
        texts_to_embed = []
        for chunk in chunks:
            texts_to_embed.append(chunk.embed_text)
        embeddings = self.embedder.embed_documents(texts_to_embed)
        for chunk, vector in zip(chunks, embeddings):
            chunk.embedding = vector

        self.store.add(chunks)
        self.store.save(settings.index_path)
        return len(chunks)

    # -- query pipeline (online) ----------------------------------
    def answer(self, question: str, history: list[Message]) -> Iterator[dict]:
        """The full lifecycle of one question, streamed as typed events.

        Yields two kinds of events (the client switches on "type"):
          {"type": "sources", "items": [{"n": 1, "label": "<breadcrumb>"}, ...]}  -- once, first
          {"type": "token",   "text": "<delta>"}                                  -- many, streamed
        Sources are metadata, NOT answer text: the model cites [n] inline and the client
        resolves each [n] to the human-readable breadcrumb -- and because they never enter the
        answer string, they never leak into the conversation history that feeds the next turn.
        """
        trimmed_history, _ = history_mod.trim_history(history)
        # RETRIEVE with a standalone rewrite (resolves "it"/"that"); GENERATE with the
        # original question + history so the model still answers in context.
        search_query = history_mod.condense_question(trimmed_history, question, self.llm)
        hits = self.retriever.retrieve(search_query)
        messages, system_tokens, kept = prompt_builder.build_messages(question, trimmed_history, hits)
        options = build_options(system_tokens)

        # The [1..N] bibliography, numbered to match render_context, so the model's inline
        # [n] resolves to items[n-1]. We send every kept chunk; the client shows only the
        # ones the answer actually cites.
        sources = [{"n": i, "label": _source_label(hit.chunk)} for i, hit in enumerate(kept, start=1)]
        yield {"type": "sources", "items": sources}

        # Our pre-send estimate of the input size (the chars_per_token heuristic).
        estimated_input_tokens = sum(
            prompt_builder.estimate_tokens(message.content) for message in messages
        )

        for token in self.llm.chat_stream(messages, options):
            yield {"type": "token", "text": token}

        # The stream is now finished, so Ollama has reported the REAL token counts on its
        # final object. Print estimate-vs-actual so we can see how far chars_per_token=4.0 is
        # off (getattr keeps the FakeLLM in tests happy -- it carries no such attribute).
        actual_input_tokens = getattr(self.llm, "last_prompt_eval_count", None)
        actual_output_tokens = getattr(self.llm, "last_eval_count", None)
        if actual_input_tokens is not None:
            _report_token_accounting(estimated_input_tokens, actual_input_tokens, actual_output_tokens)


def _source_label(chunk: Chunk) -> str:
    """Turn a chunk's metadata into a human-readable source breadcrumb, e.g.
    'architecture.md > Ingestion > Loader'.

    The markdown splitter can keep section numbering on headers ("2. Ingestion"),
    which is noise in a citation, so strip a leading "N. " from each header segment.
    """
    parts = [chunk.source]
    for header in chunk.headers:
        parts.append(re.sub(r"^\d+\.\s*", "", header))
    return " > ".join(parts)


def _report_token_accounting(estimated_input: int, actual_input: int, actual_output: int | None) -> None:
    """Print a one-line comparison of our token estimate against Ollama's real counts.

    Dev instrumentation (prints to the uvicorn console). A ratio > 1 means we OVER-estimated
    the input, which is the safe direction: we reserve more room than needed and never let
    the prompt silently overflow num_ctx. output_tokens is what the model actually generated.
    """
    output = actual_output or 0
    total_actual = actual_input + output
    ratio = estimated_input / actual_input if actual_input else float("nan")
    print(
        f"[token-accounting] input: estimated {estimated_input} vs actual {actual_input} "
        f"(est/actual={ratio:.2f}, off by {estimated_input - actual_input:+d})  |  "
        f"output {output}  |  total {total_actual}/{settings.num_ctx} ctx",
        flush=True,
    )

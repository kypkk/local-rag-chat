# Local RAG Chat

A **fully-local** RAG-enabled chat backend: the LLM and the entire RAG pipeline run
on the local machine, the context window is capped at **6K tokens**, and it ingests
`.md` / `.txt` documents.

> Status: **complete**. All modules are implemented and covered by tests
> (`cd backend && uv run pytest`), and the app runs end-to-end against a local Ollama —
> grounded, cited, streaming answers, with a single-file streaming frontend.

---

## Architecture

Monorepo: `backend/` (the RAG server) + `frontend/` (the chat UI). Two pipelines
(offline ingestion + online query); module boundaries map to the pipeline stages:

```
local-rag-chat/            # git repo root
  backend/
    rag/
      ingest/    loader.py  chunker.py        # files -> chunks
      embed/     embedder.py                  # text -> vectors (Embedder interface; nomic prefix)
      store/     vector_store.py              # interface + numpy exact search (persisted)
      retrieve/  retriever.py                 # top-k + threshold
      chat/      prompt_builder.py  history.py   # 6K budget, sliding-window history
      llm/       ollama_client.py             # /api/chat streaming (LLMClient interface)
      service.py                              # orchestration: answer() / ingest()
      api/       server.py                    # FastAPI: thin endpoints (+ serves the UI)
      config.py  models.py                    # centralized config / shared data types
    tests/       sample_docs/   pyproject.toml
  frontend/  index.html                        # single-file streaming chat UI
  slide/     index.html  src/                  # presentation deck
  README.md
```

The three external dependencies (`Embedder` / `VectorStore` / `LLMClient`) are all
`Protocol`s — swappable implementations (for scale) and injectable fakes (for tests).

---

## Build & run on another machine

### Prerequisites (one-time download, offline afterwards)

1. **Ollama** (local LLM runtime)
   - macOS (recommended): `brew install --cask ollama`
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Pull the models** (one-time):
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
   Verify GPU acceleration (Apple Silicon): `ollama ps` should show `100% GPU` in the
   `PROCESSOR` column.
3. **uv** (Python environment / package manager):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Install dependencies and run

`uv` creates and manages the virtual environment (`.venv`) for you; `uv run` executes
commands inside it, so there is no need to activate anything manually.

```bash
cd backend
uv sync                              # create .venv and install deps (from pyproject.toml + lockfile)
uv run uvicorn rag.api.server:app --port 8000
```

- Health check: `curl http://localhost:8000/health`
- Build the index (or let it scan `sample_docs/` at startup):
  `curl -X POST http://localhost:8000/ingest`
- Open the chat UI: browse to `http://localhost:8000`

### Tests

```bash
cd backend && uv run pytest
```

---

## Configuration

All knobs live in [`backend/rag/config.py`](./backend/rag/config.py) as one frozen
`Settings` dataclass: model names, `num_ctx=6000`, `num_predict`, `top_k`,
`score_threshold`, chunk size, folder paths, and the grounding system prompt.
`OLLAMA_HOST` is the one value overridable from the environment.

## Frontend

`frontend/index.html` is a single-file streaming chat UI (no framework, no build step):
it POSTs to `/chat`, consumes the SSE token stream with `fetch` + `ReadableStream`, and
renders each answer with its cited sources as footnotes. FastAPI serves it at `GET /`
from the same origin, so there is no CORS to configure — just open `http://localhost:8000`.

# Local RAG Chat — Usage

How to run this fully-local RAG chat application and ask it questions.

## Prerequisites

Everything runs locally, so the only one-time setup is a few downloads:

- **Ollama** — the local model runtime.
- **The two models**, pulled once: `llama3.2:3b` for chat and `nomic-embed-text` for
  embeddings.
- **uv** — the Python package and virtual-environment manager.

After the models are pulled, the app needs no internet — it can run fully offline.

## Running the app

From the `backend` folder, install dependencies and start the server:

- `uv sync` creates the virtual environment and installs the dependencies.
- `uv run uvicorn rag.api.server:app --port 8000` starts the server.

On the first start the server reads the documents in `sample_docs/`, embeds them, and
saves the index to disk; on later starts it just loads that saved index.

## Using the chat

Open `http://localhost:8000` in a browser. Type a question and the answer streams back
token by token. Answers are grounded in the ingested documents: the app cites its sources
inline as bracket numbers and shows them as footnotes under the answer. If the documents
do not contain the answer, the app says so instead of guessing.

## Endpoints

The server exposes a few HTTP endpoints:

- `GET /` serves the chat interface.
- `GET /health` reports that the server is running and which models it uses.
- `POST /ingest` rebuilds the index from the documents (safe to call repeatedly).
- `POST /chat` streams an answer as Server-Sent Events; the request body carries the
  question and the prior conversation.

## Updating the documents

After editing, adding, or removing files in `sample_docs/`, rebuild the index so the new
content is searchable — either call `POST /ingest` while the server runs, or delete the
saved index and restart the server to rebuild it from scratch.

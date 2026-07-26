# Local RAG Chat — Architecture

An overview of how this fully-local RAG chat application is put together. The LLM and
the entire retrieval pipeline run on the local machine via Ollama; the context window is
capped at about 6K tokens; the corpus is a handful of `.md` / `.txt` files.

## Two pipelines

RAG is two separate pipelines that share one piece of state — the vector store.

- **Ingestion (offline, runs once):** load the documents, split them into chunks, embed
  each chunk into a vector, and write them to the vector store (persisted to disk).
- **Query (online, runs every question):** embed the question, retrieve the most similar
  chunks, assemble a prompt that fits the 6K budget, and stream the answer from the LLM.

## Layers

The code is three layers, each thinner than the last.

- **api layer (thin):** FastAPI endpoints. A route only parses the request, calls the
  service, and streams the result back. It also serves the single-page frontend at `GET /`.
- **service layer (orchestration):** `RagService` wires the components together in the
  right order. It owns `answer()` and `ingest()` — the two pipelines.
- **components (behind interfaces):** the chunker, embedder, vector store, and LLM client.
  Each does one job and hides behind a `Protocol` interface, so any one can be swapped
  for a different implementation, and tests can inject fakes instead of calling Ollama.

## Components

- **Chunker:** splits on markdown headers first, so each chunk carries its header path as
  a citation breadcrumb; oversized sections are split further, measured in tokens. What
  gets embedded is the breadcrumb together with the body, so a chunk is findable even when
  its body never names its own topic; the chunk text placed in the prompt is just the
  body, labelled with its citation.
- **Embedder:** `nomic-embed-text` via Ollama, producing 768-dimensional vectors that are
  L2-normalized so a dot product equals cosine similarity. It uses asymmetric prefixes —
  `search_document:` when indexing and `search_query:` when asking.
- **Vector store:** a hand-written NumPy store doing exact (flat) cosine search — one
  matrix multiply scores every chunk — persisted to disk. At a few hundred chunks this is
  milliseconds and exact; at much larger scale the store can be swapped for a vector
  database behind the same interface, without changing retrieval or orchestration.
- **LLM client:** `llama3.2:3b` via Ollama's chat endpoint, streaming its answer token by
  token.

## Retrieval and the 6K budget

Retrieval returns the top-k chunks whose similarity clears a threshold. Those chunks are
the **residual** of the 6K window: the system prompt, the conversation history, the
question, and the space reserved for the model's output are subtracted first, and chunks
are packed into whatever room is left. A hard check refuses to send a prompt that would
overflow the window.

## Configuration

Every tunable value lives in one config file, so there is a single place to answer "where
is this set, and why that value?".

- Context window: 6000 tokens total, with 1024 reserved for the model's output and a 256
  token margin for estimation error.
- Conversation history is capped at 1000 tokens.
- Retrieval returns up to 8 chunks and ignores anything scoring below 0.35 similarity.
- Chunks target about 400 tokens with 15% overlap.
- Generation temperature is 0.2, kept low so the model stays close to the context.
- Token counts are estimated at roughly 4 characters per token, then checked against the
  real count the model reports after each answer.

## Streaming and citations

The answer streams to the browser token by token over Server-Sent Events. The model cites
its sources inline as bracket numbers like `[1]`; the frontend resolves each number to a
full human-readable source breadcrumb and shows it as a footnote under the answer.

## Multi-turn conversation

History is kept in a sliding window that fits the budget. For a follow-up question, the
app first rewrites it into a standalone question (resolving pronouns like "it") and uses
that rewrite for retrieval, while generating the answer from the user's original wording.

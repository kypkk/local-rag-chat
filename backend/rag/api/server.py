"""FastAPI app -- thin endpoints.

A route only: parse the request -> call the service -> return. Business logic lives in
RagService. Dependency injection (get_service) is the interface seam: inject the real
implementation in production, override with a fake in tests.
Endpoints: POST /chat (SSE stream), POST /ingest, GET /health.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag.config import settings
from rag.embed.embedder import OllamaEmbedder
from rag.llm.ollama_client import OllamaClient
from rag.models import Message
from rag.service import RagService
from rag.store.vector_store import NumpyStore


# -- request / response models (Pydantic validation) --------------
class ChatMessage(BaseModel):
    # Only the two roles a client is allowed to send. Accepting "system" here would let a
    # caller splice a second system message into the prompt and override the grounding rules.
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


# -- dependency injection: assemble real components (override in tests) --
_service: RagService | None = None


def build_service() -> RagService:
    """Wire the real components together. Built once at startup."""
    return RagService(
        embedder=OllamaEmbedder(),
        store=NumpyStore(),
        llm=OllamaClient(),
    )


def get_service() -> RagService:
    # raise, not assert: this guard must survive `python -O`, which strips asserts.
    if _service is None:
        raise RuntimeError("service not initialized; the app's lifespan did not run")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """At startup, build the service and load the index if it exists, else build it."""
    global _service
    _service = build_service()

    index_path = Path(settings.index_path)
    if index_path.exists():
        _service.store.load(str(index_path))   # reuse the persisted index (fast)
    else:
        _service.ingest()                      # first run: read docs, embed, persist
    yield


app = FastAPI(title="Local RAG Chat", lifespan=lifespan)

# the single-file frontend, served from this same origin so the browser needs no CORS
FRONTEND_INDEX = Path(__file__).resolve().parents[3] / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the chat UI."""
    return FRONTEND_INDEX.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    """Confirm the server is alive before a demo."""
    return {"ok": True, "llm_model": settings.llm_model, "embed_model": settings.embed_model}


@app.post("/ingest")
def ingest(svc: RagService = Depends(get_service)) -> dict:
    """(Re)build the index."""
    n = svc.ingest()
    return {"ingested_chunks": n}


@app.post("/chat")
async def chat(req: ChatRequest, svc: RagService = Depends(get_service)) -> StreamingResponse:
    """SSE streaming Q&A."""
    history = [Message(role=m.role, content=m.content) for m in req.history]

    def sse():
        # Each event is a typed object ({"type":"sources"|"token", ...}). json.dumps also
        # guards the SSE "\n\n" framing: any newline inside a token is escaped in the JSON,
        # and the browser JSON.parses each line back into the exact object.
        for event in svc.answer(req.question, history):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")

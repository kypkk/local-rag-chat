# Frontend

`index.html` — a single-file streaming chat UI (no framework, no build step).

The backend exposes:
- `GET /` — serves this UI (same origin, so no CORS)
- `GET /health`
- `POST /ingest`
- `POST /chat` — SSE stream, body `{"question": "...", "history": []}`

The page POSTs to `/chat` and consumes the SSE stream with `fetch` + `ReadableStream`
(not `EventSource`, which is GET-only), rendering tokens incrementally and resolving each
`[n]` citation into a source footnote.

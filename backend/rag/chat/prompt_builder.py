"""PromptBuilder -- assemble messages within the 6K budget.

Core discipline:
  chunk_budget = num_ctx - num_predict - margin - system - history - question   (chunks are the residual)
  before sending, assert the hard limit: estimate_tokens(prompt) + num_predict <= num_ctx
  (token_margin is reserved inside chunk_budget as headroom for citation prefixes and
  estimation error, so it is not subtracted a second time in the final assert)
Deterministic -> good to write and unit-test first (especially the token budget).
"""

from rag.config import SYSTEM_PROMPT, settings
from rag.models import Hit, Message


def estimate_tokens(text: str) -> int:
    """Rough token estimate (no public tokenize endpoint).

    Conservatively uses chars_per_token; verify the error against Ollama's
    prompt_eval_count afterwards.
    """
    return int(len(text) / settings.chars_per_token) + 1


def render_context(hits: list[Hit]) -> str:
    """Format the selected chunks into a numbered, cited context block."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] ({hit.chunk.citation}) {hit.chunk.text}")
    return "\n".join(lines)


def fill_by_budget(ranked_hits: list[Hit], chunk_budget_tokens: int) -> list[Hit]:
    """Pack chunks (already ranked by similarity) into the token budget: add them in
    order and stop before the running total would exceed chunk_budget_tokens.

    Counts the chunk body only; the citation prefix that render_context adds is absorbed
    by token_margin (reserved when the caller computes chunk_budget). Returns the hits used.
    """
    kept: list[Hit] = []
    used_tokens = 0
    for hit in ranked_hits:
        chunk_tokens = estimate_tokens(hit.chunk.text)
        if used_tokens + chunk_tokens > chunk_budget_tokens:
            break  # adding this chunk would overflow -> stop
        kept.append(hit)
        used_tokens += chunk_tokens
    return kept


def build_messages(question: str, history: list[Message],
                   hits: list[Hit]) -> tuple[list[Message], int, list[Hit]]:
    """Assemble the messages sent to Ollama, and return system_tokens (for num_keep) plus
    the kept hits (the chunks that actually made it into the context, in the [1..N] order
    the model cites -- so the caller can resolve each [n] to a human-readable source).

    Layout:
      system = static grounding rules (-> protected by num_keep)
      *history (raw Q/A -- the caller is expected to have trimmed it already)
      user   = "Context:\n{numbered chunks}\n\nQuestion: {question}"   <- chunks live here

    Chunks are the residual of the 6K budget: everyone else takes their share first, and
    chunks are packed into whatever room is left.
    """
    system_tokens = estimate_tokens(SYSTEM_PROMPT)
    history_tokens = sum(estimate_tokens(m.content) for m in history)
    question_tokens = estimate_tokens(question)

    chunk_budget = (
        settings.num_ctx
        - settings.num_predict     # reserved for the model's output (shares the window)
        - settings.token_margin    # headroom (see the assert below)
        - system_tokens
        - history_tokens
        - question_tokens
    )
    kept = fill_by_budget(hits, chunk_budget)

    user_content = f"Context:\n{render_context(kept)}\n\nQuestion: {question}"
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        *history,
        Message(role="user", content=user_content),
    ]

    # Hard 6K guard: input + reserved output must fit num_ctx, otherwise Ollama would
    # silently truncate from the front. token_margin was already reserved in chunk_budget
    # above (it is the slack that absorbs render_context's citation prefixes + wrapper),
    # so it does not appear again here. Use an explicit raise, not assert -- a safety
    # invariant must survive `python -O` (which strips asserts).
    prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
    if prompt_tokens + settings.num_predict > settings.num_ctx:
        raise ValueError(
            f"prompt {prompt_tokens} + reserved output {settings.num_predict} "
            f"exceeds num_ctx {settings.num_ctx}"
        )

    return messages, system_tokens, kept

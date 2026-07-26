"""History -- multi-turn conversation management.

Two independent problems:
  (1) budget    -> trim_history: sliding window; history shares the 6K residual with chunks
  (2) retrieval -> condense_question: rewrite a pronoun-laden follow-up into a standalone query (v2)
History stores raw Q/A only (never the assembled prompt with chunks), otherwise a few
turns overflow the 6K budget.
"""

from rag.config import settings
from rag.models import Message
from rag.chat.prompt_builder import estimate_tokens


def trim_history(history: list[Message],
                 cap_tokens: int | None = None) -> tuple[list[Message], int]:
    """Sliding window: keep the most recent messages whose tokens total <= cap, dropping
    the oldest. Returns (kept messages in chronological order, their total token count).
    """
    cap = cap_tokens if cap_tokens is not None else settings.history_token_cap

    # Walk from the newest message backwards, keeping messages until the next one would
    # push us over the cap.
    kept_newest_first: list[Message] = []
    total = 0
    for message in reversed(history):
        cost = estimate_tokens(message.content)
        if total + cost > cap:
            break  # this (older) message would overflow -> stop, drop it and everything older
        kept_newest_first.append(message)
        total += cost

    kept = list(reversed(kept_newest_first))  # restore chronological order
    return kept, total


def condense_question(history: list[Message], follow_up: str, llm) -> str:
    """Rewrite a follow-up into a standalone question using the conversation, so a
    pronoun-laden query ("what perms does it need?") becomes searchable on its own.

    First turn (no history) needs no rewrite, so we skip the LLM call entirely.
    """
    if not history:
        return follow_up

    conversation_lines = []
    for message in history:
        conversation_lines.append(f"{message.role}: {message.content}")
    conversation = "\n".join(conversation_lines)

    system = (
        "Rewrite the user's follow-up question into a standalone question that can be "
        "understood without the conversation. Resolve any pronouns or references using "
        "the conversation. Output ONLY the rewritten question, with no preamble."
    )
    user = f"Conversation:\n{conversation}\n\nFollow-up: {follow_up}\n\nStandalone question:"
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]
    options = {
        "num_ctx": settings.num_ctx,
        "num_predict": 80,     # a rewritten question is short
        "temperature": 0.0,    # deterministic rewrite
    }

    rewritten = "".join(llm.chat_stream(messages, options)).strip()
    return rewritten or follow_up   # fall back to the original if the rewrite comes back empty

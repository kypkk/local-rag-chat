"""Unit tests for the token budget / prompt assembly -- the
deterministic part, no LLM.
"""
from rag.chat.prompt_builder import (
    build_messages,
    estimate_tokens,
    fill_by_budget,
)
from rag.config import SYSTEM_PROMPT, settings
from rag.models import Chunk, Hit, Message


def _hit(text: str, score: float = 1.0) -> Hit:
    return Hit(chunk=Chunk(id="x", text=text, source="doc.md", headers=["Section"]), score=score)


def test_estimate_tokens_grows_with_length():
    assert estimate_tokens("") < estimate_tokens("a" * 100) < estimate_tokens("a" * 1000)
    assert estimate_tokens("a" * 400) >= 100   # ~4 chars/token


def test_fill_by_budget_respects_cap():
    hits = [_hit("word " * 60) for _ in range(5)]
    per = estimate_tokens("word " * 60)
    budget = per * 2 + 1                        # room for exactly two chunks, not three

    kept = fill_by_budget(hits, budget)

    total = sum(estimate_tokens(h.chunk.text) for h in kept)
    assert total <= budget
    assert len(kept) == 2                       # the budget forced a cutoff


def test_build_messages_invariant():
    # throw far more (large) chunks at it than can possibly fit
    hits = [_hit("lorem ipsum dolor sit amet " * 30) for _ in range(30)]

    messages, system_tokens, _ = build_messages("what is the rollback process?", [], hits)

    # the hard 6K guard holds: input + reserved output fits num_ctx
    prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
    assert prompt_tokens + settings.num_predict <= settings.num_ctx
    # system_tokens is the grounding prompt's size (used for num_keep)
    assert system_tokens == estimate_tokens(SYSTEM_PROMPT)
    # layout: system first, user last, chunks only in the user turn
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert "Context:" in messages[-1].content


def test_history_is_included_and_stays_in_budget():
    history = [
        Message(role="user", content="an earlier question"),
        Message(role="assistant", content="an earlier answer"),
    ]
    hits = [_hit("some context sentence. " * 20) for _ in range(10)]

    messages, _, _ = build_messages("a follow-up?", history, hits)

    # history sits between the system message and the final user turn, in order
    assert messages[1].content == "an earlier question"
    assert messages[2].content == "an earlier answer"
    prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
    assert prompt_tokens + settings.num_predict <= settings.num_ctx

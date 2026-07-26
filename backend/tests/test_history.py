"""Unit tests for history trimming + condense -- deterministic, no real LLM."""
from rag.chat.history import condense_question, trim_history
from rag.chat.prompt_builder import estimate_tokens
from rag.models import Message
from tests.fakes import FakeLLM


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def test_empty_history():
    kept, total = trim_history([])
    assert kept == []
    assert total == 0


def test_keeps_all_when_under_cap():
    history = [_msg("user", "hi"), _msg("assistant", "hello")]
    kept, total = trim_history(history, cap_tokens=1000)
    assert kept == history
    assert total == estimate_tokens("hi") + estimate_tokens("hello")


def test_drops_oldest_and_preserves_order():
    oldest = _msg("user", "word " * 40)
    middle = _msg("assistant", "word " * 40)
    newest = _msg("user", "word " * 40)
    per = estimate_tokens("word " * 40)

    kept, total = trim_history([oldest, middle, newest], cap_tokens=per * 2 + 1)

    assert kept == [middle, newest]        # oldest dropped, chronological order kept
    assert total <= per * 2 + 1


def test_single_oversized_message_drops_everything():
    kept, total = trim_history([_msg("user", "x" * 10000)], cap_tokens=10)
    assert kept == []
    assert total == 0


def test_condense_returns_followup_unchanged_when_no_history():
    # first turn: nothing to resolve, and no LLM call is made
    assert condense_question([], "what about it?", FakeLLM()) == "what about it?"


def test_condense_calls_llm_when_history_exists():
    history = [
        _msg("user", "how do I roll back?"),
        _msg("assistant", "re-deploy the previous release"),
    ]
    llm = FakeLLM()
    out = condense_question(history, "what perms does it need?", llm)

    assert out == "ok"                                    # FakeLLM echoes the "rewritten" question
    # the conversation + follow-up were handed to the LLM
    assert "what perms does it need?" in llm.last_messages[-1].content
    assert "how do I roll back?" in llm.last_messages[-1].content

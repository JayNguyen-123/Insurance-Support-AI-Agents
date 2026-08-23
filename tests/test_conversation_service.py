"""End-to-end coverage of the conversation service: session creation, the
clarification pause/resume cycle across two separate calls (simulating two
separate HTTP requests), and a full run through to a final answer -- all
without going through the notebook's blocking `input()` pattern.
"""

from __future__ import annotations

from app.agents import supervisor as supervisor_module
from app.llm import client as llm_client_module
from app.services.conversation_service import handle_chat_turn
from app.services.session_store import get_session_store
from tests.fakes import ScriptedOpenAIClient, text_response, tool_call_response


def _patch_llm(monkeypatch, responses):
    fake = ScriptedOpenAIClient(responses)
    monkeypatch.setattr(llm_client_module, "get_openai_client", lambda: fake)
    monkeypatch.setattr(supervisor_module, "get_openai_client", lambda: fake)
    return fake


def test_full_turn_no_clarification_needed(isolated_env, sample_policy_number, monkeypatch):
    _patch_llm(
        monkeypatch,
        [
            text_response(
                '{"next_agent": "policy_agent", "task": "get premium", "justification": "have policy number"}'
            ),
            tool_call_response("get_policy_details", {"policy_number": sample_policy_number}),
            text_response(f"Policy {sample_policy_number}: premium is $150.00/month."),
            text_response('{"next_agent": "end", "task": "done", "justification": "answered"}'),
            text_response("Your premium is $150.00 per month!"),
        ],
    )

    result = handle_chat_turn(None, f"What's the premium on {sample_policy_number}?")

    assert result.done is True
    assert result.requires_clarification is False
    assert result.requires_human_escalation is False
    assert "150.00" in result.reply
    assert result.session_id


def test_clarification_pause_then_resume_across_two_calls(isolated_env, sample_policy_number, monkeypatch):
    _patch_llm(
        monkeypatch,
        [tool_call_response("ask_user", {"question": "What is your policy number?", "missing_info": "policy_number"})],
    )

    first = handle_chat_turn(None, "What is my premium?")
    assert first.requires_clarification is True
    assert first.done is False
    assert first.reply == "What is your policy number?"

    # Confirm the session store actually persisted the paused graph state.
    record = get_session_store().get(first.session_id)
    assert record is not None
    assert record.pending_state is not None
    assert record.pending_state["needs_clarification"] is True

    _patch_llm(
        monkeypatch,
        [
            text_response(
                '{"next_agent": "policy_agent", "task": "get premium", "justification": "have policy number now"}'
            ),
            tool_call_response("get_policy_details", {"policy_number": sample_policy_number}),
            text_response(f"Policy {sample_policy_number}: premium is $77.50/month."),
            text_response('{"next_agent": "end", "task": "done", "justification": "answered"}'),
            text_response("Your premium is $77.50 per month. Anything else?"),
        ],
    )

    second = handle_chat_turn(first.session_id, sample_policy_number)

    assert second.session_id == first.session_id
    assert second.done is True
    assert second.requires_clarification is False
    assert "77.50" in second.reply

    # Pending state must be cleared once the turn completes.
    record_after = get_session_store().get(second.session_id)
    assert record_after.pending_state is None
    # Full history (including the clarification round-trip) should be preserved.
    assert "What is your policy number?" in record_after.conversation_history
    assert sample_policy_number in record_after.conversation_history


def test_human_escalation_turn(isolated_env, monkeypatch):
    _patch_llm(
        monkeypatch,
        [
            text_response(
                '{"next_agent": "human_escalation_agent", "task": "escalate", "justification": "explicit request"}'
            ),
            text_response("A human representative will be with you shortly."),
        ],
    )

    result = handle_chat_turn(None, "I want to speak to a human")

    assert result.done is True
    assert result.requires_human_escalation is True
    assert "human representative" in result.reply.lower()


def test_empty_message_raises_value_error(isolated_env):
    import pytest

    with pytest.raises(ValueError):
        handle_chat_turn(None, "   ")

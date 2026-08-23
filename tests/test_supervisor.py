from __future__ import annotations

from app.agents import supervisor
from tests.fakes import ScriptedOpenAIClient, text_response, tool_call_response


def _patch_client(monkeypatch, responses):
    fake = ScriptedOpenAIClient(responses)
    monkeypatch.setattr(supervisor, "get_openai_client", lambda: fake)
    return fake


def test_supervisor_routes_to_billing_agent(isolated_env, monkeypatch):
    _patch_client(
        monkeypatch,
        [text_response('{"next_agent": "billing_agent", "task": "check balance", "justification": "billing question"}')],
    )

    state = {
        "conversation_history": "User: What's my balance?",
        "n_iteration": 0,
    }
    result = supervisor.supervisor_agent(state)

    assert result["next_agent"] == "billing_agent"
    assert result["task"] == "check balance"
    assert result["n_iteration"] == 1
    assert result["needs_clarification"] is False
    assert "Routing to billing_agent" in result["conversation_history"]


def test_supervisor_falls_back_on_invalid_json(isolated_env, monkeypatch):
    _patch_client(monkeypatch, [text_response("not valid json")])

    state = {"conversation_history": "User: hello", "n_iteration": 0}
    result = supervisor.supervisor_agent(state)

    assert result["next_agent"] == "general_help_agent"


def test_supervisor_rejects_unknown_next_agent(isolated_env, monkeypatch):
    _patch_client(
        monkeypatch,
        [text_response('{"next_agent": "some_made_up_agent", "task": "x", "justification": "y"}')],
    )

    state = {"conversation_history": "User: hello", "n_iteration": 0}
    result = supervisor.supervisor_agent(state)

    assert result["next_agent"] == "general_help_agent"


def test_supervisor_pauses_for_clarification_without_blocking(isolated_env, monkeypatch):
    """The critical production fix: asking the user a question must never
    call input() or otherwise block -- it returns needs_clarification=True
    and the question text, and the caller (API layer) is responsible for
    getting the answer on a later request."""
    _patch_client(
        monkeypatch,
        [tool_call_response("ask_user", {"question": "What is your policy number?", "missing_info": "policy_number"})],
    )

    state = {"conversation_history": "User: What's my premium?", "n_iteration": 0}
    result = supervisor.supervisor_agent(state)

    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "What is your policy number?"
    assert "next_agent" not in result  # no routing decision made yet


def test_supervisor_resumes_with_clarification_in_one_call(isolated_env, monkeypatch):
    """Regression test: in the original notebook, processing a returned
    clarification answer took a separate supervisor invocation from making
    the actual routing decision, silently burning the iteration budget. Here,
    resuming and routing happen in a single call."""
    _patch_client(
        monkeypatch,
        [text_response('{"next_agent": "policy_agent", "task": "get policy info", "justification": "have policy number now"}')],
    )

    state = {
        "conversation_history": "User: What's my premium?\nAssistant: What is your policy number?",
        "n_iteration": 1,
        "needs_clarification": True,
        "clarification_question": "What is your policy number?",
        "user_clarification": "POL000001",
    }
    result = supervisor.supervisor_agent(state)

    assert result["next_agent"] == "policy_agent"
    assert result["n_iteration"] == 2
    assert "User: POL000001" in result["conversation_history"]


def test_supervisor_escalates_after_max_iterations(isolated_env, monkeypatch):
    from app.config import get_settings

    # Pin a tight cap for this boundary test, independent of the shared
    # fixture's more realistic default (see app/config.py).
    monkeypatch.setenv("SUPERVISOR_MAX_ITERATIONS", "3")
    get_settings.cache_clear()

    fake = ScriptedOpenAIClient([])  # should not be called at all
    monkeypatch.setattr(supervisor, "get_openai_client", lambda: fake)

    state = {"conversation_history": "User: still confused", "n_iteration": 2}
    result = supervisor.supervisor_agent(state)

    assert result["escalate_to_human"] is True
    assert result["requires_human_escalation"] is True
    assert result["next_agent"] == "human_escalation_agent"
    assert fake.calls == []

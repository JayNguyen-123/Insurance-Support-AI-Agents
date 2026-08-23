"""End-to-end API tests, driving the FastAPI app through TestClient with a
fully scripted (mocked) OpenAI client -- no real network/API key needed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agents import supervisor as supervisor_module
from app.llm import client as llm_client_module
from tests.fakes import ScriptedOpenAIClient, text_response, tool_call_response


def _patch_llm(monkeypatch, responses):
    fake = ScriptedOpenAIClient(responses)
    monkeypatch.setattr(llm_client_module, "get_openai_client", lambda: fake)
    monkeypatch.setattr(supervisor_module, "get_openai_client", lambda: fake)
    return fake


def test_health_endpoint(isolated_env):
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True


def test_chat_happy_path_with_policy_number(isolated_env, sample_policy_number, monkeypatch):
    from app.main import app

    _patch_llm(
        monkeypatch,
        [
            text_response(
                '{"next_agent": "policy_agent", "task": "get premium", "justification": "policy number provided"}'
            ),
            tool_call_response("get_policy_details", {"policy_number": sample_policy_number}),
            text_response(f"Policy {sample_policy_number} has a premium of $123.45 per month."),
            text_response('{"next_agent": "end", "task": "done", "justification": "answered"}'),
            text_response(f"Sure! Your premium for {sample_policy_number} is $123.45/month. Anything else?"),
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat",
            json={"message": f"What is the premium on my policy {sample_policy_number}?"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is True
    assert body["requires_clarification"] is False
    assert body["requires_human_escalation"] is False
    assert "123.45" in body["reply"]
    assert body["session_id"]


def test_chat_clarification_pause_and_resume(isolated_env, sample_policy_number, monkeypatch):
    from app.main import app

    _patch_llm(
        monkeypatch,
        [tool_call_response("ask_user", {"question": "What is your policy number?", "missing_info": "policy_number"})],
    )

    with TestClient(app) as client:
        first = client.post("/api/v1/chat", json={"message": "What is my premium?"})
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["requires_clarification"] is True
        assert first_body["done"] is False
        assert "policy number" in first_body["reply"].lower()
        session_id = first_body["session_id"]

        # Second call resumes the SAME session with the answer. Re-script
        # the fake client for the next leg of the conversation.
        _patch_llm(
            monkeypatch,
            [
                text_response(
                    '{"next_agent": "policy_agent", "task": "get premium", "justification": "have policy number now"}'
                ),
                tool_call_response("get_policy_details", {"policy_number": sample_policy_number}),
                text_response(f"Policy {sample_policy_number} has a premium of $99.99 per month."),
                text_response('{"next_agent": "end", "task": "done", "justification": "answered"}'),
                text_response("Your premium is $99.99/month. Let me know if you need anything else!"),
            ],
        )

        second = client.post(
            "/api/v1/chat", json={"session_id": session_id, "message": sample_policy_number}
        )

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["session_id"] == session_id
    assert second_body["done"] is True
    assert second_body["requires_clarification"] is False
    assert "99.99" in second_body["reply"]


def test_chat_human_escalation(isolated_env, monkeypatch):
    from app.main import app

    _patch_llm(
        monkeypatch,
        [
            text_response(
                '{"next_agent": "human_escalation_agent", "task": "escalate", "justification": "user requested a human"}'
            ),
            text_response("Understood -- a human representative will join shortly to assist you."),
        ],
    )

    with TestClient(app) as client:
        resp = client.post("/api/v1/chat", json={"message": "I want to talk to a human executive"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is True
    assert body["requires_human_escalation"] is True
    assert "human" in body["reply"].lower()


def test_chat_rejects_empty_message(isolated_env):
    from app.main import app

    with TestClient(app) as client:
        resp = client.post("/api/v1/chat", json={"message": ""})

    assert resp.status_code == 422  # pydantic min_length validation


def test_chat_without_api_key_returns_503(isolated_env, monkeypatch):
    from app.config import get_settings
    from app.main import app

    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    with TestClient(app) as client:
        resp = client.post("/api/v1/chat", json={"message": "hello"})

    assert resp.status_code == 503
    get_settings.cache_clear()

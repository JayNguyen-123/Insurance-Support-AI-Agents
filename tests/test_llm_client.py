from __future__ import annotations

from app.llm import client as llm_client_module
from tests.fakes import ScriptedOpenAIClient, text_response, tool_call_response


def test_run_llm_with_tools_no_tool_call_returns_content(isolated_env, monkeypatch):
    fake = ScriptedOpenAIClient([text_response("plain answer")])
    monkeypatch.setattr(llm_client_module, "get_openai_client", lambda: fake)

    result = llm_client_module.run_llm_with_tools("system prompt")
    assert result == "plain answer"


def test_run_llm_with_tools_executes_tool_and_returns_followup(isolated_env, monkeypatch):
    fake = ScriptedOpenAIClient(
        [
            tool_call_response("double", {"x": 21}),
            text_response("the answer is 42"),
        ]
    )
    monkeypatch.setattr(llm_client_module, "get_openai_client", lambda: fake)

    result = llm_client_module.run_llm_with_tools(
        "system prompt",
        tools=[{"type": "function", "function": {"name": "double", "parameters": {}}}],
        tool_functions={"double": lambda x: {"result": x * 2}},
    )

    assert result == "the answer is 42"
    # The tool result should have been serialized into the followup call's messages.
    followup_kwargs = fake.calls[1]
    tool_msg = [m for m in followup_kwargs["messages"] if m.get("role") == "tool"][0]
    assert '"result": 42' in tool_msg["content"]


def test_run_llm_with_tools_handles_tool_exception_gracefully(isolated_env, monkeypatch):
    def boom(**kwargs):
        raise ValueError("db is down")

    fake = ScriptedOpenAIClient(
        [
            tool_call_response("explode", {}),
            text_response("recovered"),
        ]
    )
    monkeypatch.setattr(llm_client_module, "get_openai_client", lambda: fake)

    result = llm_client_module.run_llm_with_tools(
        "system prompt",
        tools=[{"type": "function", "function": {"name": "explode", "parameters": {}}}],
        tool_functions={"explode": boom},
    )

    assert result == "recovered"
    followup_kwargs = fake.calls[1]
    tool_msg = [m for m in followup_kwargs["messages"] if m.get("role") == "tool"][0]
    assert "db is down" in tool_msg["content"]


def test_run_llm_with_tools_unimplemented_tool(isolated_env, monkeypatch):
    fake = ScriptedOpenAIClient(
        [
            tool_call_response("mystery_tool", {}),
            text_response("handled gracefully"),
        ]
    )
    monkeypatch.setattr(llm_client_module, "get_openai_client", lambda: fake)

    result = llm_client_module.run_llm_with_tools(
        "system prompt",
        tools=[{"type": "function", "function": {"name": "mystery_tool", "parameters": {}}}],
        # Non-empty but missing "mystery_tool" -- exercises the per-call
        # "not implemented" fallback rather than the "no tools at all" early return.
        tool_functions={"some_other_tool": lambda: None},
    )

    assert result == "handled gracefully"
    followup_kwargs = fake.calls[1]
    tool_msg = [m for m in followup_kwargs["messages"] if m.get("role") == "tool"][0]
    assert "not implemented" in tool_msg["content"]

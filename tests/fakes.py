"""Test doubles for the OpenAI client: a scripted, in-order queue of
canned chat-completion responses. No network calls, no real API key needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


class FakeToolCall:
    def __init__(self, name: str, arguments: dict[str, Any]):
        self.id = f"call_{name}"
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[FakeToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeMessage):
        self.choices = [FakeChoice(message)]


def text_response(content: str) -> FakeResponse:
    return FakeResponse(FakeMessage(content=content))


def tool_call_response(name: str, arguments: dict[str, Any], content: str | None = None) -> FakeResponse:
    return FakeResponse(FakeMessage(content=content, tool_calls=[FakeToolCall(name, arguments)]))


class ScriptedOpenAIClient:
    """Returns responses from a fixed script, one per call, in order."""

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> FakeResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(
                f"ScriptedOpenAIClient exhausted: no more scripted responses for call #{len(self.calls)} "
                f"(kwargs={kwargs!r})"
            )
        return self._responses.pop(0)

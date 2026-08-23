"""OpenAI client wrapper: retries, timeouts, and a generalized tool-calling loop.

Ported from the notebook's `run_llm`, with production fixes:

- The OpenAI client is lazily constructed (not at import time), so importing
  this module never requires `OPENAI_API_KEY` to already be set -- important
  for tests and for the app to fail with a clear error at request time
  instead of at import time.
- Network calls are wrapped with `tenacity` retries (exponential backoff) on
  transient errors (rate limits, timeouts, connection errors) instead of
  failing the whole user request on the first hiccup.
- `print(...)` debug statements are replaced with structured logging.
- Tool-call argument JSON parsing failures are caught per tool call instead
  of raising and losing the whole response.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in your environment or .env file."
        )
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_request_timeout_seconds,
    )


def _retrying():
    settings = get_settings()
    return retry(
        reraise=True,
        stop=stop_after_attempt(settings.openai_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    )


def complete(prompt: str, model: str | None = None) -> str:
    """Simple single-turn completion, no tools. Used for the escalation and
    final-answer agents, which just need a plain text response."""
    settings = get_settings()
    client = get_openai_client()
    model = model or settings.openai_model

    @_retrying()
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
        )

    response = _call()
    return response.choices[0].message.content or ""


def run_llm_with_tools(
    prompt: str,
    tools: list[dict] | None = None,
    tool_functions: dict[str, Callable[..., Any]] | None = None,
    model: str | None = None,
) -> str:
    """Run an LLM request that optionally supports tool/function calling.

    Executes at most one round of tool calls (matching the original
    notebook's behavior): if the model requests tools, each is executed and
    the results are sent back for a single follow-up completion.
    """
    settings = get_settings()
    client = get_openai_client()
    model = model or settings.openai_model

    @_retrying()
    def _initial_call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )

    response = _initial_call()
    message = response.choices[0].message
    logger.debug("LLM initial response", extra={"has_tool_calls": bool(message.tool_calls)})

    if not getattr(message, "tool_calls", None):
        return message.content or ""

    if not tool_functions:
        return (message.content or "") + "\n\n[No tool functions were provided to execute the requested tool calls.]"

    tool_messages = []
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool call arguments", extra={"tool": func_name})
            args = {}

        tool_fn = tool_functions.get(func_name)
        try:
            result = tool_fn(**args) if tool_fn else {"error": f"Tool '{func_name}' not implemented."}
        except Exception as exc:  # noqa: BLE001 - tool errors must not crash the agent
            logger.exception("Tool execution failed", extra={"tool": func_name})
            result = {"error": str(exc)}

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, default=str),
        })

    followup_messages = [
        {"role": "system", "content": prompt},
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        },
        *tool_messages,
    ]

    @_retrying()
    def _followup_call():
        return client.chat.completions.create(model=model, messages=followup_messages)

    final = _followup_call()
    return final.choices[0].message.content or ""

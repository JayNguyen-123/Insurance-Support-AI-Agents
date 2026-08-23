"""Orchestrates one chat turn: loads/creates a session, resumes a paused
clarification if needed, invokes the LangGraph app, and persists the result.

This is the layer that replaces the notebook's `run_test_query`: instead of
a single blocking `app.invoke()` call per test query with no cross-call
memory, each HTTP request is one "turn" against a persistent session. A turn
either pauses (the supervisor needs clarification -- the caller is expected
to send another message with the answer, reusing the same session_id) or
completes (a final answer or a human-escalation acknowledgment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.graph import get_app
from app.agents.state import GraphState, new_initial_state
from app.logging_config import get_logger
from app.services.session_store import SessionRecord, get_session_store

logger = get_logger(__name__)

# Bound how many supervisor<->specialist hops a single turn can take. This is
# independent of SUPERVISOR_MAX_ITERATIONS (which bounds supervisor *routing
# decisions* and forces an escalation past that point) -- this is a hard
# safety net against runaway graph execution.
GRAPH_RECURSION_LIMIT = 25


class ConversationServiceError(RuntimeError):
    """Raised when the graph cannot complete a turn (upstream LLM/DB failure)."""


@dataclass
class ChatTurnResult:
    session_id: str
    reply: str
    requires_clarification: bool
    requires_human_escalation: bool
    done: bool
    agent_used: str | None = None


def _build_resumed_state(record: SessionRecord, user_message: str) -> GraphState:
    state = dict(record.pending_state or {})
    state["needs_clarification"] = True
    state["user_clarification"] = user_message
    state["user_input"] = user_message
    return state  # type: ignore[return-value]


def _build_fresh_state(record: SessionRecord, user_message: str) -> GraphState:
    history = record.conversation_history
    if history:
        history = history + f"\nUser: {user_message}"
    else:
        history = f"User: {user_message}"
    return new_initial_state(session_id=record.session_id, user_input=user_message, conversation_history=history)


def handle_chat_turn(session_id: str | None, user_message: str) -> ChatTurnResult:
    if not user_message or not user_message.strip():
        raise ValueError("message must not be empty")

    store = get_session_store()
    record = store.get_or_create(session_id)

    is_resume = bool(record.pending_state and record.pending_state.get("needs_clarification"))
    state = _build_resumed_state(record, user_message) if is_resume else _build_fresh_state(record, user_message)

    logger.info(
        "Handling chat turn",
        extra={"session_id": record.session_id, "resume": is_resume},
    )

    try:
        graph_app = get_app()
        final_state: dict[str, Any] = graph_app.invoke(state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
    except Exception as exc:  # noqa: BLE001 - surface as a clean service error to the API layer
        logger.exception("Graph execution failed", extra={"session_id": record.session_id})
        raise ConversationServiceError(str(exc)) from exc

    conversation_history = final_state.get("conversation_history", record.conversation_history)

    if final_state.get("needs_clarification"):
        record.pending_state = final_state
        record.conversation_history = conversation_history
        store.save(record)
        return ChatTurnResult(
            session_id=record.session_id,
            reply=final_state.get("clarification_question", "Could you provide more details?"),
            requires_clarification=True,
            requires_human_escalation=False,
            done=False,
        )

    # Turn completed (final answer or human escalation) -- clear any pending state.
    record.pending_state = None
    record.conversation_history = conversation_history
    store.save(record)

    reply = final_state.get("final_answer") or "I'm sorry, I wasn't able to generate a response."
    return ChatTurnResult(
        session_id=record.session_id,
        reply=reply,
        requires_clarification=False,
        requires_human_escalation=bool(final_state.get("requires_human_escalation")),
        done=True,
        agent_used=final_state.get("next_agent"),
    )


def reset_session(session_id: str) -> None:
    get_session_store().delete(session_id)

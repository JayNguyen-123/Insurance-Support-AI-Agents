"""Final answer agent: turns the latest specialist response into a clean,
customer-facing summary."""

from __future__ import annotations

from app.agents.prompts import FINAL_ANSWER_PROMPT
from app.agents.state import GraphState
from app.llm.client import complete
from app.logging_config import get_logger
from app.tracing import trace_agent

logger = get_logger(__name__)


def _extract_specialist_response(state: GraphState) -> str:
    """Find the most recent non-clarification assistant message.

    `state["messages"]` entries are normalized to LangChain BaseMessage
    objects by LangGraph's `add_messages` reducer (the nodes append plain
    `("assistant", text)` tuples; the reducer converts them), so `.content`
    is available here even though the nodes never construct message objects
    directly.
    """
    for msg in reversed(state.get("messages", []) or []):
        content = getattr(msg, "content", None)
        if content and "clarification" not in content.lower():
            return content
    return "No response available"


@trace_agent
def final_answer_agent(state: GraphState) -> dict:
    logger.info("Final answer agent started")

    user_query = state.get("user_input", "")
    conversation_history = state.get("conversation_history", "")
    specialist_response = _extract_specialist_response(state)

    prompt = FINAL_ANSWER_PROMPT.format(
        specialist_response=specialist_response,
        user_query=user_query,
    )

    final_answer = complete(prompt)
    logger.info("Final answer generated")

    return {
        "final_answer": final_answer,
        "end_conversation": True,
        "conversation_history": conversation_history + f"\nAssistant: {final_answer}",
        "messages": [("assistant", final_answer)],
    }

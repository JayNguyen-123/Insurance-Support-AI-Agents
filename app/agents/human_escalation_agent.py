from __future__ import annotations

from app.agents.prompts import HUMAN_ESCALATION_PROMPT
from app.agents.state import GraphState
from app.llm.client import complete
from app.logging_config import get_logger
from app.tracing import trace_agent

logger = get_logger(__name__)


@trace_agent
def human_escalation_node(state: GraphState) -> dict:
    logger.warning("Escalation triggered")

    prompt = HUMAN_ESCALATION_PROMPT.format(
        task=state.get("task"),
        conversation_history=state.get("conversation_history", ""),
    )

    response = complete(prompt)

    logger.info("Conversation escalated to human")
    current_history = state.get("conversation_history", "")
    return {
        "final_answer": response,
        "requires_human_escalation": True,
        "escalation_reason": state.get("escalation_reason") or "Customer requested human assistance.",
        "conversation_history": current_history + f"\nAssistant: {response}",
        "messages": [("assistant", response)],
        "end_conversation": True,
    }

from __future__ import annotations

from app.agents.prompts import BILLING_AGENT_PROMPT
from app.agents.state import GraphState
from app.agents.tools import BILLING_TOOL_FUNCTIONS, BILLING_TOOLS
from app.llm.client import run_llm_with_tools
from app.logging_config import get_logger
from app.tracing import trace_agent

logger = get_logger(__name__)


@trace_agent
def billing_agent_node(state: GraphState) -> dict:
    logger.info("Billing agent started", extra={"task": state.get("task")})

    prompt = BILLING_AGENT_PROMPT.format(
        task=state.get("task"),
        conversation_history=state.get("conversation_history", ""),
    )

    result = run_llm_with_tools(prompt, BILLING_TOOLS, BILLING_TOOL_FUNCTIONS)

    logger.info("Billing agent completed")

    updated_state: dict = {"messages": [("assistant", result)]}
    if state.get("policy_number"):
        updated_state["policy_number"] = state["policy_number"]
    if state.get("customer_id"):
        updated_state["customer_id"] = state["customer_id"]

    current_history = state.get("conversation_history", "")
    updated_state["conversation_history"] = current_history + f"\nBilling Agent: {result}"
    return updated_state

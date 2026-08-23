from __future__ import annotations

from app.agents.prompts import POLICY_AGENT_PROMPT
from app.agents.state import GraphState
from app.agents.tools import POLICY_TOOL_FUNCTIONS, POLICY_TOOLS
from app.llm.client import run_llm_with_tools
from app.logging_config import get_logger
from app.tracing import trace_agent

logger = get_logger(__name__)


@trace_agent
def policy_agent_node(state: GraphState) -> dict:
    logger.info("Policy agent started")

    prompt = POLICY_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        customer_id=state.get("customer_id", "Not provided"),
        conversation_history=state.get("conversation_history", ""),
    )

    result = run_llm_with_tools(prompt, POLICY_TOOLS, POLICY_TOOL_FUNCTIONS)

    logger.info("Policy agent completed")
    current_history = state.get("conversation_history", "")
    return {
        "messages": [("assistant", result)],
        "conversation_history": current_history + f"\nPolicy Agent: {result}",
    }

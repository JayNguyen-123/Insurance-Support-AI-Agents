from __future__ import annotations

from app.agents.prompts import CLAIMS_AGENT_PROMPT
from app.agents.state import GraphState
from app.agents.tools import CLAIMS_TOOL_FUNCTIONS, CLAIMS_TOOLS
from app.llm.client import run_llm_with_tools
from app.logging_config import get_logger
from app.tracing import trace_agent

logger = get_logger(__name__)


@trace_agent
def claims_agent_node(state: GraphState) -> dict:
    logger.info("Claims agent started")

    prompt = CLAIMS_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        claim_id=state.get("claim_id", "Not provided"),
        conversation_history=state.get("conversation_history", ""),
    )

    result = run_llm_with_tools(prompt, CLAIMS_TOOLS, CLAIMS_TOOL_FUNCTIONS)

    logger.info("Claims agent completed")
    current_history = state.get("conversation_history", "")
    return {
        "messages": [("assistant", result)],
        "conversation_history": current_history + f"\nClaims Agent: {result}",
    }

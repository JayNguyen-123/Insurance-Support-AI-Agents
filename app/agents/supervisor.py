"""Supervisor agent: routes to specialists, or pauses the graph to ask the user.

This is the most significant change from the original notebook. The notebook's
supervisor called a local `ask_user()` helper that did a blocking
`input(...)` call -- fine in a Colab cell talking to a human at a keyboard,
fatal in a web server handling concurrent requests (it would block an entire
worker thread waiting on stdin that nothing will ever write to).

Here, when the LLM decides it needs to ask the user something, the supervisor
node simply returns `needs_clarification=True` plus the question, and the
graph (see `graph.py`) routes straight to END instead of looping back into
itself. The FastAPI layer sees `needs_clarification=True` in the final state,
returns the question to the caller, and stashes the in-flight state in the
session store. When the caller's next message arrives for that session, the
service layer resumes the graph by re-invoking it with
`needs_clarification=True` and `user_clarification=<the new message>` set on
the loaded state -- which this node detects and folds into
`conversation_history` before proceeding, in the same call, to the normal
LLM routing decision below.

This also fixes a real bug in the original: there, `n_iteration` was
incremented on *every* supervisor invocation, and a single clarification
round-trip took three separate invocations (ask -> merge -> route) to
resolve -- silently burning the entire 3-iteration budget on one
clarification and forcing an escalation on what should have been the very
next real routing decision. Here, resuming with an answer and making the
routing decision happen in one invocation, so `n_iteration` tracks actual
supervisor decisions.
"""

from __future__ import annotations

import json

from app.agents.prompts import SUPERVISOR_PROMPT
from app.agents.state import GraphState
from app.agents.tools import ASK_USER_TOOL
from app.config import get_settings
from app.llm.client import get_openai_client
from app.logging_config import get_logger
from app.tracing import trace_agent

logger = get_logger(__name__)

VALID_NEXT_AGENTS = {
    "policy_agent",
    "billing_agent",
    "claims_agent",
    "general_help_agent",
    "human_escalation_agent",
    "end",
}


@trace_agent
def supervisor_agent(state: GraphState) -> dict:
    settings = get_settings()
    conversation_history = state.get("conversation_history", "") or ""

    # --- Resuming after the caller answered a clarification question ---
    if state.get("needs_clarification") and state.get("user_clarification"):
        user_clarification = state["user_clarification"]
        logger.info("Resuming supervisor with user clarification.")
        conversation_history = conversation_history + f"\nUser: {user_clarification}"

    n_iter = state.get("n_iteration", 0) + 1
    logger.info("Supervisor iteration %d", n_iter)

    if n_iter >= settings.supervisor_max_iterations:
        logger.warning("Max supervisor iterations reached; escalating to human agent.")
        updated_history = (
            conversation_history
            + "\nAssistant: It seems this issue requires human review. Escalating to a human support specialist."
        )
        return {
            "escalate_to_human": True,
            "requires_human_escalation": True,
            "conversation_history": updated_history,
            "next_agent": "human_escalation_agent",
            "n_iteration": n_iter,
            "needs_clarification": False,
            "clarification_question": None,
            "user_clarification": None,
        }

    prompt = SUPERVISOR_PROMPT.format(conversation_history=f"Full Conversation:\n{conversation_history}")

    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "system", "content": prompt}],
        tools=[ASK_USER_TOOL],
        tool_choice="auto",
    )
    message = response.choices[0].message

    if getattr(message, "tool_calls", None):
        for tool_call in message.tool_calls:
            if tool_call.function.name != "ask_user":
                continue
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            question = args.get("question", "Can you please provide more details?")
            logger.info("Supervisor requesting clarification: %s", question)

            updated_history = conversation_history + f"\nAssistant: {question}"
            return {
                "needs_clarification": True,
                "clarification_question": question,
                "user_clarification": None,
                "conversation_history": updated_history,
                "n_iteration": n_iter,
            }

    # No clarification requested -> parse the routing decision.
    message_content = message.content or ""
    try:
        parsed = json.loads(message_content)
    except json.JSONDecodeError:
        logger.warning("Supervisor produced non-JSON output; falling back to general_help_agent.")
        parsed = {}

    next_agent = parsed.get("next_agent", "general_help_agent")
    if next_agent not in VALID_NEXT_AGENTS:
        logger.warning("Supervisor returned unknown next_agent=%r; defaulting.", next_agent)
        next_agent = "general_help_agent"

    task = parsed.get("task", "Assist the user with their query.")
    justification = parsed.get("justification", "")

    logger.info("Supervisor decision: next_agent=%s task=%s", next_agent, task)

    updated_conversation = conversation_history + f"\nAssistant: Routing to {next_agent} for: {task}"

    return {
        "next_agent": next_agent,
        "task": task,
        "justification": justification,
        "conversation_history": updated_conversation,
        "n_iteration": n_iter,
        "needs_clarification": False,
        "clarification_question": None,
        "user_clarification": None,
    }

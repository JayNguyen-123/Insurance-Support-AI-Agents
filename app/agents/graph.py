"""LangGraph wiring: nodes, edges, and the routing function.

Fixes three bugs present in the original notebook's graph-construction cell:

1. `decide_next_agent` had a syntax error (a stray/misplaced parenthesis on
   the `needs_clarification` check) that would have raised `SyntaxError` at
   import time.
2. The loop-back edges were registered with `for node in ["policy_agennt",
   ...]` -- a typo (`policy_agennt`) meant `policy_agent` itself never got an
   edge back to the supervisor, leaving it a dead-end node that LangGraph
   would reject at compile time (or silently strand the conversation, if the
   library didn't validate it).
3. `needs_clarification` now routes straight to `END` instead of looping
   back into `supervisor_agent` -- see `app/agents/supervisor.py` for why:
   the graph must pause and hand control back to the API layer rather than
   block on stdin.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.billing_agent import billing_agent_node
from app.agents.claims_agent import claims_agent_node
from app.agents.final_answer_agent import final_answer_agent
from app.agents.general_help_agent import general_help_agent_node
from app.agents.human_escalation_agent import human_escalation_node
from app.agents.policy_agent import policy_agent_node
from app.agents.state import GraphState
from app.agents.supervisor import supervisor_agent


def decide_next_agent(state: GraphState) -> str:
    if state.get("needs_clarification"):
        return "await_clarification"

    if state.get("escalate_to_human") or state.get("requires_human_escalation"):
        return "human_escalation_agent"

    next_agent = state.get("next_agent", "general_help_agent")
    if next_agent == "end":
        return "end"
    return next_agent


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("policy_agent", policy_agent_node)
    workflow.add_node("claims_agent", claims_agent_node)
    workflow.add_node("billing_agent", billing_agent_node)
    workflow.add_node("general_help_agent", general_help_agent_node)
    workflow.add_node("human_escalation_agent", human_escalation_node)
    workflow.add_node("final_answer_agent", final_answer_agent)

    workflow.set_entry_point("supervisor_agent")

    workflow.add_conditional_edges(
        "supervisor_agent",
        decide_next_agent,
        {
            "await_clarification": END,
            "policy_agent": "policy_agent",
            "billing_agent": "billing_agent",
            "claims_agent": "claims_agent",
            "human_escalation_agent": "human_escalation_agent",
            "general_help_agent": "general_help_agent",
            "end": "final_answer_agent",
        },
    )

    # Specialists report back to the supervisor, which decides whether the
    # request is fully answered or needs another hop.
    for node in ["policy_agent", "billing_agent", "claims_agent", "general_help_agent"]:
        workflow.add_edge(node, "supervisor_agent")

    workflow.add_edge("final_answer_agent", END)
    workflow.add_edge("human_escalation_agent", END)

    return workflow.compile()


_compiled_app = None


def get_app():
    """Return the compiled LangGraph app, building it lazily (and once)."""
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = build_graph()
    return _compiled_app

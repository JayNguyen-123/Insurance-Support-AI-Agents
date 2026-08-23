"""Regression tests for graph-construction bugs in the original notebook:
a syntax error in `decide_next_agent`, and a typo (`policy_agennt`) that left
`policy_agent` without an edge back to the supervisor.
"""

from __future__ import annotations

from app.agents.graph import build_graph, decide_next_agent


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_decide_next_agent_routes_clarification_to_pause():
    assert decide_next_agent({"needs_clarification": True}) == "await_clarification"


def test_decide_next_agent_routes_escalation():
    assert decide_next_agent({"requires_human_escalation": True}) == "human_escalation_agent"


def test_decide_next_agent_routes_end():
    assert decide_next_agent({"next_agent": "end"}) == "end"


def test_decide_next_agent_routes_specialist():
    assert decide_next_agent({"next_agent": "policy_agent"}) == "policy_agent"


def test_all_specialists_have_edge_back_to_supervisor():
    graph = build_graph()
    # LangGraph's compiled graph exposes the underlying node/edge structure.
    graph_repr = graph.get_graph()
    edges = {(e.source, e.target) for e in graph_repr.edges}
    for specialist in ["policy_agent", "billing_agent", "claims_agent", "general_help_agent"]:
        assert (specialist, "supervisor_agent") in edges, f"{specialist} is missing its edge back to supervisor_agent"

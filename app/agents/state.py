"""LangGraph shared state definition.

Ported from the notebook's `GraphState`, with two changes:

1. Fields that were read/written by agent nodes but never declared on the
   original TypedDict (`needs_clarification`, `user_clarification`,
   `clarification_question`, `escalate_to_human`, `retrieved_faqs`,
   `final_answer`) are now declared, so the state shape is fully documented
   and type-checkable.
2. `session_id` was added -- the conversation/session service (see
   `app/services/conversation_service.py`) uses it to persist and resume
   state across separate HTTP requests, since a production chat turn can no
   longer block on a single in-process `input()` call the way the notebook
   did.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import add_messages


class GraphState(TypedDict, total=False):
    # Core conversation tracking
    messages: Annotated[List[Any], add_messages]
    session_id: str
    user_input: str
    conversation_history: Optional[str]

    n_iteration: Optional[int]

    # Extracted context & metadata
    user_intent: Optional[str]
    customer_id: Optional[str]
    policy_number: Optional[str]
    claim_id: Optional[str]

    # Supervisor / routing layer
    next_agent: Optional[str]
    task: Optional[str]
    justification: Optional[str]
    end_conversation: Optional[bool]

    # Clarification (pause/resume) flow
    needs_clarification: Optional[bool]
    clarification_question: Optional[str]
    user_clarification: Optional[str]

    # Entity extraction and DB lookups
    extracted_entities: Dict[str, Any]
    database_lookup_result: Dict[str, Any]
    retrieved_faqs: Optional[List[Any]]

    # Escalation state
    escalate_to_human: Optional[bool]
    requires_human_escalation: bool
    escalation_reason: Optional[str]

    # Billing-specific fields
    billing_amount: Optional[float]
    payment_method: Optional[str]
    billing_frequency: Optional[str]
    invoice_date: Optional[str]

    # System-level metadata
    timestamp: Optional[str]
    final_answer: Optional[str]


def new_initial_state(session_id: str, user_input: str, conversation_history: str) -> GraphState:
    """Build a fresh GraphState for a brand-new conversation turn."""
    return {
        "session_id": session_id,
        "messages": [],
        "user_input": user_input,
        "user_intent": "",
        "claim_id": "",
        "next_agent": "supervisor_agent",
        "extracted_entities": {},
        "database_lookup_result": {},
        "requires_human_escalation": False,
        "escalation_reason": "",
        "billing_amount": None,
        "payment_method": None,
        "billing_frequency": None,
        "invoice_date": None,
        "conversation_history": conversation_history,
        "task": "Help user with their query",
        "final_answer": "",
        "n_iteration": 0,
        "needs_clarification": False,
    }

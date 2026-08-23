"""OpenAI function-calling tool schemas, mapped to the DB repository functions.

Centralized here (rather than redefined inline in each agent node, as in the
original notebook) so the schema and the Python implementation can't drift
apart, and so they're easy to unit test.
"""

from __future__ import annotations

from app.db.repository import (
    get_auto_policy_details,
    get_billing_info,
    get_claim_status,
    get_payment_history,
    get_policy_details,
)

POLICY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_policy_details",
            "description": "Fetch policy info by policy number",
            "parameters": {
                "type": "object",
                "properties": {"policy_number": {"type": "string"}},
                "required": ["policy_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_auto_policy_details",
            "description": "Get auto policy details (vehicle info, deductibles)",
            "parameters": {
                "type": "object",
                "properties": {"policy_number": {"type": "string"}},
                "required": ["policy_number"],
            },
        },
    },
]
POLICY_TOOL_FUNCTIONS = {
    "get_policy_details": get_policy_details,
    "get_auto_policy_details": get_auto_policy_details,
}

BILLING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_billing_info",
            "description": "Retrieve billing information (balance, due date, premium)",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_number": {"type": "string"},
                    "customer_id": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_payment_history",
            "description": "Fetch recent payment history for a policy",
            "parameters": {
                "type": "object",
                "properties": {"policy_number": {"type": "string"}},
                "required": ["policy_number"],
            },
        },
    },
]
BILLING_TOOL_FUNCTIONS = {
    "get_billing_info": get_billing_info,
    "get_payment_history": get_payment_history,
}

CLAIMS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_claim_status",
            "description": "Retrieve claim details by claim ID or policy number",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "policy_number": {"type": "string"},
                },
            },
        },
    },
]
CLAIMS_TOOL_FUNCTIONS = {"get_claim_status": get_claim_status}

ASK_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user for clarification or additional information when their query is "
            "unclear or missing important details. ONLY use this if essential information "
            "like policy number or customer ID is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question to ask the user for clarification",
                },
                "missing_info": {
                    "type": "string",
                    "description": "What specific information is missing or needs clarification",
                },
            },
            "required": ["question", "missing_info"],
        },
    },
}

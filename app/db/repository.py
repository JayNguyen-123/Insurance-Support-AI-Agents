"""Read-only data access functions used as LLM tool calls by the agents.

These are direct ports of the notebook's `get_policy_details`,
`get_claim_status`, `get_billing_info`, `get_payment_history`, and
`get_auto_policy_details`, with three production fixes:

1. Connections are acquired via `session.get_connection()` (context manager)
   instead of a bare `sqlite3.connect()` per call, so they're always closed.
2. SQL errors are caught and returned as `{"error": ...}` payloads (matching
   the existing "tool result" contract the agents already expect) instead of
   raising and crashing the request.
3. `policy_number` / `claim_id` / `customer_id` inputs are stripped and
   validated to be non-empty before hitting the DB.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.db.session import get_connection, row_to_dict, rows_to_dicts
from app.logging_config import get_logger

logger = get_logger(__name__)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_policy_details(policy_number: str) -> dict[str, Any]:
    """Fetch a customer's policy details by policy number."""
    policy_number = _clean(policy_number)
    if not policy_number:
        return {"error": "policy_number is required"}

    logger.info("Fetching policy details", extra={"policy_number": policy_number})
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT p.*, c.first_name, c.last_name
                FROM policies p
                JOIN customers c ON p.customer_id = c.customer_id
                WHERE p.policy_number = ?
                """,
                (policy_number,),
            )
            result = row_to_dict(cursor.fetchone())
    except sqlite3.Error as exc:
        logger.exception("DB error in get_policy_details")
        return {"error": f"Database error: {exc}"}

    if result:
        return result
    logger.warning("Policy not found", extra={"policy_number": policy_number})
    return {"error": "Policy not found"}


def get_claim_status(claim_id: str | None = None, policy_number: str | None = None) -> Any:
    """Get claim status and details, by claim_id or (fallback) policy_number."""
    claim_id = _clean(claim_id)
    policy_number = _clean(policy_number)
    if not claim_id and not policy_number:
        return {"error": "claim_id or policy_number is required"}

    logger.info(
        "Fetching claim status", extra={"claim_id": claim_id, "policy_number": policy_number}
    )
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            if claim_id:
                cursor.execute(
                    """
                    SELECT c.*, p.policy_type
                    FROM claims c
                    JOIN policies p ON c.policy_number = p.policy_number
                    WHERE c.claim_id = ?
                    """,
                    (claim_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT c.*, p.policy_type
                    FROM claims c
                    JOIN policies p ON c.policy_number = p.policy_number
                    WHERE c.policy_number = ?
                    ORDER BY c.claim_date DESC LIMIT 3
                    """,
                    (policy_number,),
                )
            rows = rows_to_dicts(cursor.fetchall())
    except sqlite3.Error as exc:
        logger.exception("DB error in get_claim_status")
        return {"error": f"Database error: {exc}"}

    if rows:
        return rows
    logger.warning("No claims found", extra={"claim_id": claim_id, "policy_number": policy_number})
    return {"error": "Claim not found"}


def get_billing_info(policy_number: str | None = None, customer_id: str | None = None) -> dict[str, Any]:
    """Get current (pending) billing information: balance, due date, premium."""
    policy_number = _clean(policy_number)
    customer_id = _clean(customer_id)
    if not policy_number and not customer_id:
        return {"error": "policy_number or customer_id is required"}

    logger.info(
        "Fetching billing info", extra={"policy_number": policy_number, "customer_id": customer_id}
    )
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            if policy_number:
                cursor.execute(
                    """
                    SELECT b.*, p.premium_amount, p.billing_frequency
                    FROM billing b
                    JOIN policies p ON b.policy_number = p.policy_number
                    WHERE b.policy_number = ? AND b.status = 'pending'
                    ORDER BY b.due_date DESC LIMIT 1
                    """,
                    (policy_number,),
                )
            else:
                cursor.execute(
                    """
                    SELECT b.*, p.premium_amount, p.billing_frequency
                    FROM billing b
                    JOIN policies p ON b.policy_number = p.policy_number
                    WHERE p.customer_id = ? AND b.status = 'pending'
                    ORDER BY b.due_date DESC LIMIT 1
                    """,
                    (customer_id,),
                )
            result = row_to_dict(cursor.fetchone())
    except sqlite3.Error as exc:
        logger.exception("DB error in get_billing_info")
        return {"error": f"Database error: {exc}"}

    if result:
        return result
    logger.warning("Billing info not found")
    return {"error": "Billing information not found"}


def get_payment_history(policy_number: str) -> list[dict[str, Any]]:
    """Get the most recent payment records for a policy."""
    policy_number = _clean(policy_number)
    if not policy_number:
        return []

    logger.info("Fetching payment history", extra={"policy_number": policy_number})
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT p.payment_date, p.amount, p.status, p.payment_method
                FROM payments p
                JOIN billing b ON p.bill_id = b.bill_id
                WHERE b.policy_number = ?
                ORDER BY p.payment_date DESC LIMIT 10
                """,
                (policy_number,),
            )
            rows = rows_to_dicts(cursor.fetchall())
    except sqlite3.Error as exc:
        logger.exception("DB error in get_payment_history")
        return [{"error": f"Database error: {exc}"}]

    if rows:
        return rows
    logger.warning("No payment history found", extra={"policy_number": policy_number})
    return []


def get_auto_policy_details(policy_number: str) -> dict[str, Any]:
    """Get auto-specific policy details: vehicle info and deductibles."""
    policy_number = _clean(policy_number)
    if not policy_number:
        return {"error": "policy_number is required"}

    logger.info("Fetching auto policy details", extra={"policy_number": policy_number})
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT apd.*, p.policy_type, p.premium_amount
                FROM auto_policy_details apd
                JOIN policies p ON apd.policy_number = p.policy_number
                WHERE apd.policy_number = ?
                """,
                (policy_number,),
            )
            result = row_to_dict(cursor.fetchone())
    except sqlite3.Error as exc:
        logger.exception("DB error in get_auto_policy_details")
        return {"error": f"Database error: {exc}"}

    if result:
        return result
    logger.warning("Auto policy details not found", extra={"policy_number": policy_number})
    return {"error": "Auto policy details not found"}

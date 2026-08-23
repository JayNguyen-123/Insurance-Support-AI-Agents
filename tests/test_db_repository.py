from __future__ import annotations

from app.db import repository


def test_get_policy_details_found(sample_policy_number):
    result = repository.get_policy_details(sample_policy_number)
    assert "error" not in result
    assert result["policy_number"] == sample_policy_number
    assert result["policy_type"] == "auto"
    assert "first_name" in result


def test_get_policy_details_not_found(seeded_db):
    result = repository.get_policy_details("POL999999")
    assert result == {"error": "Policy not found"}


def test_get_policy_details_requires_input(seeded_db):
    result = repository.get_policy_details("")
    assert "error" in result


def test_get_auto_policy_details_found(sample_policy_number):
    result = repository.get_auto_policy_details(sample_policy_number)
    assert "error" not in result
    assert result["policy_number"] == sample_policy_number
    assert result["vehicle_make"]


def test_get_auto_policy_details_not_auto(seeded_db):
    from app.db.session import get_connection

    with get_connection() as conn:
        row = conn.execute(
            "SELECT policy_number FROM policies WHERE policy_type != 'auto' LIMIT 1"
        ).fetchone()
    assert row is not None
    result = repository.get_auto_policy_details(row["policy_number"])
    assert result == {"error": "Auto policy details not found"}


def test_get_billing_info_by_policy(sample_policy_number):
    result = repository.get_billing_info(policy_number=sample_policy_number)
    # May or may not have a pending bill (random data), but must not error on shape.
    assert isinstance(result, dict)


def test_get_billing_info_requires_input(seeded_db):
    result = repository.get_billing_info()
    assert result == {"error": "policy_number or customer_id is required"}


def test_get_payment_history_returns_list(sample_policy_number):
    result = repository.get_payment_history(sample_policy_number)
    assert isinstance(result, list)


def test_get_claim_status_no_input_errors(seeded_db):
    result = repository.get_claim_status()
    assert result == {"error": "claim_id or policy_number is required"}


def test_get_claim_status_by_policy_number(seeded_db):
    from app.db.session import get_connection

    with get_connection() as conn:
        row = conn.execute("SELECT policy_number FROM claims LIMIT 1").fetchone()
    assert row is not None
    result = repository.get_claim_status(policy_number=row["policy_number"])
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0]["policy_number"] == row["policy_number"]


def test_get_claim_status_by_claim_id(seeded_db):
    from app.db.session import get_connection

    with get_connection() as conn:
        row = conn.execute("SELECT claim_id FROM claims LIMIT 1").fetchone()
    assert row is not None
    result = repository.get_claim_status(claim_id=row["claim_id"])
    assert isinstance(result, list)
    assert result[0]["claim_id"] == row["claim_id"]

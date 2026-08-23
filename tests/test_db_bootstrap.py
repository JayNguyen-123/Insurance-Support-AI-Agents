"""Regression test for a bug in the original notebook: the DROP script
referenced a table named `billings` while the table is actually created as
`billing`, so re-running the setup on an already-seeded DB silently failed
to drop (and therefore duplicated) the billing rows on every re-run.
"""

from __future__ import annotations

from app.db.bootstrap import setup_insurance_database
from app.db.session import get_connection


def test_setup_is_idempotent_on_rerun(isolated_env):
    setup_insurance_database()
    with get_connection() as conn:
        first_count = conn.execute("SELECT COUNT(*) FROM billing").fetchone()[0]

    setup_insurance_database()
    with get_connection() as conn:
        second_count = conn.execute("SELECT COUNT(*) FROM billing").fetchone()[0]

    assert first_count == second_count == 5000


def test_state_codes_are_not_concatenated(isolated_env):
    """Regression test for a typo (`'OH' 'GA'`, missing comma) in the
    original notebook that silently produced a single bogus state code
    'OHGA' via Python string literal concatenation."""
    setup_insurance_database()
    with get_connection() as conn:
        states = {row[0] for row in conn.execute("SELECT DISTINCT state FROM customers")}
    assert "OHGA" not in states
    assert states <= {"CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA"}

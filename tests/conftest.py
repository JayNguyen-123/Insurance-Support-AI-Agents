from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure `app` package is importable regardless of CWD the test runner uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path, monkeypatch):
    """Point every stateful path (DB, vector store, logs) at a fresh tmp dir,
    and clear the cached Settings singleton so the new env vars take effect.
    """
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "test.log"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("FAQ_USE_HUGGINGFACE_DATASET", "false")
    monkeypatch.setenv("SUPERVISOR_MAX_ITERATIONS", "5")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "")

    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture()
def seeded_db(isolated_env):
    from app.db.bootstrap import setup_insurance_database

    setup_insurance_database()
    return isolated_env


@pytest.fixture()
def seeded_faqs(isolated_env):
    from app.vectorstore.faq_store import seed_faq_collection

    seed_faq_collection()
    return isolated_env


@pytest.fixture()
def sample_policy_number(seeded_db):
    from app.db.session import get_connection

    with get_connection() as conn:
        row = conn.execute(
            "SELECT policy_number FROM policies WHERE policy_type = 'auto' AND status = 'active' LIMIT 1"
        ).fetchone()
    assert row is not None
    return row["policy_number"]

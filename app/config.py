"""Centralized application configuration.

All runtime configuration is read from environment variables (optionally via a
local `.env` file during development). Nothing here is hardcoded the way the
original notebook hardcoded API keys, DB paths, and prompt endpoints -- every
value is overridable per-environment (dev / staging / prod) without code
changes.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-5-mini")
    openai_request_timeout_seconds: float = Field(default=30.0)
    openai_max_retries: int = Field(default=3)

    # --- Database ---
    database_path: str = Field(default="./data/insurance_support.db")

    # --- Vector store ---
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    chroma_collection_name: str = Field(default="insurance_faq_collection")
    faq_use_huggingface_dataset: bool = Field(default=False)
    faq_sample_size: int = Field(default=200)

    # --- Supervisor / conversation ---
    # Bounds how many supervisor *decision points* (ask-for-clarification,
    # dispatch-to-specialist, or confirm-done) a single logical exchange can
    # take before forcing a human escalation. A clean single-specialist path
    # with one clarification round-trip is already 3 decision points (ask,
    # route, confirm-done), so this needs headroom above that -- see the
    # `test_clarification_pause_then_resume_across_two_calls` test and its
    # comment for how a too-tight value here causes premature escalation.
    supervisor_max_iterations: int = Field(default=5)
    session_ttl_seconds: int = Field(default=1800)

    # --- Observability ---
    phoenix_collector_endpoint: str = Field(default="")
    phoenix_project_name: str = Field(default="insurance-support-agents")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # "json" | "text"
    log_file: str = Field(default="./data/insurance_agent.log")

    # --- API ---
    api_cors_origins: str = Field(default="*")

    @field_validator("openai_api_key")
    @classmethod
    def _warn_on_missing_key(cls, v: str) -> str:
        # Intentionally not raising here: config is imported at module load
        # time in many places (including tests), and we want failures to
        # surface as a clear 503 from the health/chat endpoints rather than
        # crashing import. See app.main for the startup check.
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        if self.api_cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached because pydantic-settings re-reads and re-validates the
    environment on every instantiation, which we don't want to pay for on
    every request. Tests that need to override env vars should call
    `get_settings.cache_clear()` after monkeypatching the environment.
    """
    return Settings()

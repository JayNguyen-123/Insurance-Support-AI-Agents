"""Pydantic request/response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Existing session ID to continue a conversation. Omit to start a new session.",
    )
    message: str = Field(..., min_length=1, max_length=4000, description="The user's message.")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    requires_clarification: bool
    requires_human_escalation: bool
    done: bool
    agent_used: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_store: str
    llm_configured: bool

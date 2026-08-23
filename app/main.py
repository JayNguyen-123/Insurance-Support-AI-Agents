"""FastAPI application: the production entry point for the insurance support
multi-agent system.

Replaces the notebook's `run_test_query` (a single blocking, stateless call
per query, with a real `input()` prompt for clarifications) with a proper
HTTP API: sessions persist conversation history across requests, and
clarification questions pause the graph and return to the caller instead of
blocking a server thread.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.bootstrap import setup_insurance_database
from app.db.session import get_connection
from app.logging_config import configure_logging, get_logger
from app.models.api_models import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from app.services.conversation_service import ConversationServiceError, handle_chat_turn, reset_session
from app.tracing import init_tracing
from app.vectorstore.faq_store import get_collection, seed_faq_collection

logger = get_logger(__name__)


def _ensure_database_ready() -> None:
    """Idempotent bootstrap: create + seed the DB on first boot only.

    Safe to run on every startup -- it only acts when the `customers` table
    is missing or empty, so restarts of an already-seeded deployment are a
    no-op. For a multi-instance deployment, prefer running
    `scripts/seed_db.py` once as a separate init step and disabling this by
    pointing DATABASE_PATH at an already-seeded volume.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='customers'"
            )
            table_exists = cursor.fetchone() is not None
            count = 0
            if table_exists:
                count = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if not table_exists or count == 0:
            logger.info("Database empty or missing; seeding synthetic sample data...")
            setup_insurance_database()
    except Exception:
        logger.exception("Database readiness check failed; the app will still start, but /health will report it.")


def _ensure_faq_collection_ready() -> None:
    try:
        collection = get_collection()
        if collection.count() == 0:
            logger.info("FAQ vector store empty; seeding...")
            seed_faq_collection()
    except Exception:
        logger.exception("FAQ vector store readiness check failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format, log_file=settings.log_file)
    logger.info("Starting Insurance Support AI Agents API")
    init_tracing()
    _ensure_database_ready()
    _ensure_faq_collection_ready()
    yield
    logger.info("Shutting down Insurance Support AI Agents API")


app = FastAPI(
    title="Insurance Support AI Agents",
    description="Multi-agent (LangGraph) insurance customer support API: policy, billing, claims, FAQ, and human escalation.",
    version="1.0.0",
    lifespan=lifespan,
)

_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_for_cors.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="internal_error", detail="An unexpected error occurred.").model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    settings = get_settings()

    db_status = "ok"
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception:
        logger.exception("Health check: database unavailable")
        db_status = "error"

    vector_status = "ok"
    try:
        get_collection().count()
    except Exception:
        logger.exception("Health check: vector store unavailable")
        vector_status = "error"

    overall = HealthResponse(
        status="ok" if db_status == "ok" and vector_status == "ok" else "degraded",
        database=db_status,
        vector_store=vector_status,
        llm_configured=bool(settings.openai_api_key),
    )
    return overall


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["chat"],
)
async def chat(payload: ChatRequest) -> ChatResponse | JSONResponse:
    settings = get_settings()
    if not settings.openai_api_key:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="llm_not_configured", detail="OPENAI_API_KEY is not configured on the server."
            ).model_dump(),
        )

    try:
        result = handle_chat_turn(payload.session_id, payload.message)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error="bad_request", detail=str(exc)).model_dump(),
        )
    except ConversationServiceError as exc:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(error="upstream_error", detail=str(exc)).model_dump(),
        )

    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        requires_clarification=result.requires_clarification,
        requires_human_escalation=result.requires_human_escalation,
        done=result.done,
        agent_used=result.agent_used,
    )


@app.post("/api/v1/sessions/{session_id}/reset", status_code=status.HTTP_204_NO_CONTENT, tags=["chat"])
async def reset(session_id: str) -> None:
    reset_session(session_id)
    return None

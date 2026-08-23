"""Optional agent-level tracing via Arize Phoenix / OpenTelemetry.

The original notebook called `getpass.getpass("Enter Phoenix Collector
Endpoint: ")` at import time -- which blocks forever waiting on stdin in any
non-interactive process (a web server, a test run, a container). Tracing is
now entirely config-driven and optional: set `PHOENIX_COLLECTOR_ENDPOINT` to
enable it. If it's unset, or the `arize-phoenix-otel` package isn't
installed, or the collector is unreachable, `trace_agent` becomes a no-op
decorator and the app runs normally -- tracing should never be able to take
the service down.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_tracer: Any = None
_tracing_enabled = False


def _try_init_tracer() -> None:
    global _tracer, _tracing_enabled
    settings = get_settings()
    if not settings.phoenix_collector_endpoint:
        logger.info("PHOENIX_COLLECTOR_ENDPOINT not set; agent tracing disabled.")
        return

    try:
        from phoenix.otel import register

        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_collector_endpoint,
            auto_instrument=True,
        )
        _tracer = tracer_provider.get_tracer(__name__)
        _tracing_enabled = True
        logger.info("Agent tracing enabled.", extra={"endpoint": settings.phoenix_collector_endpoint})
    except Exception:
        logger.exception("Failed to initialize Phoenix tracing; continuing without it.")
        _tracer = None
        _tracing_enabled = False


def trace_agent(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an agent node function in a tracing span, if tracing is enabled."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _tracing_enabled or _tracer is None:
            return func(*args, **kwargs)

        from opentelemetry.trace import Status, StatusCode

        state = args[0] if args else {}
        agent_name = func.__name__

        with _tracer.start_as_current_span(agent_name) as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.type", agent_name.replace("_node", ""))
            span.set_attribute("session.id", str(state.get("session_id", "unknown")))
            span.set_attribute("customer.id", str(state.get("customer_id", "unknown")))
            span.set_attribute("policy.number", str(state.get("policy_number", "unknown")))
            span.set_attribute("claim.id", str(state.get("claim_id", "unknown")))
            span.set_attribute("task", str(state.get("task", "none")))

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                span.set_attribute("execution.duration_sec", time.time() - start_time)
                if isinstance(result, dict):
                    span.set_attribute("result.keys", list(result.keys()))
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapper


def init_tracing() -> None:
    """Call once at app startup."""
    _try_init_tracer()

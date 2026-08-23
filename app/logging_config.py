"""Structured logging setup.

The original notebook called `logging.basicConfig(...)` at import time with a
hardcoded log file path -- fine in a Colab cell, but it means importing the
module for a unit test writes to disk and duplicate-registers handlers on
re-import. Here, configuration is explicit, idempotent, and driven by
Settings so log level/format/destination are environment-controlled.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any structured extras (e.g. session_id, agent) passed via `extra=`.
        for key, value in record.__dict__.items():
            if key in (
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "taskName",
            ):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json", log_file: str | None = None) -> None:
    """Configure the root logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    formatter: logging.Formatter
    if fmt == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            # Non-fatal: e.g. read-only filesystem in some deployment targets.
            root.warning("Could not open log file %s; logging to stdout only.", log_file)

    logging.Formatter.converter = time.gmtime
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

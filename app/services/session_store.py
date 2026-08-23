"""In-memory, TTL-bound session store.

Each chat session needs two things to survive across separate HTTP requests:
the running `conversation_history` text, and -- only while paused awaiting a
clarification answer -- the in-flight `GraphState` needed to resume the
LangGraph run where it left off (see `app/agents/supervisor.py` and
`app/agents/graph.py`).

This is a single-process, in-memory implementation: sessions live in a plain
dict guarded by a lock, with lazy TTL eviction on access. That's a
deliberate, documented limitation of this baseline -- it means sessions
don't survive a restart and aren't shared across multiple worker processes.
For a real multi-worker/multi-instance deployment, swap this module for a
Redis- or database-backed store; every call site only depends on the small
interface below (`get`, `put`, `delete`), so no other code needs to change.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import get_settings


@dataclass
class SessionRecord:
    session_id: str
    conversation_history: str = ""
    pending_state: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def _ttl(self) -> int:
        return self._ttl_seconds if self._ttl_seconds is not None else get_settings().session_ttl_seconds

    def _evict_expired_locked(self) -> None:
        ttl = self._ttl()
        now = time.time()
        expired = [sid for sid, rec in self._sessions.items() if now - rec.updated_at > ttl]
        for sid in expired:
            del self._sessions[sid]

    def create(self) -> SessionRecord:
        with self._lock:
            self._evict_expired_locked()
            session_id = str(uuid.uuid4())
            record = SessionRecord(session_id=session_id)
            self._sessions[session_id] = record
            return record

    def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            self._evict_expired_locked()
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> SessionRecord:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing
        with self._lock:
            self._evict_expired_locked()
            record = SessionRecord(session_id=session_id or str(uuid.uuid4()))
            self._sessions[record.session_id] = record
            return record

    def save(self, record: SessionRecord) -> None:
        record.updated_at = time.time()
        with self._lock:
            self._sessions[record.session_id] = record

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self._sessions)


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store

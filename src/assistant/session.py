"""
Session management for the Voxline Assistant.

A session isolates conversation state by mode (chat, business, coding).
Sessions do not share context unless explicitly merged.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionMode(Enum):
    """Assistant operating mode."""
    CHAT = "chat"
    BUSINESS = "business"
    CODING = "coding"


@dataclass
class Session:
    """An isolated assistant session."""
    session_id: str
    mode: SessionMode
    created_at: str
    updated_at: str
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Append a message and update timestamp."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
        })
        self.updated_at = _now_iso()

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Return conversation messages, optionally capped to the most recent."""
        if limit is not None and limit > 0:
            return list(self.history[-limit:])
        return list(self.history)

    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        self.updated_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history_length": len(self.history),
            "metadata": self.metadata,
        }


class SessionManager:
    """
    In-memory session store.

    Sessions are keyed by session_id. Each session belongs to exactly one mode.
    The manager does not persist to disk — session history lives for the
    lifetime of the process.
    """

    def __init__(self, max_sessions: int = 100):
        self._sessions: Dict[str, Session] = {}
        self._max_sessions = max_sessions

    def create(
        self,
        mode: SessionMode,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session."""
        if len(self._sessions) >= self._max_sessions:
            self._evict_oldest()

        sid = session_id or _generate_id()
        now = _now_iso()
        session = Session(
            session_id=sid,
            mode=mode,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Get session by ID, or None."""
        return self._sessions.get(session_id)

    def get_or_create(
        self,
        session_id: Optional[str],
        mode: SessionMode,
    ) -> Session:
        """Get existing session or create one. If session_id is None, always creates."""
        if session_id is None:
            return self.create(mode)
        session = self.get(session_id)
        if session is None:
            return self.create(mode, session_id=session_id)
        return session

    def list_sessions(self) -> List[Session]:
        """Return all active sessions."""
        return list(self._sessions.values())

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def _evict_oldest(self) -> None:
        """Remove the session with the earliest updated_at."""
        if not self._sessions:
            return
        oldest_id = min(
            self._sessions,
            key=lambda sid: self._sessions[sid].updated_at,
        )
        del self._sessions[oldest_id]


def _now_iso() -> str:
    """Current time as ISO-8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _generate_id() -> str:
    """Generate a unique session ID."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"sess_{ts}_{short}"

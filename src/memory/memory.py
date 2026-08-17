"""
Long-term memory system for Voxline AI Core

Separates external memory from model weights:
- Model weights: learned during training
- External memory: retrieved dynamically during inference

Supports:
- Episodic memory (conversation history, events)
- Semantic memory (facts, knowledge)
- Preferences
- Timestamps and sources
- Retrieval by relevance
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sqlite3


@dataclass
class MemoryEntry:
    """Single memory entry."""

    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "preference"
    timestamp: str
    source: str  # "conversation", "user_input", "system"
    relevant_keywords: List[str]
    tags: List[str]
    embedding_vector: Optional[List[float]] = None  # For semantic search
    metadata: Dict = None

    def to_dict(self):
        return asdict(self)


class MemoryStore:
    """Persistent memory storage."""

    def __init__(self, db_path: str = "memory/voxline_memory.db"):
        """
        Initialize memory store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                keywords TEXT NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT
            )
        """
        )
        self.conn.commit()

    def add_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        source: str = "conversation",
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Add memory entry.

        Args:
            content: Memory content
            memory_type: Type of memory
            source: Source of memory
            keywords: Keywords for retrieval
            tags: Tags for organization
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        import uuid

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        keywords = keywords or []
        tags = tags or []

        self.cursor.execute(
            """
            INSERT INTO memories (id, content, memory_type, timestamp, source, keywords, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                memory_id,
                content,
                memory_type,
                timestamp,
                source,
                json.dumps(keywords),
                json.dumps(tags),
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()

        return memory_id

    def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Search memories by content.

        Args:
            query: Search query
            memory_type: Optional filter by memory type
            limit: Maximum results

        Returns:
            List of matching memories
        """
        if memory_type:
            self.cursor.execute(
                """
                SELECT * FROM memories 
                WHERE (content LIKE ? OR keywords LIKE ?)
                AND memory_type = ?
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", memory_type, limit),
            )
        else:
            self.cursor.execute(
                """
                SELECT * FROM memories 
                WHERE content LIKE ? OR keywords LIKE ?
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", limit),
            )

        rows = self.cursor.fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get specific memory by ID."""
        self.cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = self.cursor.fetchone()
        return self._row_to_entry(row) if row else None

    def update_memory(self, memory_id: str, content: str):
        """Update memory content."""
        self.cursor.execute(
            "UPDATE memories SET content = ? WHERE id = ?",
            (content, memory_id),
        )
        self.conn.commit()

    def delete_memory(self, memory_id: str):
        """Delete memory entry."""
        self.cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()

    def clear_memories(self, memory_type: Optional[str] = None):
        """
        Clear memories.

        Args:
            memory_type: If specified, only clear this type. Otherwise clear all.
        """
        if memory_type:
            self.cursor.execute("DELETE FROM memories WHERE memory_type = ?", (memory_type,))
        else:
            self.cursor.execute("DELETE FROM memories")
        self.conn.commit()

    def get_recent_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent memories."""
        self.cursor.execute(
            """
            SELECT * FROM memories 
            ORDER BY timestamp DESC 
            LIMIT ?
        """,
            (limit,),
        )
        rows = self.cursor.fetchall()
        return [self._row_to_entry(row) for row in rows]

    def _row_to_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            timestamp=row[3],
            source=row[4],
            relevant_keywords=json.loads(row[5]),
            tags=json.loads(row[6]),
            metadata=json.loads(row[7]),
        )

    def close(self):
        """Close database connection."""
        self.conn.close()


class ConversationMemory:
    """Manages conversation history and context."""

    def __init__(self, memory_store: MemoryStore, max_history: int = 10):
        """
        Initialize conversation memory.

        Args:
            memory_store: Underlying memory store
            max_history: Maximum conversation history to keep
        """
        self.memory_store = memory_store
        self.max_history = max_history
        self.current_conversation = []

    def add_message(
        self,
        role: str,  # "user", "assistant", "system"
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Add message to conversation.

        Args:
            role: Message role
            content: Message content
            metadata: Additional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.current_conversation.append(message)

        # Store in permanent memory
        self.memory_store.add_memory(
            content=content,
            memory_type="episodic",
            source="conversation",
            tags=[role],
            metadata={"role": role, **metadata} if metadata else {"role": role},
        )

        # Trim if too long
        if len(self.current_conversation) > self.max_history:
            self.current_conversation.pop(0)

    def get_context(self, num_messages: Optional[int] = None) -> str:
        """
        Get conversation context as formatted string.

        Args:
            num_messages: Number of recent messages to include

        Returns:
            Formatted conversation context
        """
        messages = self.current_conversation
        if num_messages:
            messages = messages[-num_messages :]

        context = ""
        for msg in messages:
            role = msg["role"].upper()
            context += f"{role}: {msg['content']}\n"

        return context.strip()

    def get_messages(self, num_messages: Optional[int] = None) -> List[Dict]:
        """Get conversation messages."""
        if num_messages:
            return self.current_conversation[-num_messages :]
        return self.current_conversation

    def clear(self):
        """Clear current conversation."""
        self.current_conversation = []

    def export(self, path: str):
        """Export conversation to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.current_conversation, f, indent=2)

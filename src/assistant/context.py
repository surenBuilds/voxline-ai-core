"""
Context construction for the Voxline Assistant.

ContextBuilder is responsible for assembling the structured context
that gets sent to the AIProvider. It does NOT generate responses —
it only prepares input.

Context sections:
  - MEMORY: relevant facts retrieved from persistent memory
  - CONVERSATION: recent session history
  - BUSINESS_CONTEXT: business domain instructions (when in business mode)
  - WORKSPACE_CONTEXT: workspace path and file info (when in coding mode)
  - USER_REQUEST: the current user message

System instructions are NOT included here — providers inject them
via their own chat() method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.assistant.session import Session, SessionMode
from src.memory.memory import MemoryStore, MemoryEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode-specific instructions (user-facing context, not system prompts)
# ---------------------------------------------------------------------------

MODE_INSTRUCTIONS: Dict[SessionMode, str] = {
    SessionMode.CHAT: (
        "You are in conversational mode. "
        "Answer clearly and concisely. "
        "You may answer in any language the user writes in."
    ),
    SessionMode.BUSINESS: (
        "You are in business analysis mode. "
        "Provide structured analysis, actionable recommendations, "
        "and consider practical implications. "
        "Use professional, data-informed language."
    ),
    SessionMode.CODING: (
        "You are in coding assistant mode. "
        "Provide precise technical guidance. "
        "When suggesting code, use the appropriate syntax. "
        "Prefer minimal, correct solutions."
    ),
}


@dataclass
class Context:
    """Structured output of context construction."""

    messages: List[Dict[str, str]]
    """Ordered messages to pass to AIProvider.chat()."""

    memory_entries: List[MemoryEntry]
    """Memories that were included."""

    memory_section: str
    """Formatted memory block (may be empty)."""

    history_count: int
    """Number of history messages included."""

    total_characters: int
    """Approximate character count of the assembled context."""

    business_context: str
    """Business context block (empty if not in business mode)."""

    workspace_context: str
    """Workspace context block (empty if not in coding mode)."""

    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """
    Constructs structured context for the AIProvider.

    Responsible for:
    - retrieving relevant memories and including them
    - formatting session history
    - adding mode-specific context sections
    - preventing unbounded prompt growth

    The system instruction is NOT included — providers handle that
    via their chat() interface.
    """

    # Approximate character budget per section as fraction of max_chars
    MEMORY_BUDGET_RATIO = 0.25
    HISTORY_BUDGET_RATIO = 0.50
    OVERHEAD_BUDGET_RATIO = 0.25

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        max_chars: int = 4000,
        max_history: int = 20,
        max_memory_results: int = 5,
    ):
        """
        Args:
            memory_store: Persistent memory store. None disables memory retrieval.
            max_chars: Approximate maximum character budget for context.
            max_history: Maximum conversation history messages to include.
            max_memory_results: Maximum memory entries to retrieve.
        """
        self.memory_store = memory_store
        self.max_chars = max_chars
        self.max_history = max_history
        self.max_memory_results = max_memory_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        session: Session,
        user_message: str,
        mode_instructions: Optional[str] = None,
        business_context: Optional[str] = None,
        workspace_path: Optional[str] = None,
        language_instruction: Optional[str] = None,
    ) -> Context:
        """
        Build a Context from session state and user input.

        Args:
            session: Current session (carries history and mode).
            user_message: The user's current message.
            mode_instructions: Override for mode-specific instructions.
            business_context: Additional business domain context.
            workspace_path: Workspace root path (coding mode).
            language_instruction: Language-aware system instruction.
                When provided it is placed as the first system message
                in the list so the provider uses it instead of its own
                default.

        Returns:
            Context with ordered messages and metadata.
        """
        # Compute character budgets
        memory_budget = int(self.max_chars * self.MEMORY_BUDGET_RATIO)
        history_budget = int(self.max_chars * self.HISTORY_BUDGET_RATIO)

        # 1. Retrieve memory
        memory_entries = self._retrieve_memory(user_message)
        memory_section = self._format_memory(memory_entries, memory_budget)

        # 2. Format history
        history_messages = self._format_history(session, history_budget)

        # 3. Business context
        biz_ctx = self._format_business_context(
            business_context, session.mode
        )

        # 4. Workspace context
        ws_ctx = self._format_workspace_context(workspace_path)

        # 5. Mode instruction
        mode_inst = mode_instructions or MODE_INSTRUCTIONS.get(
            session.mode, ""
        )

        # 6. Assemble ordered messages
        messages = self._assemble_messages(
            memory_section=memory_section,
            mode_instruction=mode_inst,
            history=history_messages,
            business_context=biz_ctx,
            workspace_context=ws_ctx,
            user_message=user_message,
            language_instruction=language_instruction,
        )

        total_chars = sum(len(m.get("content", "")) for m in messages)

        return Context(
            messages=messages,
            memory_entries=memory_entries,
            memory_section=memory_section,
            history_count=len(history_messages),
            total_characters=total_chars,
            business_context=biz_ctx,
            workspace_context=ws_ctx,
        )

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    def _retrieve_memory(self, query: str) -> List[MemoryEntry]:
        """Retrieve relevant memories. Returns empty list on error or
        when no memory_store is configured."""
        if self.memory_store is None or not query.strip():
            return []
        try:
            return self.memory_store.search_memories(
                query=query,
                limit=self.max_memory_results,
            )
        except Exception:
            logger.debug("Memory retrieval failed; proceeding without memory")
            return []

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_memory(
        entries: List[MemoryEntry], budget: int
    ) -> str:
        """Format memory entries into a text block within the budget."""
        if not entries:
            return ""
        lines = []
        used = 0
        for entry in entries:
            line = f"- {entry.content}"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1  # +1 for newline
        return "\n".join(lines)

    def _format_history(
        self, session: Session, budget: int
    ) -> List[Dict[str, str]]:
        """Return recent history messages that fit within the budget.
        Preserves chronological order."""
        all_msgs = session.get_messages(limit=self.max_history)
        result: List[Dict[str, str]] = []
        used = 0
        # Walk from oldest to newest, dropping oldest if over budget
        for msg in all_msgs:
            content = msg.get("content", "")
            if used + len(content) > budget and result:
                break
            result.append({"role": msg["role"], "content": content})
            used += len(content) + 1
        return result

    @staticmethod
    def _format_business_context(
        business_context: Optional[str], mode: SessionMode
    ) -> str:
        """Return business context block if available."""
        if mode != SessionMode.BUSINESS:
            return ""
        if business_context:
            return business_context
        return ""

    @staticmethod
    def _format_workspace_context(
        workspace_path: Optional[str],
    ) -> str:
        """Return workspace context block if available."""
        if not workspace_path:
            return ""
        return f"Workspace: {workspace_path}"

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_messages(
        self,
        memory_section: str,
        mode_instruction: str,
        history: List[Dict[str, str]],
        business_context: str,
        workspace_context: str,
        user_message: str,
        language_instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Assemble the final ordered message list.

        Order (for provider.chat()):
          [language system] [memory] [mode instruction] [history...] [biz ctx] [ws ctx] [user msg]
        """
        messages: List[Dict[str, str]] = []

        # Language-aware system instruction (FIRST — provider respects this)
        if language_instruction:
            messages.append({
                "role": "system",
                "content": language_instruction,
            })

        # Memory context (single message if non-empty)
        if memory_section:
            messages.append({
                "role": "user",
                "content": (
                    f"[RELEVANT MEMORY]\n{memory_section}\n[/RELEVANT MEMORY]"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "I have noted the relevant context from memory.",
            })

        # Mode instruction
        if mode_instruction:
            messages.append({
                "role": "user",
                "content": f"[MODE INSTRUCTION]\n{mode_instruction}\n[/MODE INSTRUCTION]",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood.",
            })

        # Conversation history
        messages.extend(history)

        # Business context
        if business_context:
            messages.append({
                "role": "user",
                "content": (
                    f"[BUSINESS CONTEXT]\n{business_context}\n[/BUSINESS CONTEXT]"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "I have the business context.",
            })

        # Workspace context
        if workspace_context:
            messages.append({
                "role": "user",
                "content": (
                    f"[WORKSPACE CONTEXT]\n{workspace_context}\n[/WORKSPACE CONTEXT]"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "I have the workspace context.",
            })

        # User message (always last)
        messages.append({"role": "user", "content": user_message})

        return messages

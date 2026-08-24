"""
ChatAssistant — the main conversational interface for Voxline.

Architecture:
    User message
        → SessionManager (get/create session)
        → ContextBuilder (assemble memory, history, mode context)
        → AIProvider.chat()  (generate response)
        → Store messages in session
        → Optionally persist useful memory
        → Return AssistantResponse

ChatAssistant NEVER uses TextGenerator directly.
All intelligence flows through AIProvider, which is replaceable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.assistant.context import ContextBuilder, Context
from src.assistant.session import Session, SessionManager, SessionMode
from src.errors import (
    SessionNotFoundError,
    ProviderError,
    VoxlineError,
)
from src.memory.memory import MemoryStore
from src.providers.base import AIProvider, GenerationConfig

logger = logging.getLogger(__name__)

# Default system instruction (used by provider, not ContextBuilder)
_DEFAULT_SYSTEM = (
    "You are Voxline, a helpful bilingual AI assistant. "
    "Be concise, accurate, and respectful. "
    "Reply in the user's language."
)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


@dataclass
class AssistantResponse:
    """Structured response from the assistant."""

    text: str
    session_id: str
    mode: str
    provider_id: str
    model_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ChatAssistant
# ---------------------------------------------------------------------------


class ChatAssistant:
    """
    Main conversational assistant.

    Orchestrates session management, context construction, provider
    invocation, and optional memory persistence.

    The provider is the only source of intelligence — ChatAssistant
    contains no model logic.
    """

    # Heuristic keywords that indicate the user wants to remember something
    _MEMORY_KEYWORDS = frozenset({
        "remember", "note", "important", "don't forget",
        "save", "always", "never forget",
    })

    def __init__(
        self,
        provider: AIProvider,
        session_manager: SessionManager,
        context_builder: Optional[ContextBuilder] = None,
        memory_store: Optional[MemoryStore] = None,
        max_history: int = 20,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        system_instruction: Optional[str] = None,
    ):
        self.provider = provider
        self.session_manager = session_manager
        self.context_builder = context_builder or ContextBuilder(
            memory_store=memory_store,
            max_history=max_history,
        )
        self.memory_store = memory_store
        self.max_history = max_history
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_instruction = system_instruction or _DEFAULT_SYSTEM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        session_id: str,
        message: str,
        persist_memory: Optional[bool] = None,
    ) -> AssistantResponse:
        """
        Process a user message and return a response.

        Args:
            session_id: Session to use (must exist).
            message: User's message text.
            persist_memory: Force memory persistence. None = auto-detect.

        Returns:
            AssistantResponse with text, session info, and metadata.

        Raises:
            SessionNotFoundError: Session does not exist.
            ValueError: Message is empty.
            ProviderError: Provider failed to generate.
        """
        self._validate_message(message)
        session = self._get_session(session_id)
        return self._handle_mode(session, message, persist_memory)

    def new_session(
        self,
        mode: SessionMode = SessionMode.CHAT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create and return a new session."""
        return self.session_manager.create(mode, metadata=metadata)

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------

    def _handle_mode(
        self,
        session: Session,
        message: str,
        persist_memory: Optional[bool],
    ) -> AssistantResponse:
        """Route to the correct mode handler."""
        handlers = {
            SessionMode.CHAT: self._chat,
            SessionMode.BUSINESS: self._business,
            SessionMode.CODING: self._coding,
        }
        handler = handlers.get(session.mode)
        if handler is None:
            raise ProviderError(f"Unknown session mode: {session.mode}")
        return handler(session, message, persist_memory)

    # ------------------------------------------------------------------
    # Chat mode
    # ------------------------------------------------------------------

    def _chat(
        self,
        session: Session,
        message: str,
        persist_memory: Optional[bool],
    ) -> AssistantResponse:
        ctx = self.context_builder.build(
            session=session,
            user_message=message,
        )
        text = self._call_provider(ctx)
        session.add_message("user", message)
        session.add_message("assistant", text)
        self._maybe_persist_memory(message, persist_memory)
        return self._make_response(session, text)

    # ------------------------------------------------------------------
    # Business mode
    # ------------------------------------------------------------------

    def _business(
        self,
        session: Session,
        message: str,
        persist_memory: Optional[bool],
    ) -> AssistantResponse:
        biz_ctx = session.metadata.get("business_context", "")
        ctx = self.context_builder.build(
            session=session,
            user_message=message,
            business_context=biz_ctx,
        )
        text = self._call_provider(ctx)
        session.add_message("user", message)
        session.add_message("assistant", text)
        self._maybe_persist_memory(message, persist_memory)
        return self._make_response(session, text)

    # ------------------------------------------------------------------
    # Coding mode
    # ------------------------------------------------------------------

    def _coding(
        self,
        session: Session,
        message: str,
        persist_memory: Optional[bool],
    ) -> AssistantResponse:
        workspace = session.metadata.get("workspace_path", "")
        ctx = self.context_builder.build(
            session=session,
            user_message=message,
            workspace_path=workspace,
        )
        text = self._call_provider(ctx)
        session.add_message("user", message)
        session.add_message("assistant", text)
        return self._make_response(session, text)

    # ------------------------------------------------------------------
    # Provider interaction
    # ------------------------------------------------------------------

    def _call_provider(self, ctx: Context) -> str:
        """Call the provider with the context messages."""
        config = GenerationConfig(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already inside an event loop — run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.provider.chat(ctx.messages, config),
                    )
                    return future.result(timeout=60)
            else:
                return asyncio.run(self.provider.chat(ctx.messages, config))
        except Exception as exc:
            raise ProviderError(f"Provider failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_message(message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must be a non-empty string.")

    def _get_session(self, session_id: str) -> Session:
        session = self.session_manager.get(session_id)
        if session is None:
            raise SessionNotFoundError(
                f"Session '{session_id}' does not exist."
            )
        return session

    # ------------------------------------------------------------------
    # Memory persistence (selective)
    # ------------------------------------------------------------------

    def _maybe_persist_memory(
        self, message: str, persist_memory: Optional[bool]
    ) -> None:
        """Persist memory if explicitly requested or heuristic matches."""
        if self.memory_store is None:
            return

        should_persist = persist_memory is True or (
            persist_memory is None and self._looks_like_memory(message)
        )

        if should_persist:
            self._store_memory(message)

    def _looks_like_memory(self, message: str) -> bool:
        lower = message.lower()
        return any(kw in lower for kw in self._MEMORY_KEYWORDS)

    def _store_memory(self, content: str) -> None:
        try:
            self.memory_store.add_memory(
                content=content,
                memory_type="semantic",
                source="conversation",
                keywords=content.split()[:5],
                tags=["user_memory"],
            )
            logger.debug("Persisted memory: %s", content[:60])
        except Exception:
            logger.debug("Memory persistence failed", exc_info=True)

    # ------------------------------------------------------------------
    # Response construction
    # ------------------------------------------------------------------

    def _make_response(
        self, session: Session, text: str
    ) -> AssistantResponse:
        info = self.provider.get_model_info()
        return AssistantResponse(
            text=text,
            session_id=session.session_id,
            mode=session.mode.value,
            provider_id=info.provider_id,
            model_id=info.model_id,
            metadata={
                "history_length": len(session.history),
            },
        )

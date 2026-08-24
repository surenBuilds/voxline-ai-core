"""
Tests for Phase 7 Step 4: ChatAssistant.
"""

import unittest
from typing import Dict, List
from unittest.mock import MagicMock

from src.assistant.chat import ChatAssistant, AssistantResponse
from src.assistant.session import Session, SessionManager, SessionMode
from src.assistant.context import ContextBuilder
from src.providers.base import (
    AIProvider, GenerationConfig, ModelInfo, ProviderHealth, ProviderStatus,
)
from src.errors import SessionNotFoundError, ProviderError, VoxlineError
from src.memory.memory import MemoryStore


# ---------------------------------------------------------------------------
# Fake provider (no model loading)
# ---------------------------------------------------------------------------


class FakeProvider(AIProvider):
    """Deterministic fake provider for testing."""

    def __init__(self, response: str = "fake response"):
        self._response = response
        self.call_count = 0
        self.last_messages = None
        self.last_config = None

    @property
    def provider_id(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return "fake-model-v1"

    @property
    def supports_streaming(self) -> bool:
        return False

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        self.call_count += 1
        return self._response

    async def chat(
        self, messages: List[Dict[str, str]], config: GenerationConfig
    ) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.last_config = config
        return self._response

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            message="Fake provider is healthy",
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            provider_id=self.provider_id,
            model_type="fake",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant(
    response: str = "hello from fake",
    memory_store: MemoryStore = None,
    system_instruction: str = None,
) -> ChatAssistant:
    provider = FakeProvider(response=response)
    sm = SessionManager()
    ctx = ContextBuilder(memory_store=memory_store)
    return ChatAssistant(
        provider=provider,
        session_manager=sm,
        context_builder=ctx,
        memory_store=memory_store,
        system_instruction=system_instruction,
    )


# ---------------------------------------------------------------------------
# Successful chat
# ---------------------------------------------------------------------------


class TestChatAssistantBasic(unittest.TestCase):

    def test_successful_chat(self):
        asst = _make_assistant(response="Hi there!")
        session = asst.new_session(mode=SessionMode.CHAT)
        resp = asst.chat(session.session_id, "hello")
        self.assertEqual(resp.text, "Hi there!")

    def test_response_type(self):
        asst = _make_assistant()
        session = asst.new_session()
        resp = asst.chat(session.session_id, "test")
        self.assertIsInstance(resp, AssistantResponse)

    def test_response_session_id(self):
        asst = _make_assistant()
        session = asst.new_session()
        resp = asst.chat(session.session_id, "hi")
        self.assertEqual(resp.session_id, session.session_id)

    def test_response_mode(self):
        asst = _make_assistant()
        session = asst.new_session(mode=SessionMode.BUSINESS)
        resp = asst.chat(session.session_id, "analyze")
        self.assertEqual(resp.mode, "business")

    def test_response_provider_info(self):
        asst = _make_assistant()
        session = asst.new_session()
        resp = asst.chat(session.session_id, "hi")
        self.assertEqual(resp.provider_id, "fake")
        self.assertEqual(resp.model_id, "fake-model-v1")


# ---------------------------------------------------------------------------
# Session handling
# ---------------------------------------------------------------------------


class TestChatAssistantSession(unittest.TestCase):

    def test_session_not_found(self):
        asst = _make_assistant()
        with self.assertRaises(SessionNotFoundError):
            asst.chat("nonexistent", "hi")

    def test_new_session(self):
        asst = _make_assistant()
        session = asst.new_session(mode=SessionMode.CODING)
        self.assertEqual(session.mode, SessionMode.CODING)
        self.assertIsNotNone(session.session_id)

    def test_conversation_persistence(self):
        asst = _make_assistant()
        session = asst.new_session()
        asst.chat(session.session_id, "q1")
        asst.chat(session.session_id, "q2")
        asst.chat(session.session_id, "q3")
        self.assertEqual(len(session.history), 6)  # 3 user + 3 assistant

    def test_history_ordering(self):
        asst = _make_assistant()
        session = asst.new_session()
        asst.chat(session.session_id, "first")
        asst.chat(session.session_id, "second")
        user_msgs = [m for m in session.history if m["role"] == "user"]
        self.assertEqual(user_msgs[0]["content"], "first")
        self.assertEqual(user_msgs[1]["content"], "second")


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------


class TestChatAssistantContext(unittest.TestCase):

    def test_context_builder_called(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        ctx_builder = MagicMock(spec=ContextBuilder)
        mock_ctx = MagicMock()
        mock_ctx.messages = [{"role": "user", "content": "hi"}]
        ctx_builder.build.return_value = mock_ctx
        asst = ChatAssistant(
            provider=provider,
            session_manager=sm,
            context_builder=ctx_builder,
        )
        session = asst.new_session()
        asst.chat(session.session_id, "hi")
        ctx_builder.build.assert_called_once()

    def test_provider_receives_messages(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = ChatAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        asst.chat(session.session_id, "hello")
        self.assertIsNotNone(provider.last_messages)
        self.assertGreater(len(provider.last_messages), 0)

    def test_provider_last_message_is_user(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = ChatAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        asst.chat(session.session_id, "question")
        last = provider.last_messages[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"], "question")


# ---------------------------------------------------------------------------
# Provider invocation
# ---------------------------------------------------------------------------


class TestChatAssistantProvider(unittest.TestCase):

    def test_provider_called_exactly_once(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        asst = ChatAssistant(provider=provider, session_manager=sm)
        session = asst.new_session()
        asst.chat(session.session_id, "hi")
        self.assertEqual(provider.call_count, 1)

    def test_provider_failure_raises(self):
        class FailingProvider(AIProvider):
            @property
            def provider_id(self): return "fail"
            @property
            def model_id(self): return "fail-m"
            @property
            def supports_streaming(self): return False
            async def generate(self, prompt, config):
                raise RuntimeError("model crashed")
            async def chat(self, messages, config):
                raise RuntimeError("model crashed")
            async def health_check(self):
                return ProviderHealth(status=ProviderStatus.UNAVAILABLE, message="down")
            def get_model_info(self):
                return ModelInfo(model_id="fail-m", provider_id="fail", model_type="fail")

        sm = SessionManager()
        asst = ChatAssistant(provider=FailingProvider(), session_manager=sm)
        session = asst.new_session()
        with self.assertRaises(ProviderError):
            asst.chat(session.session_id, "hi")


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


class TestChatAssistantValidation(unittest.TestCase):

    def test_empty_message(self):
        asst = _make_assistant()
        session = asst.new_session()
        with self.assertRaises(ValueError):
            asst.chat(session.session_id, "")

    def test_whitespace_message(self):
        asst = _make_assistant()
        session = asst.new_session()
        with self.assertRaises(ValueError):
            asst.chat(session.session_id, "   ")

    def test_none_message(self):
        asst = _make_assistant()
        session = asst.new_session()
        with self.assertRaises((ValueError, TypeError)):
            asst.chat(session.session_id, None)


# ---------------------------------------------------------------------------
# Mode isolation
# ---------------------------------------------------------------------------


class TestChatAssistantModeIsolation(unittest.TestCase):

    def test_chat_vs_business_different_context(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        ctx_builder = MagicMock(spec=ContextBuilder)
        mock_ctx = MagicMock()
        mock_ctx.messages = []
        ctx_builder.build.return_value = mock_ctx
        asst = ChatAssistant(
            provider=provider, session_manager=sm, context_builder=ctx_builder,
        )
        chat_s = asst.new_session(mode=SessionMode.CHAT)
        biz_s = asst.new_session(mode=SessionMode.BUSINESS)
        asst.chat(chat_s.session_id, "hi")
        asst.chat(biz_s.session_id, "hi")
        calls = ctx_builder.build.call_args_list
        chat_kw = calls[0].kwargs
        biz_kw = calls[1].kwargs
        self.assertIsNone(chat_kw.get("business_context"))
        self.assertIsNotNone(biz_kw.get("business_context")) or \
            self.assertEqual(biz_kw.get("business_context"), "")

    def test_coding_session_gets_workspace(self):
        provider = FakeProvider(response="ok")
        sm = SessionManager()
        ctx_builder = MagicMock(spec=ContextBuilder)
        mock_ctx = MagicMock()
        mock_ctx.messages = []
        ctx_builder.build.return_value = mock_ctx
        asst = ChatAssistant(
            provider=provider, session_manager=sm, context_builder=ctx_builder,
        )
        session = asst.new_session(
            mode=SessionMode.CODING,
            metadata={"workspace_path": "/project"},
        )
        asst.chat(session.session_id, "list files")
        call_kw = ctx_builder.build.call_args.kwargs
        self.assertEqual(call_kw["workspace_path"], "/project")


# ---------------------------------------------------------------------------
# Memory persistence
# ---------------------------------------------------------------------------


class TestChatAssistantMemory(unittest.TestCase):

    def test_explicit_persist(self):
        fd, path = __import__("tempfile").mkstemp(suffix=".db")
        __import__("os").close(fd)
        store = MemoryStore(db_path=path)
        asst = _make_assistant(memory_store=store)
        session = asst.new_session()
        asst.chat(session.session_id, "random text", persist_memory=True)
        results = store.search_memories("random text")
        self.assertGreater(len(results), 0)
        store.close()

    def test_no_persist_by_default(self):
        fd, path = __import__("tempfile").mkstemp(suffix=".db")
        __import__("os").close(fd)
        store = MemoryStore(db_path=path)
        asst = _make_assistant(memory_store=store)
        session = asst.new_session()
        asst.chat(session.session_id, "just chatting")
        results = store.search_memories("just chatting")
        self.assertEqual(len(results), 0)
        store.close()

    def test_keyword_triggers_persist(self):
        fd, path = __import__("tempfile").mkstemp(suffix=".db")
        __import__("os").close(fd)
        store = MemoryStore(db_path=path)
        asst = _make_assistant(memory_store=store)
        session = asst.new_session()
        asst.chat(session.session_id, "please remember this fact")
        results = store.search_memories("remember")
        self.assertGreater(len(results), 0)
        store.close()

    def test_no_memory_store(self):
        """When no memory_store is provided, memory ops should be silent no-ops."""
        asst = _make_assistant(memory_store=None)
        session = asst.new_session()
        resp = asst.chat(session.session_id, "remember this")
        self.assertEqual(resp.text, "hello from fake")


# ---------------------------------------------------------------------------
# Repeated conversation turns
# ---------------------------------------------------------------------------


class TestChatAssistantMultiTurn(unittest.TestCase):

    def test_multiple_turns(self):
        asst = _make_assistant(response="reply")
        session = asst.new_session()
        for i in range(5):
            resp = asst.chat(session.session_id, f"turn {i}")
            self.assertEqual(resp.text, "reply")
        self.assertEqual(len(session.history), 10)  # 5 user + 5 assistant

    def test_repeated_same_message(self):
        asst = _make_assistant(response="same")
        session = asst.new_session()
        for _ in range(3):
            asst.chat(session.session_id, "same question")
        self.assertEqual(len(session.history), 6)

    def test_session_isolation_across_assistants(self):
        sm = SessionManager()
        asst1 = ChatAssistant(provider=FakeProvider("a"), session_manager=sm)
        asst2 = ChatAssistant(provider=FakeProvider("b"), session_manager=sm)
        s1 = asst1.new_session(mode=SessionMode.CHAT)
        s2 = asst2.new_session(mode=SessionMode.BUSINESS)
        asst1.chat(s1.session_id, "from chat")
        asst2.chat(s2.session_id, "from business")
        self.assertEqual(len(s1.history), 2)
        self.assertEqual(len(s2.history), 2)
        self.assertEqual(s1.history[0]["content"], "from chat")
        self.assertEqual(s2.history[0]["content"], "from business")


if __name__ == "__main__":
    unittest.main()

"""
Tests for Phase 7 Step 3: ContextBuilder.
"""

import unittest
import os
import tempfile

from src.assistant.session import Session, SessionManager, SessionMode
from src.assistant.context import ContextBuilder, Context, MODE_INSTRUCTIONS
from src.memory.memory import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(mode: SessionMode = SessionMode.CHAT, sid: str = "test") -> Session:
    """Create a bare session for testing."""
    return Session(
        session_id=sid,
        mode=mode,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )


def _memory_store_with_entries(count: int = 3) -> MemoryStore:
    """Create a temporary MemoryStore with sample entries."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = MemoryStore(db_path=path)
    for i in range(count):
        store.add_memory(
            content=f"Fact number {i}: the sky is blue",
            memory_type="semantic",
            source="system",
            keywords=[f"fact{i}", "sky", "color"],
            tags=["test"],
        )
    return store


# ---------------------------------------------------------------------------
# Basic context construction
# ---------------------------------------------------------------------------


class TestContextBuilderBasic(unittest.TestCase):

    def test_returns_context_object(self):
        builder = ContextBuilder()
        session = _make_session()
        ctx = builder.build(session, "hello")
        self.assertIsInstance(ctx, Context)

    def test_user_message_is_last(self):
        builder = ContextBuilder()
        session = _make_session()
        ctx = builder.build(session, "what is 2+2?")
        self.assertEqual(ctx.messages[-1]["role"], "user")
        self.assertEqual(ctx.messages[-1]["content"], "what is 2+2?")

    def test_empty_session_history(self):
        builder = ContextBuilder()
        session = _make_session()
        ctx = builder.build(session, "hi")
        self.assertEqual(ctx.history_count, 0)

    def test_metadata_default_empty(self):
        builder = ContextBuilder()
        session = _make_session()
        ctx = builder.build(session, "hi")
        self.assertIsInstance(ctx.metadata, dict)


# ---------------------------------------------------------------------------
# History handling
# ---------------------------------------------------------------------------


class TestContextBuilderHistory(unittest.TestCase):

    def test_includes_existing_history(self):
        builder = ContextBuilder()
        session = _make_session()
        session.add_message("user", "old question")
        session.add_message("assistant", "old answer")
        ctx = builder.build(session, "new question")
        self.assertEqual(ctx.history_count, 2)
        # History messages appear before the final user message
        history_roles = [m["role"] for m in ctx.messages[:-1]]
        self.assertIn("user", history_roles)
        self.assertIn("assistant", history_roles)

    def test_history_preserves_order(self):
        builder = ContextBuilder()
        session = _make_session()
        session.add_message("user", "q1")
        session.add_message("assistant", "a1")
        session.add_message("user", "q2")
        ctx = builder.build(session, "q3")
        # History should be q1, a1, q2 (3 messages), then memory/mode/user
        history = [m for m in ctx.messages if m["content"] in ("q1", "a1", "q2")]
        self.assertEqual(history[0]["content"], "q1")
        self.assertEqual(history[1]["content"], "a1")
        self.assertEqual(history[2]["content"], "q2")

    def test_history_respects_max_history(self):
        builder = ContextBuilder(max_history=3)
        session = _make_session()
        for i in range(10):
            session.add_message("user" if i % 2 == 0 else "assistant", f"msg{i}")
        ctx = builder.build(session, "final")
        # History should contain at most 3 messages
        history = [m for m in ctx.messages[:-1]
                   if m["content"].startswith("msg")]
        self.assertLessEqual(len(history), 3)

    def test_history_within_budget(self):
        builder = ContextBuilder(max_chars=200, max_history=100)
        session = _make_session()
        for i in range(50):
            session.add_message("user", f"message {i} " + "x" * 20)
        ctx = builder.build(session, "final")
        self.assertLessEqual(ctx.total_characters, 400)  # budget + mode instruction overhead


# ---------------------------------------------------------------------------
# Memory inclusion
# ---------------------------------------------------------------------------


class TestContextBuilderMemory(unittest.TestCase):

    def test_memory_entries_included(self):
        store = _memory_store_with_entries(3)
        builder = ContextBuilder(memory_store=store, max_memory_results=3)
        session = _make_session()
        ctx = builder.build(session, "sky")
        self.assertEqual(len(ctx.memory_entries), 3)
        store.close()

    def test_memory_section_format(self):
        store = _memory_store_with_entries(2)
        builder = ContextBuilder(memory_store=store, max_memory_results=2)
        session = _make_session()
        ctx = builder.build(session, "sky")
        self.assertIn("Fact number", ctx.memory_section)
        self.assertTrue(ctx.memory_section.startswith("- "))
        store.close()

    def test_empty_memory(self):
        builder = ContextBuilder()
        session = _make_session()
        ctx = builder.build(session, "hello")
        self.assertEqual(len(ctx.memory_entries), 0)
        self.assertEqual(ctx.memory_section, "")

    def test_no_memory_store(self):
        builder = ContextBuilder(memory_store=None)
        session = _make_session()
        ctx = builder.build(session, "hello")
        self.assertEqual(len(ctx.memory_entries), 0)

    def test_memory_reaches_messages(self):
        """The critical test: retrieved memory IS injected into provider input."""
        store = _memory_store_with_entries(1)
        builder = ContextBuilder(memory_store=store, max_memory_results=1)
        session = _make_session()
        ctx = builder.build(session, "sky")
        # Memory block should appear in messages before user message
        memory_msgs = [m for m in ctx.messages if "RELEVANT MEMORY" in m.get("content", "")]
        self.assertEqual(len(memory_msgs), 1)
        store.close()

    def test_memory_failure_handled_gracefully(self):
        """Memory retrieval errors should not crash the builder."""
        store = _memory_store_with_entries(1)
        store.close()  # Close the DB so queries fail
        builder = ContextBuilder(memory_store=store)
        session = _make_session()
        ctx = builder.build(session, "query")
        self.assertEqual(len(ctx.memory_entries), 0)

    def test_empty_query_skips_memory(self):
        store = _memory_store_with_entries(3)
        builder = ContextBuilder(memory_store=store)
        session = _make_session()
        ctx = builder.build(session, "   ")
        self.assertEqual(len(ctx.memory_entries), 0)
        store.close()


# ---------------------------------------------------------------------------
# Mode-specific context
# ---------------------------------------------------------------------------


class TestContextBuilderModes(unittest.TestCase):

    def test_chat_mode_instructions(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.CHAT)
        ctx = builder.build(session, "hello")
        mode_msgs = [m for m in ctx.messages if "MODE INSTRUCTION" in m.get("content", "")]
        self.assertEqual(len(mode_msgs), 1)
        self.assertIn("conversational", mode_msgs[0]["content"].lower())

    def test_business_mode_instructions(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.BUSINESS)
        ctx = builder.build(session, "analyze revenue")
        mode_msgs = [m for m in ctx.messages if "MODE INSTRUCTION" in m.get("content", "")]
        self.assertEqual(len(mode_msgs), 1)
        self.assertIn("business", mode_msgs[0]["content"].lower())

    def test_coding_mode_instructions(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.CODING)
        ctx = builder.build(session, "fix this bug")
        mode_msgs = [m for m in ctx.messages if "MODE INSTRUCTION" in m.get("content", "")]
        self.assertEqual(len(mode_msgs), 1)
        self.assertIn("coding", mode_msgs[0]["content"].lower())

    def test_mode_isolation(self):
        """Sessions in different modes produce different context."""
        builder = ContextBuilder()
        chat_ctx = builder.build(_make_session(mode=SessionMode.CHAT), "hi")
        biz_ctx = builder.build(_make_session(mode=SessionMode.BUSINESS), "hi")
        chat_mode_str = str([m for m in chat_ctx.messages if "MODE" in m.get("content", "")])
        biz_mode_str = str([m for m in biz_ctx.messages if "MODE" in m.get("content", "")])
        self.assertNotEqual(chat_mode_str, biz_mode_str)

    def test_custom_mode_instructions(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.CHAT)
        ctx = builder.build(session, "hi", mode_instructions="Custom mode text")
        mode_msgs = [m for m in ctx.messages if "MODE INSTRUCTION" in m.get("content", "")]
        self.assertIn("Custom mode text", mode_msgs[0]["content"])


# ---------------------------------------------------------------------------
# Business context
# ---------------------------------------------------------------------------


class TestContextBuilderBusinessContext(unittest.TestCase):

    def test_business_context_included(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.BUSINESS)
        ctx = builder.build(
            session, "analyze", business_context="Q3 revenue: $500K"
        )
        biz_msgs = [m for m in ctx.messages if "BUSINESS CONTEXT" in m.get("content", "")]
        self.assertEqual(len(biz_msgs), 1)
        self.assertIn("$500K", biz_msgs[0]["content"])

    def test_business_context_ignored_in_chat_mode(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.CHAT)
        ctx = builder.build(
            session, "hi", business_context="Q3 revenue: $500K"
        )
        biz_msgs = [m for m in ctx.messages if "BUSINESS CONTEXT" in m.get("content", "")]
        self.assertEqual(len(biz_msgs), 0)

    def test_business_context_property(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.BUSINESS)
        ctx = builder.build(session, "hi", business_context="data")
        self.assertEqual(ctx.business_context, "data")


# ---------------------------------------------------------------------------
# Workspace context
# ---------------------------------------------------------------------------


class TestContextBuilderWorkspaceContext(unittest.TestCase):

    def test_workspace_context_included(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.CODING)
        ctx = builder.build(session, "list files", workspace_path="/project/src")
        ws_msgs = [m for m in ctx.messages if "WORKSPACE CONTEXT" in m.get("content", "")]
        self.assertEqual(len(ws_msgs), 1)
        self.assertIn("/project/src", ws_msgs[0]["content"])

    def test_workspace_context_property(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.CODING)
        ctx = builder.build(session, "hi", workspace_path="/tmp")
        self.assertIn("/tmp", ctx.workspace_context)

    def test_no_workspace_path(self):
        builder = ContextBuilder()
        session = _make_session(mode=SessionMode.CODING)
        ctx = builder.build(session, "hi")
        self.assertEqual(ctx.workspace_context, "")


# ---------------------------------------------------------------------------
# Context limits
# ---------------------------------------------------------------------------


class TestContextBuilderLimits(unittest.TestCase):

    def test_respects_max_chars(self):
        builder = ContextBuilder(max_chars=500, max_history=100, max_memory_results=10)
        store = _memory_store_with_entries(20)
        builder_with_store = ContextBuilder(
            memory_store=store, max_chars=500, max_history=100, max_memory_results=10
        )
        session = _make_session()
        for i in range(30):
            session.add_message("user", f"Message {i}: " + "word " * 30)
        ctx = builder_with_store.build(session, "final")
        self.assertLessEqual(ctx.total_characters, 700)  # tolerance for structure
        store.close()

    def test_history_budget_respected(self):
        builder = ContextBuilder(max_chars=200, max_history=50)
        session = _make_session()
        for i in range(20):
            session.add_message("user", "x" * 50)
        ctx = builder.build(session, "final")
        # With budget=100 for history (50% of 200), should fit ~2 messages of 50 chars
        history = [m for m in ctx.messages if m["content"].startswith("x")]
        self.assertLessEqual(len(history), 4)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestContextBuilderOrdering(unittest.TestCase):

    def test_user_message_is_always_last(self):
        builder = ContextBuilder()
        session = _make_session()
        session.add_message("user", "prev")
        session.add_message("assistant", "ans")
        ctx = builder.build(session, "current")
        self.assertEqual(ctx.messages[-1]["content"], "current")
        self.assertEqual(ctx.messages[-1]["role"], "user")

    def test_memory_comes_before_history(self):
        store = _memory_store_with_entries(1)
        builder = ContextBuilder(memory_store=store, max_memory_results=1)
        session = _make_session()
        session.add_message("user", "old msg")
        ctx = builder.build(session, "sky")
        memory_idx = next(
            i for i, m in enumerate(ctx.messages)
            if "RELEVANT MEMORY" in m.get("content", "")
        )
        history_idx = next(
            i for i, m in enumerate(ctx.messages)
            if m.get("content") == "old msg"
        )
        self.assertLess(memory_idx, history_idx)
        store.close()

    def test_mode_before_history(self):
        builder = ContextBuilder()
        session = _make_session()
        session.add_message("user", "old msg")
        ctx = builder.build(session, "new msg")
        mode_idx = next(
            i for i, m in enumerate(ctx.messages)
            if "MODE INSTRUCTION" in m.get("content", "")
        )
        history_idx = next(
            i for i, m in enumerate(ctx.messages)
            if m.get("content") == "old msg"
        )
        self.assertLess(mode_idx, history_idx)


# ---------------------------------------------------------------------------
# Malformed / edge cases
# ---------------------------------------------------------------------------


class TestContextBuilderEdgeCases(unittest.TestCase):

    def test_empty_user_message(self):
        builder = ContextBuilder()
        session = _make_session()
        ctx = builder.build(session, "")
        self.assertEqual(ctx.messages[-1]["content"], "")

    def test_whitespace_user_message(self):
        builder = ContextBuilder()
        session = _make_session()
        ctx = builder.build(session, "   ")
        self.assertEqual(ctx.messages[-1]["content"], "   ")

    def test_long_user_message(self):
        builder = ContextBuilder()
        session = _make_session()
        long_msg = "x" * 10000
        ctx = builder.build(session, long_msg)
        self.assertEqual(ctx.messages[-1]["content"], long_msg)

    def test_session_with_metadata(self):
        builder = ContextBuilder()
        session = _make_session()
        session.metadata["workspace"] = "/tmp"
        ctx = builder.build(session, "hi")
        self.assertIsNotNone(ctx)

    def test_history_with_missing_content(self):
        builder = ContextBuilder()
        session = _make_session()
        session.history.append({"role": "user", "timestamp": "2026-01-01"})
        ctx = builder.build(session, "hi")
        # Should handle missing content gracefully
        self.assertIsNotNone(ctx)


if __name__ == "__main__":
    unittest.main()

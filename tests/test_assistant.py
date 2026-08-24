"""
Tests for Phase 7 Step 1-2: Config extensions and Session management.
"""

import unittest
from src.config.settings import VoxlineConfig, reset_config
from src.assistant.session import Session, SessionManager, SessionMode


# ---------------------------------------------------------------------------
# Config extension tests
# ---------------------------------------------------------------------------

class TestAssistantConfig(unittest.TestCase):

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_assistant_defaults_exist(self):
        config = VoxlineConfig()
        self.assertEqual(config.get("ASSISTANT_NAME"), "Voxline")
        self.assertEqual(config.get("ASSISTANT_DEFAULT_MODE"), "chat")
        self.assertEqual(config.get("ASSISTANT_MAX_HISTORY"), "20")

    def test_agent_defaults_exist(self):
        config = VoxlineConfig()
        self.assertEqual(config.get("AGENT_MAX_ITERATIONS"), "15")
        self.assertEqual(config.get("AGENT_STEP_TIMEOUT"), "60")
        self.assertEqual(config.get("AGENT_EXECUTION_POLICY"), "safe")

    def test_coding_defaults_exist(self):
        config = VoxlineConfig()
        self.assertEqual(config.get("CODING_WORKSPACE_ROOT"), ".")
        self.assertIn("python", config.get("CODING_ALLOWED_COMMANDS"))
        self.assertEqual(config.get("CODING_MAX_FILE_SIZE_MB"), "10")

    def test_assistant_properties(self):
        config = VoxlineConfig()
        self.assertEqual(config.assistant_name, "Voxline")
        self.assertEqual(config.assistant_default_mode, "chat")
        self.assertEqual(config.assistant_max_history, 20)

    def test_agent_properties(self):
        config = VoxlineConfig()
        self.assertEqual(config.agent_max_iterations, 15)
        self.assertEqual(config.agent_step_timeout, 60)
        self.assertEqual(config.agent_execution_policy, "safe")

    def test_coding_properties(self):
        config = VoxlineConfig()
        self.assertEqual(str(config.coding_workspace_root), ".")
        cmds = config.coding_allowed_commands
        self.assertIn("python", cmds)
        self.assertIn("pytest", cmds)
        self.assertEqual(config.coding_max_file_size_mb, 10)
        self.assertEqual(config.coding_max_output_bytes, 1048576)

    def test_api_properties(self):
        config = VoxlineConfig()
        self.assertEqual(config.api_host, "0.0.0.0")
        self.assertEqual(config.api_port, 8000)


# ---------------------------------------------------------------------------
# Error hierarchy tests
# ---------------------------------------------------------------------------

class TestNewErrors(unittest.TestCase):

    def test_session_errors(self):
        from src.errors import SessionError, SessionNotFoundError, SessionExpiredError
        self.assertTrue(issubclass(SessionError, Exception))
        self.assertTrue(issubclass(SessionNotFoundError, SessionError))
        self.assertTrue(issubclass(SessionExpiredError, SessionError))

    def test_workspace_errors(self):
        from src.errors import WorkspaceError, WorkspaceBoundaryError
        self.assertTrue(issubclass(WorkspaceError, Exception))
        self.assertTrue(issubclass(WorkspaceBoundaryError, WorkspaceError))

    def test_coding_agent_errors(self):
        from src.errors import CodingAgentError, AgentPlanError, CommandDeniedError
        self.assertTrue(issubclass(CodingAgentError, Exception))
        self.assertTrue(issubclass(AgentPlanError, CodingAgentError))
        self.assertTrue(issubclass(CommandDeniedError, Exception))


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSession(unittest.TestCase):

    def test_create_session(self):
        s = Session(
            session_id="test_001",
            mode=SessionMode.CHAT,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        self.assertEqual(s.session_id, "test_001")
        self.assertEqual(s.mode, SessionMode.CHAT)
        self.assertEqual(s.history, [])

    def test_add_message(self):
        s = Session(
            session_id="test_002",
            mode=SessionMode.CHAT,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        s.add_message("user", "hello")
        self.assertEqual(len(s.history), 1)
        self.assertEqual(s.history[0]["role"], "user")
        self.assertEqual(s.history[0]["content"], "hello")
        self.assertIn("timestamp", s.history[0])

    def test_add_multiple_messages(self):
        s = Session(
            session_id="test_003",
            mode=SessionMode.CHAT,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        s.add_message("user", "hello")
        s.add_message("assistant", "hi there")
        s.add_message("user", "how are you?")
        self.assertEqual(len(s.history), 3)

    def test_get_messages_limit(self):
        s = Session(
            session_id="test_004",
            mode=SessionMode.CHAT,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        s.add_message("user", "m1")
        s.add_message("assistant", "m2")
        s.add_message("user", "m3")
        msgs = s.get_messages(limit=2)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "m2")
        self.assertEqual(msgs[1]["content"], "m3")

    def test_clear(self):
        s = Session(
            session_id="test_005",
            mode=SessionMode.BUSINESS,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
        )
        s.add_message("user", "hello")
        s.clear()
        self.assertEqual(len(s.history), 0)

    def test_to_dict(self):
        s = Session(
            session_id="test_006",
            mode=SessionMode.CODING,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
            metadata={"workspace": "/tmp"},
        )
        d = s.to_dict()
        self.assertEqual(d["session_id"], "test_006")
        self.assertEqual(d["mode"], "coding")
        self.assertEqual(d["metadata"]["workspace"], "/tmp")
        self.assertEqual(d["history_length"], 0)


class TestSessionManager(unittest.TestCase):

    def test_create_session(self):
        mgr = SessionManager()
        s = mgr.create(SessionMode.CHAT)
        self.assertIsNotNone(s.session_id)
        self.assertEqual(s.mode, SessionMode.CHAT)

    def test_get_existing(self):
        mgr = SessionManager()
        s = mgr.create(SessionMode.CHAT)
        found = mgr.get(s.session_id)
        self.assertIs(found, s)

    def test_get_nonexistent(self):
        mgr = SessionManager()
        self.assertIsNone(mgr.get("nonexistent"))

    def test_get_or_create_new(self):
        mgr = SessionManager()
        s = mgr.get_or_create(None, SessionMode.BUSINESS)
        self.assertIsNotNone(s.session_id)
        self.assertEqual(s.mode, SessionMode.BUSINESS)

    def test_get_or_create_existing(self):
        mgr = SessionManager()
        s1 = mgr.create(SessionMode.CHAT, session_id="my_session")
        s2 = mgr.get_or_create("my_session", SessionMode.CHAT)
        self.assertIs(s1, s2)

    def test_delete(self):
        mgr = SessionManager()
        s = mgr.create(SessionMode.CHAT)
        self.assertTrue(mgr.delete(s.session_id))
        self.assertIsNone(mgr.get(s.session_id))

    def test_delete_nonexistent(self):
        mgr = SessionManager()
        self.assertFalse(mgr.delete("nonexistent"))

    def test_list_sessions(self):
        mgr = SessionManager()
        mgr.create(SessionMode.CHAT)
        mgr.create(SessionMode.BUSINESS)
        sessions = mgr.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_mode_isolation(self):
        mgr = SessionManager()
        chat_s = mgr.create(SessionMode.CHAT)
        biz_s = mgr.create(SessionMode.BUSINESS)
        chat_s.add_message("user", "hello chat")
        biz_s.add_message("user", "hello business")
        self.assertEqual(len(chat_s.history), 1)
        self.assertEqual(len(biz_s.history), 1)
        self.assertEqual(chat_s.history[0]["content"], "hello chat")
        self.assertEqual(biz_s.history[0]["content"], "hello business")

    def test_eviction(self):
        mgr = SessionManager(max_sessions=2)
        s1 = mgr.create(SessionMode.CHAT)
        s1.add_message("user", "old")
        import time
        time.sleep(0.01)
        s2 = mgr.create(SessionMode.CHAT)
        s2.add_message("user", "middle")
        time.sleep(0.01)
        s3 = mgr.create(SessionMode.CHAT)
        s3.add_message("user", "new")
        self.assertEqual(len(mgr.list_sessions()), 2)
        self.assertIsNone(mgr.get(s1.session_id))
        self.assertIsNotNone(mgr.get(s2.session_id))
        self.assertIsNotNone(mgr.get(s3.session_id))

    def test_custom_session_id(self):
        mgr = SessionManager()
        s = mgr.create(SessionMode.CODING, session_id="custom_id")
        self.assertEqual(s.session_id, "custom_id")

    def test_metadata(self):
        mgr = SessionManager()
        s = mgr.create(SessionMode.CODING, metadata={"workspace": "/project"})
        self.assertEqual(s.metadata["workspace"], "/project")


if __name__ == "__main__":
    unittest.main()

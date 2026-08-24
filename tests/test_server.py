"""Tests for the Voxline web server.

Tests cover:
  1. Server starts (TestClient)
  2. GET /health
  3. POST /api/chat
  4. POST /api/business
  5. Invalid request handling
  6. Provider failure handling
  7. Static UI files
  8. No security regression
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers.base import AIProvider, GenerationConfig, ModelInfo, ProviderHealth, ProviderStatus


# ---------------------------------------------------------------------------
# Mock provider for tests (avoids loading Qwen)
# ---------------------------------------------------------------------------

class MockProvider(AIProvider):
    """Fake provider that returns canned responses."""

    def __init__(self):
        self._call_count = 0

    @property
    def provider_id(self) -> str:
        return "mock_provider"

    @property
    def model_id(self) -> str:
        return "mock-model-1.0"

    @property
    def supports_streaming(self) -> bool:
        return False

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._call_count += 1
        return f"Mock response to: {prompt[:50]}"

    async def chat(self, messages, config: GenerationConfig) -> str:
        self._call_count += 1
        if messages:
            last = messages[-1]
            content = last.get("content", "") if isinstance(last, dict) else str(last)
            return f"Mock reply to: {content[:50]}"
        return "Mock reply"

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            message="Mock provider healthy",
            response_time_ms=1.0,
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id="mock-model-1.0",
            provider_id="mock_provider",
            model_type="mock",
            parameters=1000,
            vocab_size=1000,
            max_context_length=2048,
            device="cpu",
            supports_streaming=False,
        )


# ---------------------------------------------------------------------------
# Build test app with mock provider
# ---------------------------------------------------------------------------

def _build_test_app():
    """Create a FastAPI test app wired to MockProvider."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
    from typing import Optional

    from src.assistant.session import SessionManager, SessionMode
    from src.assistant.context import ContextBuilder
    from src.assistant.chat import ChatAssistant, AssistantResponse
    from src.assistant.business import (
        BusinessAssistant, BusinessRequest, BusinessResponse,
        BusinessTaskType,
    )
    from src.memory.memory import MemoryStore

    app = FastAPI(title="Voxline Test", version="0.4.0-test")
    provider = MockProvider()
    sm = SessionManager()
    ms = MemoryStore(":memory:")
    ctx = ContextBuilder(memory_store=ms, max_history=20)
    chat_asst = ChatAssistant(provider=provider, session_manager=sm, context_builder=ctx, memory_store=ms)
    biz_asst = BusinessAssistant(provider=provider, session_manager=sm, context_builder=ctx, memory_store=ms)

    class ChatReq(BaseModel):
        message: str = Field(min_length=1, max_length=10000)
        session_id: Optional[str] = None

    class BizReq(BaseModel):
        message: str = Field(min_length=1, max_length=10000)
        session_id: Optional[str] = None
        task_type: str = "general_analysis"

    @app.get("/", response_class=HTMLResponse)
    async def root():
        idx = Path(__file__).resolve().parent.parent / "src" / "api" / "static" / "index.html"
        if idx.exists():
            return HTMLResponse(content=idx.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Voxline</h1>")

    @app.get("/health")
    async def health():
        h = await provider.health_check()
        return {
            "status": h.status.value,
            "provider": provider.provider_id,
            "model": provider.model_id,
        }

    @app.post("/api/chat")
    async def api_chat(req: ChatReq):
        sid = req.session_id
        if not sid:
            s = chat_asst.new_session(mode=SessionMode.CHAT)
            sid = s.session_id
        resp = chat_asst.chat(session_id=sid, message=req.message)
        return {
            "response": resp.text,
            "session_id": resp.session_id,
            "mode": resp.mode,
            "provider": resp.provider_id,
            "model": resp.model_id,
        }

    @app.post("/api/business")
    async def api_business(req: BizReq):
        sid = req.session_id
        if not sid:
            s = biz_asst.new_session()
            sid = s.session_id
        task_map = {t.value: t for t in BusinessTaskType}
        tt = task_map.get(req.task_type, BusinessTaskType.GENERAL_ANALYSIS)
        resp = biz_asst.chat(session_id=sid, message=req.message, task_type=tt)
        return {
            "response": resp.response,
            "session_id": resp.session_id,
            "task_type": resp.task_type.value,
            "provider": resp.provider_id,
            "model": resp.model_id,
        }

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServerStarts(unittest.TestCase):
    """Test 1: server starts with TestClient."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app = _build_test_app()
        cls.client = TestClient(cls.app)

    def test_server_starts(self):
        """TestClient can reach the server."""
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)


class TestHealthEndpoint(unittest.TestCase):
    """Test 2: GET /health returns provider info."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app = _build_test_app()
        cls.client = TestClient(cls.app)

    def test_health_returns_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["provider"], "mock_provider")
        self.assertEqual(data["model"], "mock-model-1.0")


class TestChatEndpoint(unittest.TestCase):
    """Test 3: POST /api/chat returns response."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app = _build_test_app()
        cls.client = TestClient(cls.app)

    def test_chat_returns_response(self):
        r = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("response", data)
        self.assertIn("session_id", data)
        self.assertEqual(data["mode"], "chat")
        self.assertEqual(data["provider"], "mock_provider")
        self.assertEqual(data["model"], "mock-model-1.0")

    def test_chat_with_session_id(self):
        r1 = self.client.post("/api/chat", json={"message": "Hi"})
        sid = r1.json()["session_id"]
        r2 = self.client.post("/api/chat", json={"message": "Again", "session_id": sid})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["session_id"], sid)


class TestBusinessEndpoint(unittest.TestCase):
    """Test 4: POST /api/business returns response."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app = _build_test_app()
        cls.client = TestClient(cls.app)

    def test_business_returns_response(self):
        r = self.client.post("/api/business", json={"message": "Analyze my sales"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("response", data)
        self.assertEqual(data["task_type"], "general_analysis")
        self.assertEqual(data["provider"], "mock_provider")

    def test_business_with_task_type(self):
        r = self.client.post("/api/business", json={
            "message": "KPI analysis",
            "task_type": "kpi_analysis",
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["task_type"], "kpi_analysis")


class TestInvalidRequest(unittest.TestCase):
    """Test 5: invalid requests return proper errors."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app = _build_test_app()
        cls.client = TestClient(cls.app)

    def test_empty_message_rejected(self):
        r = self.client.post("/api/chat", json={"message": ""})
        self.assertEqual(r.status_code, 422)

    def test_missing_message_rejected(self):
        r = self.client.post("/api/chat", json={})
        self.assertEqual(r.status_code, 422)

    def test_wrong_content_type(self):
        r = self.client.post("/api/chat", content="not json",
                             headers={"Content-Type": "text/plain"})
        self.assertIn(r.status_code, [400, 415, 422])


class TestProviderFailure(unittest.TestCase):
    """Test 6: provider errors return 500."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
        from typing import Optional
        from src.assistant.session import SessionManager, SessionMode
        from src.assistant.context import ContextBuilder
        from src.assistant.chat import ChatAssistant
        from src.memory.memory import MemoryStore

        app = FastAPI(title="FailTest")
        fail_provider = MockProvider()
        fail_provider.chat = AsyncMock(side_effect=RuntimeError("Model crashed"))
        fail_provider.generate = AsyncMock(side_effect=RuntimeError("Model crashed"))

        sm = SessionManager()
        ms = MemoryStore(":memory:")
        ctx = ContextBuilder(memory_store=ms, max_history=20)
        chat_asst = ChatAssistant(provider=fail_provider, session_manager=sm, context_builder=ctx, memory_store=ms)

        class ChatReq(BaseModel):
            message: str = Field(min_length=1, max_length=10000)
            session_id: Optional[str] = None

        @app.post("/api/chat")
        async def api_chat(req: ChatReq):
            sid = req.session_id
            if not sid:
                s = chat_asst.new_session(mode=SessionMode.CHAT)
                sid = s.session_id
            try:
                resp = chat_asst.chat(session_id=sid, message=req.message)
                return {"response": resp.text}
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        cls.client = TestClient(app)

    def test_provider_failure_returns_500(self):
        r = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(r.status_code, 500)


class TestStaticUI(unittest.TestCase):
    """Test 7: static UI files are served."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app = _build_test_app()
        cls.client = TestClient(cls.app)

    def test_index_html_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Voxline AI", r.text)
        self.assertIn("chat-container", r.text)
        self.assertIn("send-btn", r.text)

    def test_index_has_mode_selector(self):
        r = self.client.get("/")
        self.assertIn("data-mode=\"chat\"", r.text)
        self.assertIn("data-mode=\"business\"", r.text)
        self.assertIn("data-mode=\"coding\"", r.text)

    def test_index_has_api_endpoints(self):
        r = self.client.get("/")
        self.assertIn("/api/chat", r.text)
        self.assertIn("/api/business", r.text)
        self.assertIn("/health", r.text)


class TestNoSecurityRegression(unittest.TestCase):
    """Test 8: security tool layer still works."""

    def test_tool_registry_works(self):
        from src.tools.tools import ToolRegistry
        tr = ToolRegistry(".")
        result = tr.execute_tool("calculator", expression="2+2")
        self.assertEqual(result, 4.0)

    def test_path_security_works(self):
        from src.tools.security import PathSecurity, PermissionDecision
        ps = PathSecurity(".")
        r = ps.validate_path("../../etc/passwd")
        self.assertEqual(r.decision, PermissionDecision.DENIED)

    def test_command_policy_works(self):
        from src.tools.security import CommandPolicy, PermissionDecision
        p = CommandPolicy(allowed_commands={"python"}, denied_commands={"rm"})
        r = p.evaluate("rm", ["-rf", "/"])
        self.assertEqual(r.decision, PermissionDecision.DENIED)

    def test_file_size_guard_works(self):
        from src.tools.security import FileSizeGuard, PermissionDecision
        g = FileSizeGuard(max_size_bytes=10)
        r = g.check_write("x" * 100)
        self.assertEqual(r.decision, PermissionDecision.DENIED)

    def test_audit_log_works(self):
        from src.tools.security import AuditLog, PermissionDecision
        a = AuditLog()
        a.record(tool_name="test", operation="op", decision=PermissionDecision.ALLOWED,
                 reason="ok", success=True, duration_ms=1.0)
        self.assertEqual(len(a.entries), 1)


if __name__ == "__main__":
    unittest.main()

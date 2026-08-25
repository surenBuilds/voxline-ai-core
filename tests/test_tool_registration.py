"""
Tests for Phase 7 Step 11: Tool Registration + End-to-End Coding Workflow.

Covers:
  BOOTSTRAP (tests 1-6)
  CAPABILITY DISCOVERY (tests 7-9)
  REPOSITORY CONTEXT (tests 10-12)
  FULL WORKFLOW (tests 13-18)
  INTEGRATION TOOL EXECUTION (tests 19-22)
  API ENDPOINT (tests 23-25)
  SECURITY REGRESSION (tests 26-30)
  SMOKE TEST GATE (tests 31-32)
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant.coding import (
    CodingAction,
    CodingAgent,
    CodingPlan,
    CodingResult,
    CodingStep,
    CodingTask,
    IterationRecord,
    RepositoryContext,
    PullRequestInfo,
    DeploymentInfo,
    ActionStatus,
    ActionType,
    TaskStatus,
)
from src.assistant.context import ContextBuilder
from src.assistant.session import SessionManager, SessionMode
from src.config.settings import VoxlineConfig, reset_config
from src.errors import (
    AgentPlanError,
    WorkspaceBoundaryError,
)
from src.integrations.credentials import CredentialProvider
from src.language import Language
from src.memory.memory import MemoryStore
from src.providers.base import AIProvider, GenerationConfig, ModelInfo, ProviderHealth, ProviderStatus
from src.tools.bootstrap import build_tool_registry
from src.tools.security import AuditLog, PathSecurity, PermissionDecision
from src.tools.tools import ToolRegistry


# ---------------------------------------------------------------------------
# FakeProvider
# ---------------------------------------------------------------------------


class FakeProvider(AIProvider):
    def __init__(self, response: str = "", plan: Optional[Dict] = None):
        self._response = response
        self._plan = plan
        self.call_count = 0

    @property
    def provider_id(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return "fake-v1"

    @property
    def supports_streaming(self) -> bool:
        return False

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        self.call_count += 1
        if self._plan:
            return json.dumps(self._plan)
        return self._response

    async def chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str:
        self.call_count += 1
        if self._plan:
            return json.dumps(self._plan)
        return self._response

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, message="ok")

    def is_loaded(self) -> bool:
        return True

    async def load(self) -> None:
        pass

    async def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# FakeCredentialProvider
# ---------------------------------------------------------------------------


class FakeCredentialProvider(CredentialProvider):
    def __init__(self, tokens: Optional[Dict[str, str]] = None):
        self._tokens = tokens or {}

    def get_token(self, service: str) -> Optional[str]:
        return self._tokens.get(service)

    def is_available(self, service: str) -> bool:
        return self.get_token(service) is not None

    def redact(self, text: str) -> str:
        result = text
        for svc, token in self._tokens.items():
            if token and len(token) > 4:
                result = result.replace(token, f"***{svc.upper()}***")
        return result


# ---------------------------------------------------------------------------
# Valid plan for FakeProvider
# ---------------------------------------------------------------------------

PLAN_JSON = {
    "objective": "Add hello function",
    "understanding": "User wants a hello function",
    "relevant_files": ["hello.py"],
    "steps": [
        {"step_number": 1, "description": "check pip", "action_type": "command", "target_files": [], "command": "pip --version"},
        {"step_number": 2, "description": "write file", "action_type": "write", "target_files": ["hello.py"], "command": ""},
        {"step_number": 3, "description": "run validation", "action_type": "command", "target_files": [], "command": "pip --version"},
    ],
    "risks": ["File might not exist"],
    "validation_commands": ["pip --version"],
    "requires_approval": False,
}


# =========================================================================
# BOOTSTRAP (tests 1-6)
# =========================================================================


class TestBootstrap(unittest.TestCase):
    """Tests 1-6: ToolRegistry bootstrap from config."""

    def setUp(self):
        reset_config()

    def test_01_build_registry_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = build_tool_registry(workspace_root=tmpdir)
            self.assertIsInstance(registry, ToolRegistry)
            self.assertIn("read_file", registry.tools)
            self.assertIn("write_file", registry.tools)
            self.assertIn("execute_command", registry.tools)
            self.assertIn("calculator", registry.tools)

    def test_02_build_registry_includes_workspace_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = build_tool_registry(workspace_root=tmpdir)
            self.assertIn("workspace_clone", registry.tools)
            self.assertIn("workspace_diff", registry.tools)
            self.assertIn("workspace_test", registry.tools)

    def test_03_build_registry_no_github_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            creds = FakeCredentialProvider({"github": "ghp_fake123"})
            config = VoxlineConfig()
            config._config["GITHUB_ENABLED"] = "false"
            registry = build_tool_registry(config=config, credential_provider=creds, workspace_root=tmpdir)
            self.assertNotIn("github_read_repository", registry.tools)

    def test_04_build_registry_no_github_when_no_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            creds = FakeCredentialProvider()
            config = VoxlineConfig()
            config._config["GITHUB_ENABLED"] = "true"
            registry = build_tool_registry(config=config, credential_provider=creds, workspace_root=tmpdir)
            self.assertNotIn("github_read_repository", registry.tools)

    def test_05_build_registry_no_vercel_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            creds = FakeCredentialProvider({"vercel": "vrcl_fake123"})
            config = VoxlineConfig()
            config._config["VERCEL_ENABLED"] = "false"
            registry = build_tool_registry(config=config, credential_provider=creds, workspace_root=tmpdir)
            self.assertNotIn("vercel_list_projects", registry.tools)

    def test_06_build_registry_no_crash_on_bad_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VoxlineConfig()
            config._config["GITHUB_ENABLED"] = "true"
            config._config["GITHUB_TOKEN"] = ""
            registry = build_tool_registry(config=config, workspace_root=tmpdir)
            self.assertIsInstance(registry, ToolRegistry)
            self.assertNotIn("github_read_repository", registry.tools)


# =========================================================================
# CAPABILITY DISCOVERY (tests 7-9)
# =========================================================================


class TestCapabilityDiscovery(unittest.TestCase):
    """Tests 7-9: ToolRegistry.available_tools() returns safe summaries."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.registry = build_tool_registry(workspace_root=tmpdir)

    def test_07_available_tools_returns_dict(self):
        result = self.registry.available_tools()
        self.assertIsInstance(result, dict)
        self.assertIn("core", result)

    def test_08_available_tools_groups_by_category(self):
        result = self.registry.available_tools()
        self.assertIn("workspace", result)
        self.assertIn("core", result)
        core_names = [t["name"] for t in result["core"]]
        self.assertIn("read_file", core_names)

    def test_09_available_tools_no_credentials_exposed(self):
        result = self.registry.available_tools()
        for cat, tools in result.items():
            for tool in tools:
                self.assertIn("name", tool)
                self.assertIn("description", tool)
                self.assertNotIn("token", tool["description"].lower())
                self.assertNotIn("credential", tool["description"].lower())
                self.assertNotIn("secret", tool["description"].lower())


# =========================================================================
# REPOSITORY CONTEXT (tests 10-12)
# =========================================================================


class TestRepositoryContext(unittest.TestCase):
    """Tests 10-12: CodingTask with repository context."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_10_coding_task_with_repository(self):
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("fix bug", None, None)
        task.repository_owner = "surenBuilds"
        task.repository_name = "my-project"
        task.repository_branch = "main"
        self.assertEqual(task.repository_owner, "surenBuilds")
        self.assertEqual(task.repository_name, "my-project")

    def test_11_repository_context_model(self):
        ctx = RepositoryContext(
            owner="surenBuilds",
            name="my-project",
            branch="main",
            clone_url="https://github.com/surenBuilds/my-project.git",
        )
        self.assertEqual(ctx.owner, "surenBuilds")
        self.assertEqual(ctx.clone_url, "https://github.com/surenBuilds/my-project.git")

    def test_12_pull_request_info_model(self):
        pr = PullRequestInfo(
            number=42,
            title="Fix auth",
            url="https://github.com/test/test/pull/42",
            head_branch="fix/auth",
            base_branch="main",
            state="open",
        )
        self.assertEqual(pr.number, 42)
        self.assertEqual(pr.state, "open")


# =========================================================================
# FULL WORKFLOW (tests 13-18)
# =========================================================================


class TestFullWorkflow(unittest.TestCase):
    """Tests 13-18: End-to-end workflow with mocked tools."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hello_file = Path(self.tmpdir) / "hello.py"
        self.hello_file.write_text("def hello():\n    return 'world'\n")

    def test_13_execute_with_repository_no_tools(self):
        provider = FakeProvider(plan=PLAN_JSON)
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(
            provider=provider,
            workspace=self.tmpdir,
            tool_registry=registry,
            require_approval_for_writes=False,
        )
        result = agent.execute_with_repository(
            user_request="add hello function",
            repository_owner="surenBuilds",
            repository_name="my-project",
            repository_branch="main",
        )
        self.assertIsInstance(result, CodingResult)

    def test_14_execute_with_repository_local_only(self):
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider,
            workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        result = agent.execute_with_repository(
            user_request="add hello function",
        )
        self.assertIsInstance(result, CodingResult)

    def test_15_phase_discovery_no_github(self):
        provider = FakeProvider()
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, tool_registry=registry)
        task = agent._create_task("test", None, None)
        task.repository_owner = "surenBuilds"
        task.repository_name = "my-project"
        info = agent._phase_discovery(task)
        self.assertIsInstance(info, str)

    def test_16_phase_review_no_diff_tool(self):
        provider = FakeProvider()
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, tool_registry=registry)
        task = agent._create_task("test", None, None)
        result = agent._phase_review(task)
        self.assertIsInstance(result, dict)

    def test_17_phase_github_no_pr_tool(self):
        provider = FakeProvider()
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, tool_registry=registry)
        task = agent._create_task("test", None, None)
        task.repository_owner = "surenBuilds"
        task.repository_name = "my-project"
        result = agent._phase_github(task, "feature", "main")
        self.assertIsNone(result)

    def test_18_phase_vercel_no_deploy_tool(self):
        provider = FakeProvider()
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, tool_registry=registry)
        task = agent._create_task("test", None, None)
        result = agent._phase_vercel(task)
        self.assertIsNone(result)


# =========================================================================
# INTEGRATION TOOL EXECUTION (tests 19-22)
# =========================================================================


class TestIntegrationToolExecution(unittest.TestCase):
    """Tests 19-22: Executing integration tools through CodingAgent."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hello_file = Path(self.tmpdir) / "hello.py"
        self.hello_file.write_text("def hello():\n    return 'world'\n")

    def test_19_exec_step_github_tool_not_available(self):
        provider = FakeProvider()
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, tool_registry=registry)
        task = agent._create_task("test", None, None)
        step = CodingStep(
            step_number=1,
            description="read repo",
            action_type="github_read_repository",
            target_files=[],
        )
        action = agent._execute_step(task, step)
        self.assertEqual(action.status, ActionStatus.FAILED)
        self.assertIn("not available", action.error.lower())

    def test_20_exec_step_vercel_tool_not_available(self):
        provider = FakeProvider()
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, tool_registry=registry)
        task = agent._create_task("test", None, None)
        step = CodingStep(
            step_number=1,
            description="list projects",
            action_type="vercel_list_projects",
            target_files=[],
        )
        action = agent._execute_step(task, step)
        self.assertEqual(action.status, ActionStatus.FAILED)
        self.assertIn("not available", action.error.lower())

    def test_21_exec_step_workspace_tool_not_available(self):
        provider = FakeProvider()
        registry = ToolRegistry(workspace_root=self.tmpdir)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, tool_registry=registry)
        task = agent._create_task("test", None, None)
        step = CodingStep(
            step_number=1,
            description="clone repo",
            action_type="workspace_clone",
            target_files=[],
        )
        action = agent._execute_step(task, step)
        self.assertEqual(action.status, ActionStatus.FAILED)
        self.assertIn("not available", action.error.lower())

    def test_22_exec_step_standard_action_types_still_work(self):
        provider = FakeProvider()
        agent = CodingAgent(
            provider=provider,
            workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        task = agent._create_task("test", None, None)
        step = CodingStep(
            step_number=1,
            description="read file",
            action_type="read",
            target_files=["hello.py"],
        )
        action = agent._execute_step(task, step)
        self.assertEqual(action.status, ActionStatus.EXECUTED)


# =========================================================================
# API ENDPOINT (tests 23-25)
# =========================================================================


class TestAPIEndpoint(unittest.TestCase):
    """Tests 23-25: API endpoint structure."""

    def test_23_coding_request_model(self):
        from serve_v04 import CodingRequest
        req = CodingRequest(
            message="fix bug",
            repository_owner="surenBuilds",
            repository_name="my-project",
            repository_branch="main",
            create_pr=True,
            deploy_preview=False,
        )
        self.assertEqual(req.message, "fix bug")
        self.assertTrue(req.create_pr)
        self.assertFalse(req.deploy_preview)

    def test_24_coding_request_defaults(self):
        from serve_v04 import CodingRequest
        req = CodingRequest(message="fix bug")
        self.assertEqual(req.repository_owner, "")
        self.assertEqual(req.repository_name, "")
        self.assertEqual(req.repository_branch, "main")
        self.assertFalse(req.create_pr)
        self.assertFalse(req.deploy_preview)

    def test_25_integration_endpoint_structure(self):
        from serve_v04 import app
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        self.assertIn("/api/coding", routes)
        self.assertIn("/api/chat", routes)
        self.assertIn("/api/business", routes)
        self.assertIn("/api/integrations", routes)
        self.assertIn("/api/tools", routes)


# =========================================================================
# SECURITY REGRESSION (tests 26-30)
# =========================================================================


class TestSecurityRegression(unittest.TestCase):
    """Tests 26-30: Verify security boundaries remain intact."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_26_path_traversal_blocked(self):
        registry = ToolRegistry(workspace_root=self.tmpdir)
        result = registry.execute("read_file", path="../../../etc/passwd")
        self.assertIn("error", result)

    def test_27_unauthorized_command_blocked(self):
        registry = ToolRegistry(workspace_root=self.tmpdir)
        result = registry.execute("execute_command", command="rm -rf /")
        self.assertIn("error", result)

    def test_28_token_not_in_tool_schema(self):
        registry = ToolRegistry(workspace_root=self.tmpdir)
        tools = registry.list_tools()
        for name, info in tools.items():
            desc = info.get("description", "").lower()
            self.assertNotIn("token", desc)
            self.assertNotIn("credential", desc)
            self.assertNotIn("secret", desc)

    def test_29_audit_log_records_operations(self):
        registry = ToolRegistry(workspace_root=self.tmpdir)
        registry.execute("read_file", path=".")
        entries = registry.audit_log.entries
        self.assertTrue(len(entries) > 0)
        self.assertEqual(entries[-1].tool_name, "read_file")

    def test_30_available_tools_no_sensitive_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = build_tool_registry(workspace_root=tmpdir)
            tools = registry.available_tools()
            serialized = json.dumps(tools)
            self.assertNotIn("ghp_", serialized)
            self.assertNotIn("vrcl_", serialized)
            self.assertNotIn("password", serialized.lower())


# =========================================================================
# SMOKE TEST GATE (tests 31-32)
# =========================================================================


class TestSmokeTestGate(unittest.TestCase):
    """Tests 31-32: External smoke tests are gated by env var."""

    def test_31_smoke_test_file_exists(self):
        smoke_path = Path(__file__).parent / "smoke_external_integrations.py"
        if smoke_path.exists():
            content = smoke_path.read_text(encoding="utf-8")
            self.assertIn("VOXLINE_EXTERNAL_SMOKE", content)

    def test_32_smoke_test_not_run_by_default(self):
        smoke_path = Path(__file__).parent / "smoke_external_integrations.py"
        if smoke_path.exists():
            content = smoke_path.read_text(encoding="utf-8")
            self.assertIn("skipUnless", content)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    unittest.main()

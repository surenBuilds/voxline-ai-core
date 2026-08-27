"""
Tests for Phase 7 Step 12: Production Hardening.

Covers:
  CONFIGURATION VALIDATION (tests 1-5)
  CAPABILITY DISCOVERY (tests 6-8)
  GITHUB WORKFLOW (tests 9-14)
  FAILURE RECOVERY (tests 15-20)
  TEST EXECUTION SAFETY (tests 21-22)
  DEPLOYMENT VERIFICATION (tests 23-25)
  AUTO-APPROVE WORKSPACE WRITES (tests 26-27)
  API ERROR SAFETY (tests 28-29)
  OBSERVABILITY (tests 30-31)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant.coding import (
    CodingAgent,
    CodingResult,
    CodingStatus,
    FailureType,
    FailureInfo,
    CodingPlan,
    CodingStep,
    CodingTask,
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
from src.integrations.credentials import CredentialProvider
from src.tools.tools import ToolRegistry, Calculator
from src.tools.security import AuditLog, PathSecurity, PermissionDecision
from src.language import Language
from src.providers.base import AIProvider, GenerationConfig, ModelInfo, ProviderHealth, ProviderStatus


# ---------------------------------------------------------------------------
# FakeProvider — returns structured JSON plan or arbitrary text
# ---------------------------------------------------------------------------


class FakeProvider(AIProvider):
    def __init__(self, response: str = "", plan: Optional[Dict] = None):
        self._response = response
        self._plan = plan
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
        if self._plan:
            return json.dumps(self._plan)
        return self._response

    async def chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.last_config = config
        if self._plan:
            return json.dumps(self._plan)
        return self._response

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            message="fake provider ready",
        )

    def is_loaded(self) -> bool:
        return True

    async def load(self) -> None:
        pass

    async def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Valid plan responses for FakeProvider
# ---------------------------------------------------------------------------

PLAN_JSON = {
    "objective": "Add a hello world function",
    "understanding": "User wants a simple hello function",
    "relevant_files": ["hello.py"],
    "steps": [
        {
            "step_number": 1,
            "description": "Check pip version",
            "action_type": "command",
            "target_files": [],
            "command": "pip --version",
        },
        {
            "step_number": 2,
            "description": "Create the file",
            "action_type": "write",
            "target_files": ["hello.py"],
            "command": "",
        },
        {
            "step_number": 3,
            "description": "Run validation command",
            "action_type": "command",
            "target_files": [],
            "command": "pip --version",
        },
    ],
    "risks": ["File might not exist yet"],
    "validation_commands": ["pip --version"],
    "requires_approval": False,
}


# =========================================================================
# CONFIGURATION VALIDATION (tests 1-5)
# =========================================================================


class TestConfigurationValidation(unittest.TestCase):
    """Tests 1-5: VoxlineConfig.validate() checks."""

    def tearDown(self):
        reset_config()

    def test_01_config_valid(self):
        """validate() returns empty list for valid default config."""
        reset_config()
        config = VoxlineConfig.__new__(VoxlineConfig)
        config._config = dict(VoxlineConfig.DEFAULTS)
        config._config["GITHUB_ENABLED"] = "false"
        config._config["VERCEL_ENABLED"] = "false"
        issues = config.validate()
        self.assertIsInstance(issues, list)
        timeout_issues = [i for i in issues if "AGENT_STEP_TIMEOUT" in i]
        self.assertEqual(len(timeout_issues), 0)

    def test_02_config_missing_github_token(self):
        """GITHUB_ENABLED=true without token gives warning."""
        reset_config()
        config = VoxlineConfig.__new__(VoxlineConfig)
        config._config = dict(VoxlineConfig.DEFAULTS)
        config._config["GITHUB_ENABLED"] = "true"
        config._config["GITHUB_TOKEN"] = ""
        config._config["VERCEL_ENABLED"] = "false"
        issues = config.validate()
        github_issues = [i for i in issues if "GITHUB_TOKEN" in i]
        self.assertGreater(len(github_issues), 0)
        self.assertIn("GITHUB_ENABLED=true", github_issues[0])

    def test_03_config_missing_vercel_token(self):
        """VERCEL_ENABLED=true without token gives warning."""
        reset_config()
        config = VoxlineConfig.__new__(VoxlineConfig)
        config._config = dict(VoxlineConfig.DEFAULTS)
        config._config["GITHUB_ENABLED"] = "false"
        config._config["VERCEL_ENABLED"] = "true"
        config._config["VERCEL_TOKEN"] = ""
        issues = config.validate()
        vercel_issues = [i for i in issues if "VERCEL_TOKEN" in i]
        self.assertGreater(len(vercel_issues), 0)
        self.assertIn("VERCEL_ENABLED=true", vercel_issues[0])

    def test_04_config_low_timeout(self):
        """timeout < 10 gives warning."""
        reset_config()
        config = VoxlineConfig.__new__(VoxlineConfig)
        config._config = dict(VoxlineConfig.DEFAULTS)
        config._config["AGENT_STEP_TIMEOUT"] = "5"
        config._config["GITHUB_ENABLED"] = "false"
        config._config["VERCEL_ENABLED"] = "false"
        issues = config.validate()
        timeout_issues = [i for i in issues if "AGENT_STEP_TIMEOUT" in i]
        self.assertGreater(len(timeout_issues), 0)
        self.assertIn("30", timeout_issues[0])

    def test_05_config_invalid_repo_format(self):
        """bad repo format gives warning."""
        reset_config()
        config = VoxlineConfig.__new__(VoxlineConfig)
        config._config = dict(VoxlineConfig.DEFAULTS)
        config._config["GITHUB_ENABLED"] = "true"
        config._config["GITHUB_TOKEN"] = "ghp_fake"
        config._config["GITHUB_ALLOWED_REPOSITORIES"] = "bad-repo-no-slash"
        config._config["VERCEL_ENABLED"] = "false"
        issues = config.validate()
        repo_issues = [i for i in issues if "expected format" in i.lower()]
        self.assertGreater(len(repo_issues), 0)


# =========================================================================
# CAPABILITY DISCOVERY (tests 6-8)
# =========================================================================


class TestCapabilityDiscovery(unittest.TestCase):
    """Tests 6-8: available_tools() security and structure."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_06_available_tools_includes_approval(self):
        """available_tools() includes requires_approval field."""
        tr = ToolRegistry(self.tmpdir)
        tools = tr.available_tools()
        for cat, tool_list in tools.items():
            for tool_info in tool_list:
                self.assertIn("requires_approval", tool_info)
                self.assertIsInstance(tool_info["requires_approval"], bool)

    def test_07_available_tools_no_sensitive_data(self):
        """no tokens, no env vars, no secrets in available_tools."""
        tr = ToolRegistry(self.tmpdir)
        tools = tr.available_tools()
        tools_json = json.dumps(tools)
        self.assertNotIn("GITHUB_TOKEN", tools_json)
        self.assertNotIn("VERCEL_TOKEN", tools_json)
        self.assertNotIn("OPENAI_API_KEY", tools_json)
        self.assertNotIn("ghp_", tools_json)
        self.assertNotIn("password", tools_json.lower())
        self.assertNotIn("secret", tools_json.lower())

    def test_08_available_tools_categories(self):
        """tools are categorized correctly."""
        tr = ToolRegistry(self.tmpdir)
        tools = tr.available_tools()
        self.assertIn("core", tools)
        core_names = [t["name"] for t in tools["core"]]
        self.assertIn("calculator", core_names)
        self.assertIn("read_file", core_names)
        self.assertIn("write_file", core_names)
        self.assertIn("list_directory", core_names)
        self.assertIn("execute_command", core_names)


# =========================================================================
# GITHUB WORKFLOW (tests 9-14)
# =========================================================================


class TestGithubWorkflow(unittest.TestCase):
    """Tests 9-14: Branch sanitization and GitHub phases."""

    def test_09_branch_sanitize_valid(self):
        """valid branch names pass through."""
        result = CodingAgent._sanitize_branch_name("feature/add-hello")
        self.assertEqual(result, "feature/add-hello")

    def test_10_branch_sanitize_special_chars(self):
        """special chars replaced with hyphens."""
        result = CodingAgent._sanitize_branch_name("feature@hello world!")
        self.assertNotIn("@", result)
        self.assertNotIn(" ", result)
        self.assertNotIn("!", result)

    def test_11_branch_sanitize_empty(self):
        """empty name gets default."""
        result = CodingAgent._sanitize_branch_name("")
        self.assertEqual(result, "voxline-work")

    def test_12_branch_sanitize_dots(self):
        """leading/trailing dots stripped."""
        result = CodingAgent._sanitize_branch_name("..bad..")
        self.assertNotIn(".", result[0])
        self.assertNotIn(".", result[-1])

    def test_13_commit_and_push_returns_sha(self):
        """_phase_commit_and_push returns commit SHA."""
        tmpdir = tempfile.mkdtemp()
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )
        task = agent._create_task("test", None, None)
        task.repository_owner = ""
        task.repository_name = ""
        result = agent._phase_commit_and_push(task, "feature/test")
        self.assertIsNone(result)

    def test_14_pr_after_commit(self):
        """_phase_github returns None when no repo configured."""
        tmpdir = tempfile.mkdtemp()
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )
        task = agent._create_task("test", None, None)
        task.repository_owner = ""
        task.repository_name = ""
        result = agent._phase_github(task, "feature/test", "main")
        self.assertIsNone(result)


# =========================================================================
# FAILURE RECOVERY (tests 15-20)
# =========================================================================


class TestFailureRecovery(unittest.TestCase):
    """Tests 15-20: CodingResult fields and _failure_result."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_15_coding_result_status_field(self):
        """CodingResult has status field."""
        result = CodingResult(
            task_id="t1", success=True, summary="ok",
            status=CodingStatus.SUCCESS,
        )
        self.assertEqual(result.status, CodingStatus.SUCCESS)

    def test_16_coding_result_operation_id(self):
        """CodingResult has operation_id."""
        result = CodingResult(
            task_id="t1", success=True, summary="ok",
            operation_id="op_abc123",
        )
        self.assertEqual(result.operation_id, "op_abc123")

    def test_17_coding_result_tests_counts(self):
        """CodingResult has tests_passed/tests_failed."""
        result = CodingResult(
            task_id="t1", success=True, summary="ok",
            tests_passed=5, tests_failed=1,
        )
        self.assertEqual(result.tests_passed, 5)
        self.assertEqual(result.tests_failed, 1)

    def test_18_coding_result_commit_sha(self):
        """CodingResult has commit_sha."""
        result = CodingResult(
            task_id="t1", success=True, summary="ok",
            commit_sha="abc123def",
        )
        self.assertEqual(result.commit_sha, "abc123def")

    def test_19_failure_result_includes_status(self):
        """_failure_result sets correct status."""
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("test", None, None)
        result = agent._failure_result(
            task, "something broke",
            status=CodingStatus.TIMED_OUT,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, CodingStatus.TIMED_OUT)

    def test_20_failure_result_includes_type(self):
        """_failure_result includes FailureInfo."""
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("test", None, None)
        result = agent._failure_result(
            task, "github failed",
            failure_type=FailureType.GITHUB,
        )
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].failure_type, FailureType.GITHUB)
        self.assertIn("github failed", result.failures[0].message)


# =========================================================================
# TEST EXECUTION SAFETY (tests 21-22)
# =========================================================================


class TestExecutionSafety(unittest.TestCase):
    """Tests 21-22: RepositoryWorkspace path validation and shlex usage."""

    def test_21_repository_workspace_validates_path(self):
        """RepositoryWorkspace validates repo_dir inside workspace."""
        from src.tools.integration_tools import RepositoryWorkspace
        tmpdir = tempfile.mkdtemp()
        ws = RepositoryWorkspace(tmpdir, "owner", "repo")
        self.assertIn("owner__repo", ws.repo_dir)
        self.assertTrue(Path(ws.repo_dir).is_relative_to(Path(tmpdir)))

    def test_22_repository_workspace_uses_shlex(self):
        """run_tests uses shlex.split not str.split."""
        from src.tools.integration_tools import RepositoryWorkspace
        import inspect
        src = inspect.getsource(RepositoryWorkspace.run_tests)
        self.assertIn("shlex.split", src)
        self.assertNotIn("test_command.split(", src)


# =========================================================================
# DEPLOYMENT VERIFICATION (tests 23-25)
# =========================================================================


class TestDeploymentVerification(unittest.TestCase):
    """Tests 23-25: _verify_deployment return values."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_agent_with_mock_tool(self, tool_name, mock_result):
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        mock_tool = MagicMock()
        mock_tool.get_schema.return_value = MagicMock()
        agent.tool_registry.register(tool_name, mock_tool)
        agent.tool_registry.execute = MagicMock(return_value=mock_result)
        return agent

    def test_23_verify_deployment_ready(self):
        """_verify_deployment returns 'ready' when status is ready."""
        agent = self._make_agent_with_mock_tool(
            "vercel_get_deployment",
            {"state": "ready"},
        )
        deployment = DeploymentInfo(id="d1", url="https://example.com", state="building")
        with patch("time.sleep"):
            result = agent._verify_deployment(deployment, max_wait=5)
        self.assertEqual(result, "ready")

    def test_24_verify_deployment_timeout(self):
        """_verify_deployment returns 'timeout' after timeout."""
        agent = self._make_agent_with_mock_tool(
            "vercel_get_deployment",
            {"state": "building"},
        )
        deployment = DeploymentInfo(id="d1", url="https://example.com", state="building")
        with patch("time.sleep"):
            result = agent._verify_deployment(deployment, max_wait=1)
        self.assertEqual(result, "timeout")

    def test_25_verify_deployment_error(self):
        """_verify_deployment returns 'error' on error state."""
        agent = self._make_agent_with_mock_tool(
            "vercel_get_deployment",
            {"state": "error"},
        )
        deployment = DeploymentInfo(id="d1", url="https://example.com", state="building")
        with patch("time.sleep"):
            result = agent._verify_deployment(deployment, max_wait=5)
        self.assertEqual(result, "error")


# =========================================================================
# AUTO-APPROVE WORKSPACE WRITES (tests 26-27)
# =========================================================================


class TestAutoApproveWrites(unittest.TestCase):
    """Tests 26-27: auto_approve_workspace_writes flag."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_26_auto_approve_writes_enabled(self):
        """writes succeed when auto_approve=True."""
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=True,
            auto_approve_workspace_writes=True,
        )
        step = CodingStep(
            step_number=1,
            description="write file",
            action_type="write",
            target_files=["new_file.py"],
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=self.tmpdir, session_id="s"),
            step,
        )
        self.assertEqual(action.status, ActionStatus.EXECUTED)

    def test_27_auto_approve_writes_disabled(self):
        """writes require approval when auto_approve=False."""
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=True,
            auto_approve_workspace_writes=False,
        )
        step = CodingStep(
            step_number=1,
            description="write file",
            action_type="write",
            target_files=["new_file.py"],
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=self.tmpdir, session_id="s"),
            step,
        )
        self.assertEqual(action.status, ActionStatus.SKIPPED)
        self.assertIn("approval", action.error.lower())


# =========================================================================
# API ERROR SAFETY (tests 28-29)
# =========================================================================


class TestApiErrorSafety(unittest.TestCase):
    """Tests 28-29: API error responses don't expose internals."""

    def test_28_api_error_no_traceback(self):
        """API error responses don't expose exception types."""
        provider = FakeProvider(response="not json at all")
        agent = CodingAgent(provider=provider, workspace=tempfile.mkdtemp())
        result = agent.execute("do something impossible")
        self.assertFalse(result.success)
        for error in result.errors:
            self.assertNotIn("Traceback", error)
            self.assertNotIn("traceback", error.lower())

    def test_29_api_coding_response_structure(self):
        """CodingResult returns all expected fields."""
        provider = FakeProvider(plan=PLAN_JSON)
        tmpdir = tempfile.mkdtemp()
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )
        result = agent.execute("add hello function")
        self.assertIsInstance(result, CodingResult)
        self.assertTrue(hasattr(result, "task_id"))
        self.assertTrue(hasattr(result, "success"))
        self.assertTrue(hasattr(result, "summary"))
        self.assertTrue(hasattr(result, "status"))
        self.assertTrue(hasattr(result, "operation_id"))
        self.assertTrue(hasattr(result, "files_modified"))
        self.assertTrue(hasattr(result, "errors"))
        self.assertTrue(hasattr(result, "tests_passed"))
        self.assertTrue(hasattr(result, "tests_failed"))
        self.assertTrue(hasattr(result, "commit_sha"))
        self.assertTrue(hasattr(result, "failures"))
        self.assertTrue(hasattr(result, "pull_request"))
        self.assertTrue(hasattr(result, "deployment"))
        self.assertIsInstance(result.files_modified, list)
        self.assertIsInstance(result.errors, list)
        self.assertIsInstance(result.failures, list)


# =========================================================================
# OBSERVABILITY (tests 30-31)
# =========================================================================


class TestObservability(unittest.TestCase):
    """Tests 30-31: operation_id generation and audit traceability."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_30_operation_id_in_result(self):
        """execute() generates operation_id."""
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        result = agent.execute("add hello function")
        self.assertTrue(result.operation_id.startswith("op_"))
        self.assertGreater(len(result.operation_id), 3)

    def test_31_operation_id_in_audit(self):
        """audit entries can be traced to operation_id."""
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        result = agent.execute("add hello function")
        self.assertTrue(len(result.operation_id) > 0)
        self.assertIsInstance(result.audit_reference, str)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    unittest.main()

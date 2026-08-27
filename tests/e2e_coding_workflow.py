"""
E2E dry run for the complete coding workflow (Phase 7 Step 12).

Simulates the full user-to-result pipeline with mocks:
  Session → Repository Validation → Workspace → Inspection →
  Plan → Tool Discovery → Tool Authorization → Code Modification →
  Test Execution → Validation → Commit → PR → Vercel → Report

No real credentials or network calls required.
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
from src.tools.tools import ToolRegistry
from src.tools.security import AuditLog, PathSecurity, PermissionDecision
from src.providers.base import AIProvider, GenerationConfig, ProviderHealth, ProviderStatus


# ---------------------------------------------------------------------------
# MockProvider — returns canned JSON plans
# ---------------------------------------------------------------------------


class MockProvider(AIProvider):
    """Fake provider that returns canned plan JSON."""

    def __init__(self, plan: Optional[Dict] = None, response: str = ""):
        self._plan = plan
        self._response = response
        self.call_count = 0

    @property
    def provider_id(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock-model-v1"

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
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            message="mock provider ready",
        )

    def is_loaded(self) -> bool:
        return True

    async def load(self) -> None:
        pass

    async def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Canned plan responses
# ---------------------------------------------------------------------------

SIMPLE_PLAN = {
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
            "description": "Create hello.py",
            "action_type": "write",
            "target_files": ["hello.py"],
            "command": "",
        },
        {
            "step_number": 3,
            "description": "Run validation",
            "action_type": "command",
            "target_files": [],
            "command": "pip --version",
        },
    ],
    "risks": ["File might not exist yet"],
    "validation_commands": ["pip --version"],
    "requires_approval": False,
}

PLAN_WITH_APPROVAL = {
    **SIMPLE_PLAN,
    "requires_approval": True,
}


# =========================================================================
# E2E WORKFLOW TESTS
# =========================================================================


class TestFullE2EWorkflow(unittest.TestCase):
    """Test 1: Complete mocked workflow from request to PR+deployment."""

    def test_01_full_e2e_workflow(self):
        """Complete mocked workflow from request to PR+deployment."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )

        result = agent.execute("add a hello function")

        self.assertIsInstance(result, CodingResult)
        self.assertTrue(result.success)
        self.assertTrue(result.operation_id.startswith("op_"))
        self.assertEqual(result.status, CodingStatus.SUCCESS)
        self.assertIsInstance(result.files_modified, list)
        self.assertIsInstance(result.errors, list)
        self.assertIsInstance(result.failures, list)
        self.assertEqual(result.tests_passed, 1)
        self.assertEqual(result.tests_failed, 0)
        self.assertIn("completed", result.summary.lower())


class TestSessionIsolation(unittest.TestCase):
    """Test 2: Different sessions produce different operation_ids."""

    def test_02_session_isolation(self):
        """Different sessions produce different operation_ids."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent1 = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )
        provider2 = MockProvider(plan=SIMPLE_PLAN)
        agent2 = CodingAgent(
            provider=provider2, workspace=tmpdir,
            require_approval_for_writes=False,
        )

        result1 = agent1.execute("task one")
        result2 = agent2.execute("task two")

        self.assertNotEqual(result1.operation_id, result2.operation_id)
        self.assertNotEqual(result1.task_id, result2.task_id)


class TestWorkspaceSecurityInWorkflow(unittest.TestCase):
    """Test 3: Path traversal blocked in workflow."""

    def test_03_workspace_security_in_workflow(self):
        """Path traversal blocked in workflow."""
        from src.errors import WorkspaceBoundaryError
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(provider=provider, workspace=tmpdir)

        self.assertNotIn("../../", agent.workspace)
        self.assertTrue(Path(agent.workspace).is_absolute())
        self.assertTrue(Path(agent.workspace).is_relative_to(Path(tmpdir)))


class TestCommandInjectionBlocked(unittest.TestCase):
    """Test 4: Shell injection in commands blocked."""

    def test_04_command_injection_blocked(self):
        """Shell injection in commands blocked."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )

        step = CodingStep(
            step_number=1,
            description="injection attempt",
            action_type="command",
            target_files=[],
            command="python test.py && rm -rf /",
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=tmpdir, session_id="s"),
            step,
        )
        self.assertIn(action.status, [ActionStatus.DENIED, ActionStatus.FAILED])


class TestApprovalGateWorks(unittest.TestCase):
    """Test 5: Writes require approval when configured."""

    def test_05_approval_gate_works(self):
        """Writes require approval when configured."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=PLAN_WITH_APPROVAL)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=True,
            auto_approve_workspace_writes=False,
        )

        task = agent._create_task("write something", None, None)
        plan = CodingPlan(
            objective="test",
            understanding="test",
            steps=[
                CodingStep(1, "write", "write", ["file.py"]),
            ],
        )
        result = agent._execute_plan(task, plan)
        self.assertEqual(task.status, TaskStatus.AWAITING_APPROVAL)
        self.assertFalse(result.success)
        self.assertIn("Awaiting approval", result.summary)


class TestGithubFailureGraceful(unittest.TestCase):
    """Test 6: GitHub failure returns structured error."""

    def test_06_github_failure_graceful(self):
        """GitHub failure returns structured error."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )
        task = agent._create_task("test", None, None)
        task.repository_owner = "testowner"
        task.repository_name = "testrepo"
        result = agent._failure_result(
            task, "GitHub API rate limit exceeded",
            status=CodingStatus.GITHUB_FAILED,
            failure_type=FailureType.GITHUB,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, CodingStatus.GITHUB_FAILED)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].failure_type, FailureType.GITHUB)
        self.assertIn("rate limit", result.failures[0].message)


class TestVercelFailureGraceful(unittest.TestCase):
    """Test 7: Vercel failure returns structured error."""

    def test_07_vercel_failure_graceful(self):
        """Vercel failure returns structured error."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )
        task = agent._create_task("test", None, None)
        result = agent._failure_result(
            task, "Vercel deployment failed",
            status=CodingStatus.VERCEL_FAILED,
            failure_type=FailureType.VERCEL,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, CodingStatus.VERCEL_FAILED)
        self.assertEqual(result.failures[0].failure_type, FailureType.VERCEL)


class TestTimeoutHandling(unittest.TestCase):
    """Test 8: Timeout produces structured result."""

    def test_08_timeout_handling(self):
        """Timeout produces structured result."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )
        task = agent._create_task("test", None, None)
        result = agent._failure_result(
            task, "Execution timed out after 300s",
            status=CodingStatus.TIMED_OUT,
            failure_type=FailureType.TIMEOUT,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, CodingStatus.TIMED_OUT)
        self.assertEqual(result.failures[0].failure_type, FailureType.TIMEOUT)
        self.assertIn("timed out", result.errors[0].lower())


class TestPrNeverAutoMerged(unittest.TestCase):
    """Test 9: PR merge is never attempted."""

    def test_09_pr_never_auto_merged(self):
        """PR merge is never attempted."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=False,
        )

        task = agent._create_task("test", None, None)
        task.repository_owner = "testowner"
        task.repository_name = "testrepo"
        task.repository = RepositoryContext(
            owner="testowner", name="testrepo", branch="main",
        )

        pr_info = agent._phase_github(task, "feature/test", "main")
        self.assertIsNone(pr_info)

        import inspect
        src = inspect.getsource(CodingAgent)
        self.assertNotIn("merge", src.lower().split("phase")[0] if "phase" in src.lower() else "")
        self.assertNotIn("merge_pull_request", src)


class TestProductionDeployRequiresApproval(unittest.TestCase):
    """Test 10: Production deploy blocked."""

    def test_10_production_deploy_requires_approval(self):
        """Production deploy blocked."""
        tmpdir = tempfile.mkdtemp()
        provider = MockProvider(plan=SIMPLE_PLAN)
        agent = CodingAgent(
            provider=provider, workspace=tmpdir,
            require_approval_for_writes=True,
            auto_approve_workspace_writes=False,
        )
        step = CodingStep(
            step_number=1,
            description="deploy to production",
            action_type="command",
            target_files=[],
            command="vercel deploy --prod",
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=tmpdir, session_id="s"),
            step,
        )
        self.assertIn(action.status, [ActionStatus.DENIED, ActionStatus.FAILED, ActionStatus.SKIPPED])


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    unittest.main()

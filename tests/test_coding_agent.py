"""
Tests for Phase 7 Step 8: CodingAgent.

Covers:
  DATA MODELS (tests 1-7)
  CODING AGENT INIT (tests 8-10)
  TASK CREATION (tests 11-13)
  PLAN GENERATION (tests 14-16)
  STEP EXECUTION (tests 17-21)
  APPROVAL WORKFLOW (tests 22-24)
  BOUNDED FIX LOOP (tests 25-26)
  FULL EXECUTION (tests 27-28)
  ERROR HANDLING (tests 29-30)
  INTEGRATION (tests 31-32)
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant.coding import (
    CodingAction,
    CodingAgent,
    CodingPlan,
    CodingResult,
    CodingStep,
    CodingTask,
    IterationRecord,
    ActionStatus,
    ActionType,
    TaskStatus,
)
from src.assistant.context import ContextBuilder
from src.assistant.session import SessionManager, SessionMode
from src.errors import (
    AgentMaxIterationsError,
    AgentPlanError,
    AgentTimeoutError,
    CodingAgentError,
    WorkspaceBoundaryError,
)
from src.language import Language
from src.memory.memory import MemoryStore
from src.providers.base import AIProvider, GenerationConfig, ModelInfo, ProviderHealth, ProviderStatus
from src.tools.tools import ToolRegistry


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


PLAN_JSON_APPROVAL = {
    **PLAN_JSON,
    "requires_approval": True,
}


# =========================================================================
# DATA MODELS (tests 1-7)
# =========================================================================


class TestDataModels(unittest.TestCase):
    """Tests 1-7: Structured model creation and defaults."""

    def test_01_coding_task_defaults(self):
        task = CodingTask(
            task_id="t1",
            user_request="hello",
            workspace="/tmp",
            session_id="s1",
        )
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.language, Language.UNKNOWN)
        self.assertEqual(task.constraints, [])
        self.assertIn("t1", task.task_id)

    def test_02_coding_step_defaults(self):
        step = CodingStep(step_number=1, description="test", action_type="read")
        self.assertEqual(step.status, ActionStatus.PENDING)
        self.assertEqual(step.target_files, [])
        self.assertEqual(step.command, "")

    def test_03_coding_plan_creation(self):
        plan = CodingPlan(
            objective="Fix bug",
            understanding="There is a bug in foo.py",
            relevant_files=["foo.py"],
            steps=[],
            risks=["Could break other things"],
            validation_commands=["pytest"],
        )
        self.assertEqual(plan.objective, "Fix bug")
        self.assertEqual(len(plan.risks), 1)
        self.assertFalse(plan.requires_approval)

    def test_04_coding_action_creation(self):
        action = CodingAction(
            action_id="a1",
            action_type=ActionType.READ,
            target="foo.py",
        )
        self.assertEqual(action.status, ActionStatus.PENDING)
        self.assertIsNone(action.result)

    def test_05_coding_result_defaults(self):
        result = CodingResult(task_id="t1", success=True, summary="ok")
        self.assertEqual(result.files_modified, [])
        self.assertEqual(result.errors, [])
        self.assertEqual(result.iterations, 0)

    def test_06_iteration_record(self):
        rec = IterationRecord(
            iteration=1,
            previous_error="test failed",
            attempted_fix="changed foo",
            result="passed",
            success=True,
        )
        self.assertTrue(rec.success)
        self.assertEqual(rec.iteration, 1)

    def test_07_task_status_transitions(self):
        task = CodingTask(
            task_id="t1",
            user_request="hello",
            workspace="/tmp",
            session_id="s1",
        )
        self.assertEqual(task.status, TaskStatus.PENDING)
        task.status = TaskStatus.ANALYZING
        self.assertEqual(task.status, TaskStatus.ANALYZING)
        task.status = TaskStatus.EXECUTING
        self.assertEqual(task.status, TaskStatus.EXECUTING)
        task.status = TaskStatus.COMPLETED
        self.assertEqual(task.status, TaskStatus.COMPLETED)


# =========================================================================
# CODING AGENT INIT (tests 8-10)
# =========================================================================


class TestCodingAgentInit(unittest.TestCase):
    """Tests 8-10: Agent initialization with various configs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_08_default_init(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        self.assertEqual(agent.workspace, str(Path(self.tmpdir).resolve()))
        self.assertEqual(agent.max_plan_steps, 10)
        self.assertEqual(agent.max_fix_iterations, 3)
        self.assertTrue(agent.require_approval_for_writes)

    def test_09_custom_config(self):
        provider = FakeProvider()
        agent = CodingAgent(
            provider=provider,
            workspace=self.tmpdir,
            max_plan_steps=5,
            max_fix_iterations=2,
            max_context_chars=4000,
            require_approval_for_writes=False,
        )
        self.assertEqual(agent.max_plan_steps, 5)
        self.assertEqual(agent.max_fix_iterations, 2)
        self.assertEqual(agent.max_context_chars, 4000)
        self.assertFalse(agent.require_approval_for_writes)

    def test_10_session_manager_injection(self):
        provider = FakeProvider()
        sm = SessionManager(max_sessions=5)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir, session_manager=sm)
        self.assertIs(agent.session_manager, sm)


# =========================================================================
# TASK CREATION (tests 11-13)
# =========================================================================


class TestTaskCreation(unittest.TestCase):
    """Tests 11-13: Task creation and validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        provider = FakeProvider()
        self.agent = CodingAgent(provider=provider, workspace=self.tmpdir)

    def test_11_create_task(self):
        task = self.agent._create_task("add hello world", None, None)
        self.assertIsInstance(task, CodingTask)
        self.assertEqual(task.user_request, "add hello world")
        self.assertEqual(task.workspace, str(Path(self.tmpdir).resolve()))
        self.assertIsInstance(task.session_id, str)
        self.assertTrue(task.session_id.startswith("sess_"))

    def test_12_create_task_with_constraints(self):
        task = self.agent._create_task(
            "add function", None, ["no external libs", "keep it simple"]
        )
        self.assertEqual(len(task.constraints), 2)
        self.assertIn("no external libs", task.constraints)

    def test_13_create_task_armenian_request(self):
        task = self.agent._create_task("գրիր hello world ֆունկցիա", None, None)
        self.assertEqual(task.language, Language.ARMENIAN)

    def test_13b_create_task_existing_session(self):
        sm = SessionManager()
        session = sm.create(SessionMode.CODING)
        task = self.agent._create_task("hello", session.session_id, None)
        self.assertEqual(task.session_id, session.session_id)


# =========================================================================
# PLAN GENERATION (tests 14-16)
# =========================================================================


class TestPlanGeneration(unittest.TestCase):
    """Tests 14-16: LLM plan generation and parsing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_14_generate_plan_valid_json(self):
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("add hello", None, None)
        plan = agent._generate_plan(task)
        self.assertIsInstance(plan, CodingPlan)
        self.assertEqual(plan.objective, "Add a hello world function")
        self.assertEqual(len(plan.steps), 3)

    def test_15_parse_plan_valid(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("test", None, None)
        plan = agent._parse_plan(json.dumps(PLAN_JSON), task)
        self.assertEqual(plan.objective, "Add a hello world function")
        self.assertEqual(plan.steps[0].action_type, "command")
        self.assertEqual(plan.steps[0].command, "pip --version")

    def test_16_parse_plan_invalid_json(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("test", None, None)
        with self.assertRaises(AgentPlanError):
            agent._parse_plan("this is not json at all", task)


# =========================================================================
# STEP EXECUTION (tests 17-21)
# =========================================================================


class TestStepExecution(unittest.TestCase):
    """Tests 17-21: Individual step execution."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hello_file = Path(self.tmpdir) / "hello.py"
        self.hello_file.write_text("def hello():\n    return 'world'\n")

    def test_17_exec_read(self):
        provider = FakeProvider()
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        step = CodingStep(
            step_number=1,
            description="read hello.py",
            action_type="read",
            target_files=["hello.py"],
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=self.tmpdir, session_id="s"),
            step,
        )
        self.assertEqual(action.status, ActionStatus.EXECUTED)
        self.assertIn("def hello", action.content)

    def test_18_exec_read_nonexistent(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        step = CodingStep(
            step_number=1,
            description="read missing",
            action_type="read",
            target_files=["no_such_file.py"],
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=self.tmpdir, session_id="s"),
            step,
        )
        self.assertEqual(action.status, ActionStatus.FAILED)

    def test_19_exec_command(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        step = CodingStep(
            step_number=1,
            description="check pip version",
            action_type="command",
            target_files=[],
            command="pip --version",
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=self.tmpdir, session_id="s"),
            step,
        )
        self.assertEqual(action.status, ActionStatus.EXECUTED)

    def test_20_exec_search(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        step = CodingStep(
            step_number=1,
            description="search for hello",
            action_type="search",
            target_files=["hello"],
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=self.tmpdir, session_id="s"),
            step,
        )
        self.assertEqual(action.status, ActionStatus.EXECUTED)
        matches = json.loads(action.content)
        self.assertIsInstance(matches, list)

    def test_21_exec_unknown_action(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        step = CodingStep(
            step_number=1,
            description="unknown",
            action_type="unknown_type",
            target_files=[],
        )
        action = agent._execute_step(
            CodingTask(task_id="t", user_request="r", workspace=self.tmpdir, session_id="s"),
            step,
        )
        self.assertEqual(action.status, ActionStatus.FAILED)
        self.assertIsNotNone(action.error)


# =========================================================================
# APPROVAL WORKFLOW (tests 22-24)
# =========================================================================


class TestApprovalWorkflow(unittest.TestCase):
    """Tests 22-24: Write approval gating."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_22_write_requires_approval(self):
        provider = FakeProvider()
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

    def test_23_approval_bypass(self):
        provider = FakeProvider()
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        provider._plan = PLAN_JSON
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

    def test_24_awaiting_approval_result(self):
        provider = FakeProvider()
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
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


# =========================================================================
# BOUNDED FIX LOOP (tests 25-26)
# =========================================================================


class TestBoundedFixLoop(unittest.TestCase):
    """Tests 25-26: Fix loop behavior."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_25_fix_loop_max_iterations(self):
        fail_script = Path(self.tmpdir) / "_fail.py"
        fail_script.write_text("import sys; sys.exit(1)\n")
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            max_fix_iterations=2,
        )
        task = agent._create_task("fix", None, None)
        plan = CodingPlan(
            objective="test",
            understanding="test",
            steps=[],
            validation_commands=["python _fail.py"],
        )
        result = agent._bounded_fix_loop(
            task, plan, [], ["test failed"], {"python _fail.py": False}
        )
        self.assertIsNone(result)

    def test_26_fix_loop_succeeds(self):
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            max_fix_iterations=3,
            require_approval_for_writes=False,
        )
        task = agent._create_task("fix", None, None)
        plan = CodingPlan(
            objective="test",
            understanding="test",
            steps=[],
            validation_commands=["pip --version"],
        )
        result = agent._bounded_fix_loop(
            task, plan, [], ["test failed"], {"pip --version": False}
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.success)


# =========================================================================
# FULL EXECUTION (tests 27-28)
# =========================================================================


class TestFullExecution(unittest.TestCase):
    """Tests 27-28: End-to-end execution."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_27_execute_success(self):
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        result = agent.execute("add a hello function")
        self.assertIsInstance(result, CodingResult)
        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 0)
        self.assertIn("completed", result.summary.lower())

    def test_28_execute_plan_failure(self):
        provider = FakeProvider(response="this is not json and has no plan")
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        result = agent.execute("do something impossible")
        self.assertFalse(result.success)
        self.assertTrue(len(result.errors) > 0)


# =========================================================================
# ERROR HANDLING (tests 29-30)
# =========================================================================


class TestErrorHandling(unittest.TestCase):
    """Tests 29-30: Error paths and edge cases."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_29_failure_result(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("test", None, None)
        result = agent._failure_result(task, "something broke")
        self.assertFalse(result.success)
        self.assertIn("something broke", result.errors[0])

    def test_30_project_inspection_empty(self):
        provider = FakeProvider()
        agent = CodingAgent(provider=provider, workspace=self.tmpdir)
        task = agent._create_task("test", None, None)
        ctx = agent._inspect_project(task)
        self.assertIn("Workspace:", ctx)


# =========================================================================
# INTEGRATION (tests 31-32)
# =========================================================================


class TestIntegration(unittest.TestCase):
    """Tests 31-32: Integration with existing infrastructure."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_31_uses_tool_registry(self):
        provider = FakeProvider(plan=PLAN_JSON)
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            require_approval_for_writes=False,
        )
        self.assertIsInstance(agent.tool_registry, ToolRegistry)

    def test_32_uses_session_manager(self):
        provider = FakeProvider()
        sm = SessionManager()
        agent = CodingAgent(
            provider=provider, workspace=self.tmpdir,
            session_manager=sm,
        )
        self.assertIs(agent.session_manager, sm)
        result = agent.execute("hello")
        sessions = sm.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].mode, SessionMode.CODING)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    unittest.main()

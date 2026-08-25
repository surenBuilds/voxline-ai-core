"""
CodingAgent — autonomous software engineering agent for Voxline.

Architecture:
    User request
        → CodingAgent.execute()
        → Task analysis
        → Workspace validation
        → Project inspection
        → LLM plan generation
        → Action execution (via ToolRegistry)
        → Validation / test execution
        → Bounded fix loop
        → Final report

CodingAgent NEVER imports QwenProvider, transformers, or raw model objects.
All LLM interaction flows through AIProvider.
All file/command operations flow through ToolRegistry with security.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.assistant.context import ContextBuilder
from src.assistant.session import Session, SessionManager, SessionMode
from src.errors import (
    AgentMaxIterationsError,
    AgentPlanError,
    AgentTimeoutError,
    CodingAgentError,
    ProviderError,
    WorkspaceBoundaryError,
)
from src.language import Language, detect_language, LanguagePolicy
from src.memory.memory import MemoryStore
from src.providers.base import AIProvider, GenerationConfig
from src.tools.security import (
    AuditLog,
    AuditEntry,
    FileSizeGuard,
    PathSecurity,
    PermissionDecision,
    ToolPermissionResult,
)
from src.tools.tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    FIXING = "fixing"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class ActionType(Enum):
    READ = "read"
    WRITE = "write"
    COMMAND = "command"
    SEARCH = "search"


class ActionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Structured models
# ---------------------------------------------------------------------------


@dataclass
class CodingTask:
    task_id: str
    user_request: str
    workspace: str
    session_id: str
    project_context: str = ""
    constraints: List[str] = field(default_factory=list)
    language: Language = Language.UNKNOWN
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)


@dataclass
class CodingStep:
    step_number: int
    description: str
    action_type: str
    target_files: List[str] = field(default_factory=list)
    command: str = ""
    status: ActionStatus = ActionStatus.PENDING


@dataclass
class CodingPlan:
    objective: str
    understanding: str
    relevant_files: List[str] = field(default_factory=list)
    steps: List[CodingStep] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    validation_commands: List[str] = field(default_factory=list)
    requires_approval: bool = False


@dataclass
class CodingAction:
    action_id: str
    action_type: ActionType
    target: str
    content: str = ""
    command: str = ""
    status: ActionStatus = ActionStatus.PENDING
    permission_decision: Optional[ToolPermissionResult] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class IterationRecord:
    iteration: int
    previous_error: str
    attempted_fix: str
    result: str
    success: bool


@dataclass
class CodingResult:
    task_id: str
    success: bool
    summary: str
    changes: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    commands_executed: List[str] = field(default_factory=list)
    tests_run: List[str] = field(default_factory=list)
    test_results: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    iterations: int = 0
    audit_reference: str = ""


# ---------------------------------------------------------------------------
# CodingAgent
# ---------------------------------------------------------------------------


class CodingAgent:
    """Autonomous coding agent.

    Uses AIProvider for all LLM interaction.
    Uses ToolRegistry for all file/command operations.
    Respects workspace boundaries and command policies at all times.
    """

    def __init__(
        self,
        provider: AIProvider,
        workspace: str = ".",
        session_manager: Optional[SessionManager] = None,
        context_builder: Optional[ContextBuilder] = None,
        memory_store: Optional[MemoryStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_plan_steps: int = 10,
        max_fix_iterations: int = 3,
        max_context_chars: int = 8000,
        require_approval_for_writes: bool = True,
        timeout: int = 300,
    ):
        self.provider = provider
        self.workspace = str(Path(workspace).resolve())
        self.session_manager = session_manager or SessionManager()
        self.context_builder = context_builder or ContextBuilder(
            memory_store=memory_store, max_chars=max_context_chars,
        )
        self.memory_store = memory_store
        self.tool_registry = tool_registry or ToolRegistry(
            workspace_root=self.workspace,
        )
        self.max_plan_steps = max_plan_steps
        self.max_fix_iterations = max_fix_iterations
        self.max_context_chars = max_context_chars
        self.require_approval_for_writes = require_approval_for_writes
        self.timeout = timeout
        self._path_security = PathSecurity(self.workspace)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        user_request: str,
        session_id: Optional[str] = None,
        constraints: Optional[List[str]] = None,
    ) -> CodingResult:
        """Execute a coding task end-to-end.

        Returns CodingResult with structured output.
        """
        task = self._create_task(user_request, session_id, constraints)

        try:
            project_context = self._inspect_project(task)
            task.project_context = project_context
            task.status = TaskStatus.ANALYZING

            plan = self._generate_plan(task)
            task.status = TaskStatus.EXECUTING

            result = self._execute_plan(task, plan)
            return result

        except AgentPlanError as exc:
            logger.error("Plan generation failed: %s", exc)
            return self._failure_result(task, f"Plan generation failed: {exc}")
        except AgentTimeoutError as exc:
            logger.error("Execution timed out: %s", exc)
            return self._failure_result(task, f"Execution timed out: {exc}")
        except AgentMaxIterationsError as exc:
            logger.error("Max iterations exceeded: %s", exc)
            return self._failure_result(task, f"Max fix iterations exceeded: {exc}")
        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)
            return self._failure_result(task, f"Unexpected error: {exc}")

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    def _create_task(
        self,
        user_request: str,
        session_id: Optional[str],
        constraints: Optional[List[str]],
    ) -> CodingTask:
        ws_result = self._path_security.validate_path(self.workspace)
        if ws_result.decision != PermissionDecision.ALLOWED:
            raise WorkspaceBoundaryError(
                f"Workspace validation failed: {ws_result.reason}"
            )

        if not session_id:
            session = self.session_manager.create(
                SessionMode.CODING,
                metadata={"workspace": self.workspace},
            )
            session_id = session.session_id
        else:
            session = self.session_manager.get_or_create(
                session_id, SessionMode.CODING
            )

        user_lang = detect_language(user_request)
        return CodingTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            user_request=user_request,
            workspace=self.workspace,
            session_id=session_id,
            constraints=constraints or [],
            language=user_lang,
        )

    # ------------------------------------------------------------------
    # Project inspection
    # ------------------------------------------------------------------

    def _inspect_project(self, task: CodingTask) -> str:
        """Inspect project workspace and return a summary."""
        ws = Path(self.workspace)
        if not ws.exists():
            return "Empty workspace."

        lines: List[str] = []
        lines.append(f"Workspace: {self.workspace}")

        try:
            items = sorted(ws.iterdir())
            dirs = [i.name for i in items if i.is_dir()][:20]
            files = [i.name for i in items if i.is_file()][:30]
            if dirs:
                lines.append(f"Directories: {', '.join(dirs)}")
            if files:
                lines.append(f"Files: {', '.join(files)}")
        except PermissionError:
            lines.append("Cannot list workspace root.")

        for name in ("README.md", "README.txt", "README"):
            readme = ws / name
            if readme.exists() and readme.is_file():
                try:
                    content = readme.read_text(encoding="utf-8", errors="replace")
                    lines.append(f"\n--- {name} ---")
                    lines.append(content[:2000])
                except Exception:
                    pass
                break

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Relevant file discovery
    # ------------------------------------------------------------------

    def _discover_relevant_files(
        self, task: CodingTask, plan: CodingPlan
    ) -> List[str]:
        """Find files relevant to the task."""
        ws = Path(self.workspace)
        candidates: List[str] = []

        for pattern in ("**/*.py", "**/*.js", "**/*.ts", "**/*.html"):
            for p in ws.glob(pattern):
                rel = str(p.relative_to(ws))
                if self._is_relevant(rel, task):
                    candidates.append(rel)
                if len(candidates) >= 20:
                    break
            if len(candidates) >= 20:
                break

        if not candidates:
            for p in ws.rglob("*"):
                if p.is_file() and p.suffix:
                    rel = str(p.relative_to(ws))
                    candidates.append(rel)
                    if len(candidates) >= 10:
                        break

        return candidates

    def _is_relevant(self, rel_path: str, task: CodingTask) -> bool:
        lower = rel_path.lower()
        skip_dirs = {"node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"}
        if any(d in lower for d in skip_dirs):
            return False
        req = task.user_request.lower()
        name_part = Path(rel_path).stem.lower()
        return any(w in req for w in name_part.split("_") if len(w) > 2)

    # ------------------------------------------------------------------
    # LLM plan generation
    # ------------------------------------------------------------------

    def _generate_plan(self, task: CodingTask) -> CodingPlan:
        """Use LLM to generate a structured plan."""
        plan_prompt = self._build_plan_prompt(task)
        max_retries = 1

        for attempt in range(max_retries + 1):
            raw = self._call_llm(
                plan_prompt,
                temperature=0.3,
                max_tokens=1000,
            )

            try:
                plan = self._parse_plan(raw, task)
                return plan
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                if attempt < max_retries:
                    logger.warning("Plan parse failed (attempt %d): %s", attempt + 1, exc)
                    plan_prompt = (
                        "Your previous response was not valid JSON. "
                        "You MUST respond with ONLY a valid JSON object matching this schema:\n"
                        '{"objective":"...","understanding":"...","relevant_files":["..."],'
                        '"steps":[{"step_number":1,"description":"...","action_type":"read|write|command",'
                        '"target_files":["..."],"command":""}],"risks":["..."],'
                        '"validation_commands":["..."],"requires_approval":false}\n\n'
                        + plan_prompt
                    )
                else:
                    raise AgentPlanError(
                        f"Failed to parse plan after {max_retries + 1} attempts: {exc}"
                    ) from exc

        raise AgentPlanError("Plan generation exhausted retries")

    def _build_plan_prompt(self, task: CodingTask) -> str:
        lang_instruction = ""
        if task.language == Language.ARMENIAN:
            lang_instruction = LanguagePolicy.get_system_instruction(Language.ARMENIAN) + "\n\n"

        relevant = self._discover_relevant_files(task, CodingPlan(objective="", understanding=""))

        file_context = ""
        for fp in relevant[:10]:
            content = self._safe_read_file(fp)
            if content and not content.startswith("{error"):
                file_context += f"\n--- {fp} ---\n{content[:2000]}\n"

        return (
            f"{lang_instruction}"
            f"You are a software engineering assistant. Generate an implementation plan.\n"
            f"Workspace: {task.workspace}\n"
            f"User request: {task.user_request}\n"
            f"Constraints: {', '.join(task.constraints) if task.constraints else 'None'}\n"
            f"Project context:\n{task.project_context}\n"
            f"Relevant files:\n{file_context}\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"objective":"...","understanding":"...","relevant_files":["..."],'
            f'"steps":[{{"step_number":1,"description":"...","action_type":"read|write|command",'
            f'"target_files":["..."],"command":""}}],'
            f'"risks":["..."],"validation_commands":["..."],"requires_approval":false}}\n'
            f"Maximum {self.max_plan_steps} steps."
        )

    def _parse_plan(self, raw: str, task: CodingTask) -> CodingPlan:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentPlanError(f"Invalid JSON in plan response: {exc}") from exc

        steps = []
        for s in data.get("steps", [])[: self.max_plan_steps]:
            steps.append(CodingStep(
                step_number=s.get("step_number", len(steps) + 1),
                description=s.get("description", ""),
                action_type=s.get("action_type", "read"),
                target_files=s.get("target_files", []),
                command=s.get("command", ""),
            ))

        return CodingPlan(
            objective=data.get("objective", ""),
            understanding=data.get("understanding", ""),
            relevant_files=data.get("relevant_files", []),
            steps=steps,
            risks=data.get("risks", []),
            validation_commands=data.get("validation_commands", []),
            requires_approval=data.get("requires_approval", False),
        )

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def _execute_plan(self, task: CodingTask, plan: CodingPlan) -> CodingResult:
        actions: List[CodingAction] = []
        files_modified: List[str] = []
        commands_executed: List[str] = []
        errors: List[str] = []

        for step in plan.steps:
            action = self._execute_step(task, step)
            actions.append(action)

            if action.status == ActionStatus.DENIED:
                errors.append(f"Step {step.step_number} denied: {action.error}")
                step.status = ActionStatus.DENIED
                continue

            if action.status == ActionStatus.SKIPPED:
                errors.append(f"Step {step.step_number} skipped: needs approval")
                step.status = ActionStatus.SKIPPED
                task.status = TaskStatus.AWAITING_APPROVAL
                return self._awaiting_approval_result(
                    task, plan, actions, errors
                )

            if action.status == ActionStatus.EXECUTED:
                step.status = ActionStatus.EXECUTED
                if action.action_type == ActionType.WRITE:
                    files_modified.append(action.target)
                if action.action_type == ActionType.COMMAND:
                    commands_executed.append(action.command)
            elif action.status == ActionStatus.FAILED:
                step.status = ActionStatus.FAILED
                errors.append(f"Step {step.step_number} failed: {action.error}")

        test_results = self._run_validations(plan.validation_commands, task)

        all_passed = all(test_results.values()) if test_results else True
        has_errors = len(errors) > 0

        if has_errors and not all_passed:
            fix_result = self._bounded_fix_loop(task, plan, actions, errors, test_results)
            if fix_result is not None:
                return fix_result

        success = not has_errors and all_passed
        summary = self._build_summary(task, plan, success, errors)

        audit_entries = self.tool_registry.audit_log.entries
        audit_ref = audit_entries[-1].session_id if audit_entries else ""

        return CodingResult(
            task_id=task.task_id,
            success=success,
            summary=summary,
            changes=[a.content[:200] for a in actions if a.status == ActionStatus.EXECUTED],
            files_modified=files_modified,
            commands_executed=commands_executed,
            tests_run=list(test_results.keys()),
            test_results=test_results,
            errors=errors,
            warnings=plan.risks[:5],
            iterations=0,
            audit_reference=audit_ref,
        )

    def _execute_step(self, task: CodingTask, step: CodingStep) -> CodingAction:
        try:
            action_type = ActionType(step.action_type)
        except ValueError:
            action = CodingAction(
                action_id=f"act_{uuid.uuid4().hex[:8]}",
                action_type=ActionType.READ,
                target=step.target_files[0] if step.target_files else "",
            )
            action.status = ActionStatus.FAILED
            action.error = f"Unknown action type: {step.action_type}"
            return action

        action = CodingAction(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            action_type=action_type,
            target=step.target_files[0] if step.target_files else "",
        )

        try:
            if action.action_type == ActionType.READ:
                return self._exec_read(action, step)
            elif action.action_type == ActionType.WRITE:
                return self._exec_write(action, step, task)
            elif action.action_type == ActionType.COMMAND:
                return self._exec_command(action, step, task)
            elif action.action_type == ActionType.SEARCH:
                return self._exec_search(action, step)
            else:
                action.status = ActionStatus.FAILED
                action.error = f"Unknown action type: {step.action_type}"
                return action
        except Exception as exc:
            action.status = ActionStatus.FAILED
            action.error = str(exc)
            return action

    def _exec_read(self, action: CodingAction, step: CodingStep) -> CodingAction:
        path = step.target_files[0] if step.target_files else action.target
        val = self.tool_registry.validate_request("read_file", path=path)
        if val.decision != PermissionDecision.ALLOWED:
            action.status = ActionStatus.DENIED
            action.error = val.reason
            return action

        result = self.tool_registry.execute("read_file", path=path)
        if isinstance(result, dict) and result.get("error"):
            action.status = ActionStatus.FAILED
            action.error = result["error"]
        else:
            action.content = result if isinstance(result, str) else json.dumps(result)
            action.status = ActionStatus.EXECUTED
            action.result = {"content_length": len(action.content)}
        return action

    def _exec_write(
        self, action: CodingAction, step: CodingStep, task: CodingTask
    ) -> CodingAction:
        if self.require_approval_for_writes:
            action.status = ActionStatus.SKIPPED
            action.error = "Write requires approval"
            action.permission_decision = ToolPermissionResult(
                decision=PermissionDecision.REQUIRES_APPROVAL,
                reason="Write action requires user approval",
            )
            return action

        path = step.target_files[0] if step.target_files else action.target
        content = self._generate_file_content(step, task)

        val = self.tool_registry.validate_request("write_file", path=path)
        if val.decision != PermissionDecision.ALLOWED:
            action.status = ActionStatus.DENIED
            action.error = val.reason
            return action

        result = self.tool_registry.execute(
            "write_file", path=path, content=content,
        )
        if isinstance(result, dict) and result.get("error"):
            action.status = ActionStatus.FAILED
            action.error = result["error"]
        else:
            action.content = content
            action.status = ActionStatus.EXECUTED
            action.result = result
        return action

    def _exec_command(
        self, action: CodingAction, step: CodingStep, task: CodingTask
    ) -> CodingAction:
        command = step.command
        val = self.tool_registry.validate_request("execute_command", command=command)
        if val.decision != PermissionDecision.ALLOWED:
            if val.decision == PermissionDecision.REQUIRES_APPROVAL:
                action.status = ActionStatus.SKIPPED
                action.error = "Command requires approval"
                action.permission_decision = val
                return action
            action.status = ActionStatus.DENIED
            action.error = val.reason
            return action

        result = self.tool_registry.execute(
            "execute_command",
            command=command,
            cwd=self.workspace,
            session_id=task.session_id,
        )
        action.command = command
        if isinstance(result, dict):
            if result.get("success"):
                action.status = ActionStatus.EXECUTED
                action.result = result
            else:
                action.status = ActionStatus.FAILED
                action.error = result.get("error", result.get("stderr", "Command failed"))
                action.result = result
        else:
            action.status = ActionStatus.EXECUTED
            action.result = {"output": str(result)}
        return action

    def _exec_search(self, action: CodingAction, step: CodingStep) -> CodingAction:
        query = step.target_files[0] if step.target_files else ""
        ws = Path(self.workspace)
        matches: List[str] = []
        for pattern in ("**/*.py", "**/*.js", "**/*.ts"):
            for p in ws.glob(pattern):
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    if query.lower() in text.lower():
                        matches.append(str(p.relative_to(ws)))
                except Exception:
                    pass
                if len(matches) >= 10:
                    break
            if len(matches) >= 10:
                break
        action.content = json.dumps(matches)
        action.status = ActionStatus.EXECUTED
        return action

    def _generate_file_content(self, step: CodingStep, task: CodingTask) -> str:
        prompt = (
            f"Generate file content for this coding step:\n"
            f"Task: {task.user_request}\n"
            f"Step: {step.description}\n"
            f"Target: {step.target_files}\n"
            f"Return ONLY the file content, no explanation."
        )
        return self._call_llm(prompt, temperature=0.3, max_tokens=2000)

    # ------------------------------------------------------------------
    # Bounded fix loop
    # ------------------------------------------------------------------

    def _bounded_fix_loop(
        self,
        task: CodingTask,
        plan: CodingPlan,
        actions: List[CodingAction],
        errors: List[str],
        test_results: Dict[str, bool],
    ) -> Optional[CodingResult]:
        records: List[IterationRecord] = []

        for i in range(self.max_fix_iterations):
            task.status = TaskStatus.FIXING
            failed_tests = [t for t, ok in test_results.items() if not ok]
            error_summary = "; ".join(errors[-3:])

            fix_prompt = (
                f"The code changes failed validation.\n"
                f"Failed tests: {', '.join(failed_tests)}\n"
                f"Errors: {error_summary}\n"
                f"Previous iteration: {i}\n"
                f"User request: {task.user_request}\n"
                f"Generate a JSON fix plan with the same schema as before, "
                f"but focused only on fixing the failures."
            )

            try:
                fix_plan = self._generate_plan(
                    CodingTask(
                        task_id=task.task_id,
                        user_request=fix_prompt,
                        workspace=task.workspace,
                        session_id=task.session_id,
                        project_context=task.project_context,
                        language=task.language,
                    )
                )
            except AgentPlanError:
                records.append(IterationRecord(
                    iteration=i + 1,
                    previous_error=error_summary,
                    attempted_fix="Plan generation failed",
                    result="skipped",
                    success=False,
                ))
                continue

            fix_errors: List[str] = []
            for step in fix_plan.steps:
                action = self._execute_step(task, step)
                if action.status == ActionStatus.FAILED:
                    fix_errors.append(action.error or "Unknown error")

            new_test_results = self._run_validations(
                plan.validation_commands, task
            )
            all_passed = all(new_test_results.values()) if new_test_results else True
            test_results.update(new_test_results)

            records.append(IterationRecord(
                iteration=i + 1,
                previous_error=error_summary,
                attempted_fix=f"Executed {len(fix_plan.steps)} fix steps",
                result="passed" if all_passed else "still failing",
                success=all_passed,
            ))

            if all_passed:
                files_modified = [
                    a.target for a in actions
                    if a.status == ActionStatus.EXECUTED
                    and a.action_type == ActionType.WRITE
                ]
                return CodingResult(
                    task_id=task.task_id,
                    success=True,
                    summary=f"Fixed after {i + 1} iteration(s)",
                    files_modified=files_modified,
                    test_results=test_results,
                    errors=[],
                    warnings=plan.risks[:5],
                    iterations=i + 1,
                )

        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validations(
        self, commands: List[str], task: CodingTask
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for cmd in commands:
            val = self.tool_registry.validate_request("execute_command", command=cmd)
            if val.decision != PermissionDecision.ALLOWED:
                results[cmd] = False
                continue

            result = self.tool_registry.execute(
                "execute_command",
                command=cmd,
                cwd=self.workspace,
                session_id=task.session_id,
            )
            if isinstance(result, dict):
                results[cmd] = result.get("success", False)
            else:
                results[cmd] = True
        return results

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 500
    ) -> str:
        import asyncio

        user_lang = detect_language(prompt)
        lang_instruction = LanguagePolicy.get_system_instruction(user_lang)

        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        messages = [
            {"role": "system", "content": lang_instruction},
            {"role": "user", "content": prompt},
        ]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.provider.chat(messages, config),
                )
                return future.result(timeout=120)
        else:
            return asyncio.run(self.provider.chat(messages, config))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_read_file(self, rel_path: str) -> Optional[str]:
        try:
            val = self.tool_registry.validate_request("read_file", path=rel_path)
            if val.decision != PermissionDecision.ALLOWED:
                return None
            result = self.tool_registry.execute("read_file", path=rel_path)
            if isinstance(result, str):
                return result[:3000]
            return None
        except Exception:
            return None

    def _build_summary(
        self,
        task: CodingTask,
        plan: CodingPlan,
        success: bool,
        errors: List[str],
    ) -> str:
        status = "completed" if success else "completed with errors"
        parts = [
            f"Task {status}.",
            f"Objective: {plan.objective}",
            f"Steps executed: {len(plan.steps)}",
        ]
        if errors:
            parts.append(f"Errors: {len(errors)}")
        return " ".join(parts)

    def _failure_result(self, task: CodingTask, error: str) -> CodingResult:
        return CodingResult(
            task_id=task.task_id,
            success=False,
            summary=f"Failed: {error}",
            errors=[error],
        )

    def _awaiting_approval_result(
        self,
        task: CodingTask,
        plan: CodingPlan,
        actions: List[CodingAction],
        errors: List[str],
    ) -> CodingResult:
        pending = [a for a in actions if a.status == ActionStatus.SKIPPED]
        return CodingResult(
            task_id=task.task_id,
            success=False,
            summary=f"Awaiting approval for {len(pending)} action(s)",
            errors=errors + [
                f"Pending approval: {a.action_type.value} {a.target}"
                for a in pending
            ],
        )

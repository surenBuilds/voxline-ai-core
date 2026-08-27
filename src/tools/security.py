"""Voxline AI Core — Tool security layer.

Provides:
- Workspace boundary enforcement (Path.is_relative_to, no startswith)
- Symlink-aware path validation
- Command permission policy (ALLOWED / DENIED / REQUIRES_APPROVAL)
- Structured command parsing (no shell=True by default)
- File size limits
- Audit logging for every tool invocation
- Environment sanitization for subprocess calls
"""

import shlex
import subprocess
import time
import logging
import os
import re
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission decision
# ---------------------------------------------------------------------------

class PermissionDecision(Enum):
    """Result of a permission check."""
    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass(frozen=True)
class ToolPermissionResult:
    """Structured result of a permission authorization check."""
    decision: PermissionDecision
    reason: str
    tool_name: str = ""
    command: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool security profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSecurityProfile:
    """Declares what a tool is allowed to do.

    Default for every capability is False (deny).
    Only capabilities explicitly granted are allowed.
    """
    filesystem_read: bool = False
    filesystem_write: bool = False
    command_execution: bool = False
    network_access: bool = False
    requires_approval: bool = False


# ---------------------------------------------------------------------------
# Audit log entry
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """Single audit log record for a tool invocation."""
    timestamp: float
    session_id: str
    tool_name: str
    operation: str
    decision: PermissionDecision
    reason: str
    success: bool
    duration_ms: float
    exit_code: Optional[int] = None
    error: Optional[str] = None


class AuditLog:
    """Append-only audit log for tool invocations.

    Stores entries in memory for the lifetime of the process.
    Safe to expose to the assistant layer — no secrets are recorded.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: List[AuditEntry] = []
        self._max = max_entries

    def record(
        self,
        *,
        session_id: str = "",
        tool_name: str,
        operation: str,
        decision: PermissionDecision,
        reason: str,
        success: bool,
        duration_ms: float,
        exit_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(),
            session_id=session_id,
            tool_name=tool_name,
            operation=operation,
            decision=decision,
            reason=reason,
            success=success,
            duration_ms=duration_ms,
            exit_code=exit_code,
            error=error,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]
        logger.info(
            "audit tool=%s op=%s decision=%s reason=%s success=%s dur=%.1fms",
            tool_name, operation, decision.value, reason, success, duration_ms,
        )
        return entry

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def log_event(
        self,
        event: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        success: bool = True,
        error: Optional[str] = None,
    ) -> AuditEntry:
        """Log a non-tool event (e.g. an internal git command).

        Flattens optional details into a generic audit entry so callers such as
        RepositoryWorkspace can record framework operations without crashing.
        """
        details = details or {}
        return self.record(
            session_id=session_id,
            tool_name=event,
            operation=event,
            decision=PermissionDecision.ALLOWED,
            reason=str(details) if details else event,
            success=success,
            duration_ms=0.0,
            error=error,
        )

    def clear(self) -> None:
        self._entries.clear()


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

class PathSecurity:
    """Workspace boundary enforcement with symlink awareness.

    Uses Path.resolve() + Path.is_relative_to() — never string prefix checks.

    Design:
        1. Resolve the candidate path to its real, canonical form.
        2. Resolve the workspace root to its real, canonical form.
        3. Check resolved candidate is_relative_to resolved workspace.
        4. Reject if outside, regardless of traversal tricks, symlinks,
           Windows drive letters, or alternate representations.
    """

    def __init__(self, workspace_root: str | Path) -> None:
        self._workspace = Path(workspace_root).resolve()

    @property
    def workspace_root(self) -> Path:
        return self._workspace

    def validate_path(self, raw_path: str | Path) -> ToolPermissionResult:
        """Validate that raw_path resolves inside the workspace.

        Returns ALLOWED if inside, DENIED otherwise.
        The resolved canonical path is available in metadata on success.

        For relative paths: resolves relative to workspace root (not cwd).
        For absolute paths: resolves directly and checks containment.
        """
        candidate = Path(raw_path)

        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            # Relative paths are anchored to workspace, not cwd
            resolved = (self._workspace / candidate).resolve()

        if not resolved.is_relative_to(self._workspace):
            return ToolPermissionResult(
                decision=PermissionDecision.DENIED,
                reason=f"Path escapes workspace: {raw_path} -> {resolved}",
                metadata={"resolved": str(resolved)},
            )
        return ToolPermissionResult(
            decision=PermissionDecision.ALLOWED,
            reason="Path inside workspace",
            metadata={"resolved": str(resolved)},
        )

    def safe_resolve(self, raw_path: str | Path) -> Path:
        """Resolve path and return canonical form, or raise if outside workspace.

        Raises WorkspaceBoundaryError on violation.
        """
        from src.errors import WorkspaceBoundaryError
        result = self.validate_path(raw_path)
        if result.decision != PermissionDecision.ALLOWED:
            raise WorkspaceBoundaryError(result.reason)
        return Path(result.metadata["resolved"])


# ---------------------------------------------------------------------------
# File size enforcement
# ---------------------------------------------------------------------------

class FileSizeGuard:
    """Enforce maximum file size for reads and writes."""

    def __init__(self, max_size_bytes: int) -> None:
        self.max_size_bytes = max_size_bytes

    def check_read(self, file_path: Path) -> ToolPermissionResult:
        """Check that a file to be read is within the size limit."""
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            return ToolPermissionResult(
                decision=PermissionDecision.DENIED,
                reason=f"Cannot stat file: {exc}",
            )
        if size > self.max_size_bytes:
            return ToolPermissionResult(
                decision=PermissionDecision.DENIED,
                reason=f"File too large: {size} bytes (limit {self.max_size_bytes})",
                metadata={"size": size, "limit": self.max_size_bytes},
            )
        return ToolPermissionResult(
            decision=PermissionDecision.ALLOWED,
            reason="File within size limit",
            metadata={"size": size},
        )

    def check_write(self, content: str | bytes) -> ToolPermissionResult:
        """Check that content to be written is within the size limit."""
        size = len(content) if isinstance(content, (str, bytes)) else 0
        if size > self.max_size_bytes:
            return ToolPermissionResult(
                decision=PermissionDecision.DENIED,
                reason=f"Content too large: {size} bytes (limit {self.max_size_bytes})",
                metadata={"size": size, "limit": self.max_size_bytes},
            )
        return ToolPermissionResult(
            decision=PermissionDecision.ALLOWED,
            reason="Content within size limit",
            metadata={"size": size},
        )


# ---------------------------------------------------------------------------
# Command parsing and validation
# ---------------------------------------------------------------------------

class CommandPolicy:
    """Controls which commands are allowed, denied, or require approval.

    Design:
        - allowed_commands: executables that MAY run (not a guarantee of safety)
        - denied_commands: executables that are NEVER allowed
        - requires_approval: executables that need human confirmation
        - blocked_arguments: argument patterns that are always denied
          regardless of executable (e.g. -c, --eval, shell metacharacters)
    """

    def __init__(
        self,
        allowed_commands: Optional[Set[str]] = None,
        denied_commands: Optional[Set[str]] = None,
        requires_approval: Optional[Set[str]] = None,
        blocked_arguments: Optional[List[str]] = None,
    ) -> None:
        self.allowed_commands = allowed_commands or set()
        self.denied_commands = denied_commands or set()
        self.requires_approval = requires_approval or set()
        # Patterns in arguments that are always denied
        self.blocked_arguments = blocked_arguments or [
            "-c", "--eval", "-e",
            "&&", "||", ";", "|", ">", ">>", "<", "$(",
            "`", "${", "\\n",
        ]

    def evaluate(self, executable: str, args: List[str]) -> ToolPermissionResult:
        """Evaluate whether a parsed command is permitted.

        Returns ALLOWED / DENIED / REQUIRES_APPROVAL with reason.
        """
        exe_name = Path(executable).name.lower()

        # Explicit deny always wins
        if exe_name in self.denied_commands or executable in self.denied_commands:
            return ToolPermissionResult(
                decision=PermissionDecision.DENIED,
                reason=f"Command explicitly denied: {executable}",
                tool_name=exe_name,
                command=executable,
            )

        # Check blocked arguments in the full command string
        full_cmd = " ".join(args)
        for pattern in self.blocked_arguments:
            if pattern in full_cmd:
                return ToolPermissionResult(
                    decision=PermissionDecision.DENIED,
                    reason=f"Blocked argument pattern detected: '{pattern}' in '{full_cmd}'",
                    tool_name=exe_name,
                    command=full_cmd,
                )

        # Check for shell metacharacters in any individual arg
        shell_chars = set(";&|`$(){}[]!#~")
        for arg in args:
            if any(c in shell_chars for c in arg):
                return ToolPermissionResult(
                    decision=PermissionDecision.DENIED,
                    reason=f"Shell metacharacter detected in argument: '{arg}'",
                    tool_name=exe_name,
                    command=" ".join(args),
                )

        # Requires approval
        if exe_name in self.requires_approval or executable in self.requires_approval:
            return ToolPermissionResult(
                decision=PermissionDecision.REQUIRES_APPROVAL,
                reason=f"Command requires approval: {executable}",
                tool_name=exe_name,
                command=executable,
            )

        # Allowed check
        if exe_name in self.allowed_commands or executable in self.allowed_commands:
            return ToolPermissionResult(
                decision=PermissionDecision.ALLOWED,
                reason=f"Command in allowed list: {executable}",
                tool_name=exe_name,
                command=executable,
            )

        # Default: deny unknown commands
        return ToolPermissionResult(
            decision=PermissionDecision.DENIED,
            reason=f"Command not in allowed list: {executable}",
            tool_name=exe_name,
            command=executable,
        )


class CommandValidator:
    """Parse and validate a command string safely.

    Uses shlex.split for structured parsing.
    Never uses shell=True.
    Validates executable + arguments against CommandPolicy.
    Validates working directory against workspace boundary.
    """

    def __init__(
        self,
        policy: CommandPolicy,
        path_security: PathSecurity,
        max_output_bytes: int = 1_048_576,
        timeout_seconds: int = 60,
    ) -> None:
        self.policy = policy
        self.path_security = path_security
        self.max_output_bytes = max_output_bytes
        self.timeout_seconds = timeout_seconds

    def set_path_security(self, path_security: PathSecurity) -> None:
        """Update the workspace-boundary security object.

        Used when the workspace root changes mid-workflow (e.g. after a
        repository clone) so command validation enforces the new boundary.
        """
        self.path_security = path_security

    def parse_command(self, command: str) -> tuple[str, List[str]]:
        """Safely parse a command string into executable and arguments.

        Returns (executable, [args]).
        Raises ValueError on unparseable input.

        Uses posix=False on Windows to preserve backslash path separators.
        """
        try:
            parts = shlex.split(command, posix=(os.name != "nt"))
        except ValueError as exc:
            raise ValueError(f"Cannot parse command: {exc}") from exc
        if not parts:
            raise ValueError("Empty command")
        return parts[0], parts[1:]

    def validate_cwd(self, cwd: Optional[str | Path]) -> ToolPermissionResult:
        """Validate that the working directory is inside the workspace."""
        if cwd is None:
            return ToolPermissionResult(
                decision=PermissionDecision.ALLOWED,
                reason="No working directory specified, using workspace root",
            )
        return self.path_security.validate_path(cwd)

    def build_safe_env(self, secrets: Optional[Set[str]] = None) -> Dict[str, str]:
        """Build a clean environment for subprocess, excluding secrets.

        Inherits PATH but removes variables matching the secrets set.
        Never exposes API keys, tokens, or passwords.
        """
        secret_keys = secrets or {
            "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
            "WEB_SEARCH_API_KEY", "INSTAGRAM_ACCESS_TOKEN",
            "INSTAGRAM_ACCOUNT_ID", "DATABASE_PASSWORD",
            "VOXLINE_SECRET", "SECRET_KEY", "API_SECRET",
        }
        safe_env: Dict[str, str] = {}
        for key, value in os.environ.items():
            if key.upper() in secret_keys or "SECRET" in key.upper() or "TOKEN" in key.upper() or "KEY" in key.upper() or "PASSWORD" in key.upper():
                continue
            safe_env[key] = value
        return safe_env

    def run_command(
        self,
        command: str,
        cwd: Optional[str | Path] = None,
        session_id: str = "",
        audit_log: Optional[AuditLog] = None,
    ) -> Dict[str, Any]:
        """Execute a command safely.

        1. Parse command
        2. Validate working directory
        3. Evaluate permission
        4. Run with shell=False, timeout, output limit
        5. Audit the result

        Returns dict with stdout, stderr, exit_code, success, error fields.
        """
        t0 = time.time()

        # Parse
        try:
            executable, args = self.parse_command(command)
        except ValueError as exc:
            return {
                "success": False,
                "error": str(exc),
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
            }

        # Validate cwd
        cwd_result = self.validate_cwd(cwd)
        if cwd_result.decision != PermissionDecision.ALLOWED:
            return {
                "success": False,
                "error": cwd_result.reason,
                "exit_code": None,
                "stdout": "",
                "stderr": cwd_result.reason,
            }

        resolved_cwd = cwd_result.metadata.get("resolved") if cwd_result.metadata else None

        # Permission check
        perm = self.policy.evaluate(executable, args)
        if perm.decision == PermissionDecision.DENIED:
            dur = (time.time() - t0) * 1000
            if audit_log:
                audit_log.record(
                    session_id=session_id,
                    tool_name="execute_command",
                    operation=command,
                    decision=PermissionDecision.DENIED,
                    reason=perm.reason,
                    success=False,
                    duration_ms=dur,
                )
            return {
                "success": False,
                "error": perm.reason,
                "exit_code": None,
                "stdout": "",
                "stderr": perm.reason,
            }

        if perm.decision == PermissionDecision.REQUIRES_APPROVAL:
            dur = (time.time() - t0) * 1000
            if audit_log:
                audit_log.record(
                    session_id=session_id,
                    tool_name="execute_command",
                    operation=command,
                    decision=PermissionDecision.REQUIRES_APPROVAL,
                    reason=perm.reason,
                    success=False,
                    duration_ms=dur,
                )
            return {
                "success": False,
                "error": perm.reason,
                "requires_approval": True,
                "exit_code": None,
                "stdout": "",
                "stderr": perm.reason,
            }

        # Execute
        safe_env = self.build_safe_env()
        full_args = [executable] + args

        try:
            result = subprocess.run(
                full_args,
                cwd=resolved_cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=safe_env,
                shell=False,
            )
            stdout = result.stdout[:self.max_output_bytes] if result.stdout else ""
            stderr = result.stderr[:self.max_output_bytes] if result.stderr else ""

            truncated = (
                len(result.stdout or "") > self.max_output_bytes
                or len(result.stderr or "") > self.max_output_bytes
            )

            success = result.returncode == 0
            dur = (time.time() - t0) * 1000

            if audit_log:
                audit_log.record(
                    session_id=session_id,
                    tool_name="execute_command",
                    operation=command,
                    decision=PermissionDecision.ALLOWED,
                    reason=perm.reason,
                    success=success,
                    duration_ms=dur,
                    exit_code=result.returncode,
                )

            return {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "truncated": truncated,
            }

        except subprocess.TimeoutExpired:
            dur = (time.time() - t0) * 1000
            error_msg = f"Command timed out after {self.timeout_seconds}s"
            if audit_log:
                audit_log.record(
                    session_id=session_id,
                    tool_name="execute_command",
                    operation=command,
                    decision=PermissionDecision.ALLOWED,
                    reason=perm.reason,
                    success=False,
                    duration_ms=dur,
                    error=error_msg,
                )
            return {
                "success": False,
                "error": error_msg,
                "exit_code": None,
                "stdout": "",
                "stderr": error_msg,
            }

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            if audit_log:
                audit_log.record(
                    session_id=session_id,
                    tool_name="execute_command",
                    operation=command,
                    decision=PermissionDecision.ALLOWED,
                    reason=perm.reason,
                    success=False,
                    duration_ms=dur,
                    error=str(exc),
                )
            return {
                "success": False,
                "error": str(exc),
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
            }

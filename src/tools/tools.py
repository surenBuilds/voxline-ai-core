"""Generic tool registry and execution system

Tools define:
- name
- description
- input schema
- execution function
- permissions
- timeout

Security integration:
- PathSecurity for workspace boundary enforcement
- FileSizeGuard for file size limits
- CommandValidator for safe command execution
- AuditLog for every tool invocation
- ToolSecurityProfile for capability declarations
"""

import json
import subprocess
import inspect
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .security import (
    PathSecurity,
    FileSizeGuard,
    CommandValidator,
    CommandPolicy,
    AuditLog,
    PermissionDecision,
    ToolPermissionResult,
    ToolSecurityProfile,
)


@dataclass
class ToolSchema:
    """Tool schema definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    permissions: List[str]  # e.g., ["read_file", "write_file", "execute_code"]
    timeout: int = 30


class Tool(ABC):
    """Abstract base class for tools."""

    def __init__(self, security_profile: Optional[ToolSecurityProfile] = None) -> None:
        self._security_profile = security_profile or ToolSecurityProfile()

    @property
    def security_profile(self) -> ToolSecurityProfile:
        return self._security_profile

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute tool with provided arguments."""
        pass

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        pass


class Calculator(Tool):
    """Simple calculator tool."""

    def __init__(self) -> None:
        super().__init__(ToolSecurityProfile(filesystem_read=False, filesystem_write=False, command_execution=False))

    def execute(self, expression: str) -> float:
        """Evaluate mathematical expression."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            return {"error": str(e)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description="Evaluate mathematical expressions",
            input_schema={"expression": {"type": "string"}},
            output_schema={"type": "number"},
            permissions=["compute"],
        )


class FileReadTool(Tool):
    """Read file tool with workspace boundary checks (Path.is_relative_to)."""

    def __init__(
        self,
        workspace_root: str = ".",
        file_size_guard: Optional[FileSizeGuard] = None,
        path_security: Optional[PathSecurity] = None,
    ) -> None:
        super().__init__(ToolSecurityProfile(filesystem_read=True))
        self._path_security = path_security or PathSecurity(workspace_root)
        self._size_guard = file_size_guard or FileSizeGuard(max_size_bytes=10 * 1024 * 1024)

    def execute(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Read file within workspace."""
        # Validate path
        result = self._path_security.validate_path(path)
        if result.decision != PermissionDecision.ALLOWED:
            return {"error": result.reason}

        resolved = Path(result.metadata["resolved"])

        if not resolved.exists():
            return {"error": f"File not found: {path}"}

        if not resolved.is_file():
            return {"error": f"Not a file: {path}"}

        # Check file size
        size_result = self._size_guard.check_read(resolved)
        if size_result.decision != PermissionDecision.ALLOWED:
            return {"error": size_result.reason}

        try:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()

            if start_line is not None or end_line is not None:
                lines = content.split("\n")
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                content = "\n".join(lines[start:end])

            return content
        except Exception as e:
            return {"error": str(e)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read file content within workspace",
            input_schema={
                "path": {"type": "string"},
                "start_line": {"type": "integer", "optional": True},
                "end_line": {"type": "integer", "optional": True},
            },
            output_schema={"type": "string"},
            permissions=["read_file"],
        )


class FileWriteTool(Tool):
    """Write file tool with workspace boundary and size checks (Path.is_relative_to)."""

    def __init__(
        self,
        workspace_root: str = ".",
        file_size_guard: Optional[FileSizeGuard] = None,
        path_security: Optional[PathSecurity] = None,
    ) -> None:
        super().__init__(ToolSecurityProfile(filesystem_write=True))
        self._path_security = path_security or PathSecurity(workspace_root)
        self._size_guard = file_size_guard or FileSizeGuard(max_size_bytes=10 * 1024 * 1024)

    def execute(self, path: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """Write file within workspace."""
        # Validate path
        result = self._path_security.validate_path(path)
        if result.decision != PermissionDecision.ALLOWED:
            return {"error": result.reason, "success": False}

        resolved = Path(result.metadata["resolved"])

        # Check content size before writing
        size_result = self._size_guard.check_write(content)
        if size_result.decision != PermissionDecision.ALLOWED:
            return {"error": size_result.reason, "success": False}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)

            with open(resolved, mode, encoding="utf-8") as f:
                f.write(content)

            return {"success": True, "path": str(resolved)}
        except Exception as e:
            return {"error": str(e), "success": False}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write file within workspace",
            input_schema={
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["w", "a"], "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["write_file"],
        )


class DirectoryListTool(Tool):
    """List directory contents with workspace boundary checks (Path.is_relative_to)."""

    def __init__(
        self,
        workspace_root: str = ".",
        path_security: Optional[PathSecurity] = None,
    ) -> None:
        super().__init__(ToolSecurityProfile(filesystem_read=True))
        self._path_security = path_security or PathSecurity(workspace_root)

    def execute(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        result = self._path_security.validate_path(path)
        if result.decision != PermissionDecision.ALLOWED:
            return {"error": result.reason}

        resolved = Path(result.metadata["resolved"])

        if not resolved.is_dir():
            return {"error": f"Not a directory: {path}"}

        items = []
        for item in sorted(resolved.iterdir()):
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })

        return {"items": items, "count": len(items)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_directory",
            description="List directory contents within workspace",
            input_schema={"path": {"type": "string", "optional": True}},
            output_schema={"type": "object"},
            permissions=["read_file"],
        )


class CommandExecutor(Tool):
    """Execute system commands safely via CommandValidator.

    Never uses shell=True.
    Uses timeout, output limit, permission checks, audit logging.
    """

    def __init__(
        self,
        command_validator: Optional[CommandValidator] = None,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        super().__init__(ToolSecurityProfile(command_execution=True))
        self._validator = command_validator
        self._audit = audit_log or AuditLog()

    def execute(self, command: str, cwd: Optional[str] = None, session_id: str = "") -> Dict[str, Any]:
        if self._validator is None:
            return {"success": False, "error": "CommandValidator not configured"}
        return self._validator.run_command(
            command=command,
            cwd=cwd,
            session_id=session_id,
            audit_log=self._audit,
        )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="execute_command",
            description="Execute a system command within workspace",
            input_schema={
                "command": {"type": "string"},
                "cwd": {"type": "string", "optional": True},
                "session_id": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["execute_command"],
            timeout=60,
        )


class ToolRegistry:
    """Registry of available tools with security integration.

    Exposes three-phase API for the future CodingAgent:
        validate_request()  — path / input sanity
        authorize_request() — permission decision
        execute()           — run tool + audit

    The three phases must not be combined into one uncontrolled method.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        max_file_size_bytes: int = 10 * 1024 * 1024,
        max_output_bytes: int = 1_048_576,
        command_timeout: int = 60,
        allowed_commands: Optional[List[str]] = None,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self.workspace_root = workspace_root
        self._path_security = PathSecurity(workspace_root)
        self._file_size_guard = FileSizeGuard(max_size_bytes=max_file_size_bytes)
        self._audit = audit_log or AuditLog()

        policy = CommandPolicy(
            allowed_commands=set(allowed_commands or ["python", "pytest", "pip", "git"]),
        )
        self._command_validator = CommandValidator(
            policy=policy,
            path_security=self._path_security,
            max_output_bytes=max_output_bytes,
            timeout_seconds=command_timeout,
        )

        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register built-in tools."""
        self.register("calculator", Calculator())
        self.register("read_file", FileReadTool(self.workspace_root, self._file_size_guard, self._path_security))
        self.register("write_file", FileWriteTool(self.workspace_root, self._file_size_guard, self._path_security))
        self.register("list_directory", DirectoryListTool(self.workspace_root, self._path_security))
        self.register("execute_command", CommandExecutor(self._command_validator, self._audit))

    def register(self, name: str, tool: Tool) -> None:
        """Register a tool."""
        self.tools[name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)

    # ---- Three-phase API ------------------------------------------------

    def validate_request(self, name: str, **kwargs) -> ToolPermissionResult:
        """Phase 1: Validate input before authorization.

        Checks tool existence and basic input sanity.
        For file tools, validates path against workspace boundary.
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolPermissionResult(
                decision=PermissionDecision.DENIED,
                reason=f"Tool not found: {name}",
                tool_name=name,
            )

        # Path validation for file tools
        if name in ("read_file", "write_file", "list_directory"):
            path = kwargs.get("path", ".")
            return self._path_security.validate_path(path)

        # Command validation for execute_command
        if name == "execute_command":
            command = kwargs.get("command", "")
            if not command.strip():
                return ToolPermissionResult(
                    decision=PermissionDecision.DENIED,
                    reason="Empty command",
                    tool_name=name,
                )
            try:
                self._command_validator.parse_command(command)
            except ValueError as exc:
                return ToolPermissionResult(
                    decision=PermissionDecision.DENIED,
                    reason=str(exc),
                    tool_name=name,
                )
            return ToolPermissionResult(
                decision=PermissionDecision.ALLOWED,
                reason="Command parseable",
                tool_name=name,
            )

        # No validation needed for other tools (e.g. calculator)
        return ToolPermissionResult(
            decision=PermissionDecision.ALLOWED,
            reason="No validation required",
            tool_name=name,
        )

    def authorize_request(self, name: str, **kwargs) -> ToolPermissionResult:
        """Phase 2: Authorization decision.

        For file tools: confirms workspace boundary (already checked by validate_request,
        but this is the authoritative security gate).
        For command tools: runs full permission policy evaluation.
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolPermissionResult(
                decision=PermissionDecision.DENIED,
                reason=f"Tool not found: {name}",
                tool_name=name,
            )

        if name in ("read_file", "write_file", "list_directory"):
            path = kwargs.get("path", ".")
            return self._path_security.validate_path(path)

        if name == "execute_command":
            command = kwargs.get("command", "")
            try:
                executable, args = self._command_validator.parse_command(command)
            except ValueError as exc:
                return ToolPermissionResult(
                    decision=PermissionDecision.DENIED,
                    reason=str(exc),
                    tool_name=name,
                )
            return self._command_validator.policy.evaluate(executable, args)

        # Calculator — always allowed
        return ToolPermissionResult(
            decision=PermissionDecision.ALLOWED,
            reason="Tool permitted",
            tool_name=name,
        )

    def execute(self, name: str, session_id: str = "", **kwargs) -> Any:
        """Phase 3: Execute tool with full audit logging.

        Combines validate + authorize + execute in one call for convenience.
        The CodingAgent should use the three phases separately.
        """
        t0 = time.time()

        val = self.validate_request(name, **kwargs)
        if val.decision != PermissionDecision.ALLOWED:
            dur = (time.time() - t0) * 1000
            self._audit.record(
                session_id=session_id,
                tool_name=name,
                operation=str(kwargs)[:200],
                decision=val.decision,
                reason=val.reason,
                success=False,
                duration_ms=dur,
            )
            return {"error": val.reason, "success": False}

        auth = self.authorize_request(name, **kwargs)
        if auth.decision != PermissionDecision.ALLOWED:
            dur = (time.time() - t0) * 1000
            self._audit.record(
                session_id=session_id,
                tool_name=name,
                operation=str(kwargs)[:200],
                decision=auth.decision,
                reason=auth.reason,
                success=False,
                duration_ms=dur,
            )
            return {"error": auth.reason, "success": False, "requires_approval": auth.decision == PermissionDecision.REQUIRES_APPROVAL}

        # Dispatch
        tool = self.get_tool(name)
        try:
            if name == "execute_command":
                result = tool.execute(
                    command=kwargs.get("command", ""),
                    cwd=kwargs.get("cwd"),
                    session_id=session_id,
                )
            elif name == "write_file":
                result = tool.execute(
                    path=kwargs.get("path", ""),
                    content=kwargs.get("content", ""),
                    mode=kwargs.get("mode", "w"),
                )
            else:
                result = tool.execute(**{k: v for k, v in kwargs.items() if k != "session_id"})

            dur = (time.time() - t0) * 1000
            success = not (isinstance(result, dict) and result.get("error"))
            self._audit.record(
                session_id=session_id,
                tool_name=name,
                operation=str(kwargs)[:200],
                decision=PermissionDecision.ALLOWED,
                reason=auth.reason,
                success=success,
                duration_ms=dur,
                exit_code=result.get("exit_code") if isinstance(result, dict) else None,
                error=result.get("error") if isinstance(result, dict) and result.get("error") else None,
            )
            return result

        except Exception as e:
            dur = (time.time() - t0) * 1000
            self._audit.record(
                session_id=session_id,
                tool_name=name,
                operation=str(kwargs)[:200],
                decision=PermissionDecision.ALLOWED,
                reason=auth.reason,
                success=False,
                duration_ms=dur,
                error=str(e),
            )
            return {"error": str(e)}

    # ---- Backward compatibility alias -----------------------------------

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool (backward-compatible alias)."""
        return self.execute(name, **kwargs)

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools with schemas."""
        tools_info = {}
        for name, tool in self.tools.items():
            schema = tool.get_schema()
            tools_info[name] = {
                "description": schema.description,
                "input_schema": schema.input_schema,
                "output_schema": schema.output_schema,
                "permissions": schema.permissions,
            }
        return tools_info

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

"""
Generic tool registry and execution system

Tools define:
- name
- description
- input schema
- execution function
- permissions
- timeout
"""

import json
import subprocess
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
    """Read file tool with workspace boundary checks."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def execute(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """
        Read file within workspace.

        Args:
            path: File path (relative to workspace)
            start_line: Optional start line
            end_line: Optional end line

        Returns:
            File content
        """
        try:
            file_path = (self.workspace_root / path).resolve()

            # Security check: ensure file is within workspace
            if not str(file_path).startswith(str(self.workspace_root)):
                return {"error": f"Access denied: {path} is outside workspace"}

            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Handle line range
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
    """Write file tool with workspace boundary checks."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def execute(self, path: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """
        Write file within workspace.

        Args:
            path: File path (relative to workspace)
            content: File content
            mode: Write mode ("w" or "a")

        Returns:
            Status dict
        """
        try:
            file_path = (self.workspace_root / path).resolve()

            # Security check
            if not str(file_path).startswith(str(self.workspace_root)):
                return {"error": f"Access denied: {path} is outside workspace", "success": False}

            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            return {"success": True, "path": str(file_path)}
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
    """List directory contents."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()

    def execute(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        try:
            dir_path = (self.workspace_root / path).resolve()

            if not str(dir_path).startswith(str(self.workspace_root)):
                return {"error": f"Access denied: {path} is outside workspace"}

            if not dir_path.is_dir():
                return {"error": f"Not a directory: {path}"}

            items = []
            for item in sorted(dir_path.iterdir()):
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })

            return {"items": items, "count": len(items)}
        except Exception as e:
            return {"error": str(e)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_directory",
            description="List directory contents within workspace",
            input_schema={"path": {"type": "string", "optional": True}},
            output_schema={"type": "object"},
            permissions=["read_file"],
        )


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = workspace_root
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in tools."""
        self.register("calculator", Calculator())
        self.register("read_file", FileReadTool(self.workspace_root))
        self.register("write_file", FileWriteTool(self.workspace_root))
        self.register("list_directory", DirectoryListTool(self.workspace_root))

    def register(self, name: str, tool: Tool):
        """Register a tool."""
        self.tools[name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool."""
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool not found: {name}"}

        try:
            result = tool.execute(**kwargs)
            return result
        except Exception as e:
            return {"error": str(e)}

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

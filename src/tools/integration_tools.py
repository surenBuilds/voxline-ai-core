"""Integration tools — GitHub, Vercel, and workspace tools for ToolRegistry.

All tools follow the three-phase security model:
    validate_request → authorize_request → execute

Tools never bypass the security layer.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.integrations.github.service import GitHubService, GitHubServiceError
from src.integrations.github.security import GitHubOperation
from src.integrations.vercel.service import VercelService, VercelServiceError
from src.integrations.vercel.security import VercelOperation
from src.tools.security import (
    AuditLog,
    PathSecurity,
    PermissionDecision,
    ToolPermissionResult,
    ToolSecurityProfile,
)
from src.tools.tools import Tool, ToolSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repository workspace
# ---------------------------------------------------------------------------


class RepositoryWorkspace:
    """Manages a local clone of a GitHub repository.

    Ensures all operations stay within the configured workspace root.
    Uses git commands via subprocess (shell=False) with security validation.
    """

    _BRANCH_RE = re.compile(r"^[a-zA-Z0-9._/\-]+$")

    def __init__(
        self,
        workspace_root: str,
        owner: str,
        repo: str,
        branch: str = "main",
        audit_log: Optional[AuditLog] = None,
    ):
        self.workspace_root = workspace_root
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.repo_dir = f"{workspace_root}/{owner}__{repo}"
        self._path_security = PathSecurity(workspace_root)
        self._audit_log = audit_log

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_repo_path(self) -> str:
        return self.repo_dir

    @staticmethod
    def branch_name_is_valid(branch: str) -> bool:
        """Return *True* if *branch* contains only safe characters."""
        if not branch:
            return False
        if branch.startswith(".") or branch.startswith("/"):
            return False
        if branch.endswith(".") or branch.endswith("/"):
            return False
        return bool(RepositoryWorkspace._BRANCH_RE.match(branch))

    def _validate_repo_dir(self) -> None:
        """Ensure repo_dir is inside workspace_root."""
        self._path_security.validate_path(self.repo_dir)

    def _run_git(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command with security checks and optional audit logging."""
        self._validate_repo_dir()
        effective_cwd = cwd or self.repo_dir
        cmd = ["git"] + args
        if self._audit_log is not None:
            self._audit_log.log_event(
                "git_command",
                {"command": cmd, "cwd": effective_cwd},
            )
        return subprocess.run(
            cmd,
            cwd=effective_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

    # ------------------------------------------------------------------
    # Git operations
    # ------------------------------------------------------------------

    def clone(self, clone_url: str) -> Dict[str, Any]:
        self._validate_repo_dir()
        cmd = ["git", "clone", clone_url, self.repo_dir]
        if self._audit_log is not None:
            self._audit_log.log_event("git_command", {"command": cmd, "cwd": None})
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, shell=False,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else "",
        }

    def checkout(self, branch: str) -> Dict[str, Any]:
        result = self._run_git(["checkout", branch])
        self.branch = branch
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else "",
        }

    def create_branch(self, branch: str) -> Dict[str, Any]:
        result = self._run_git(["checkout", "-b", branch])
        self.branch = branch
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else "",
        }

    def diff(self) -> str:
        result = self._run_git(["diff"])
        return result.stdout

    def status(self) -> str:
        result = self._run_git(["status", "--short"])
        return result.stdout

    def commit(self, message: str) -> Dict[str, Any]:
        self._run_git(["add", "-A"])
        result = self._run_git(["commit", "-m", message])
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else "",
        }

    def push(self, branch: str) -> Dict[str, Any]:
        result = self._run_git(["push", "origin", branch], timeout=60)
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else "",
        }

    def run_tests(self, test_command: str = "pytest") -> Dict[str, Any]:
        self._validate_repo_dir()
        try:
            parts = shlex.split(test_command, posix=(os.name != "nt"))
            result = subprocess.run(
                parts,
                cwd=self.repo_dir,
                capture_output=True, text=True, timeout=120, shell=False,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
                "error": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
                "returncode": result.returncode,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": f"Command not found: {test_command}",
                "returncode": -1,
            }


# ---------------------------------------------------------------------------
# GitHub tools
# ---------------------------------------------------------------------------


class GitHubReadRepositoryTool(Tool):
    def __init__(self, service: GitHubService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        try:
            result = self._service.get_repository(owner, repo)
            return {
                "owner": result.owner,
                "name": result.name,
                "full_name": result.full_name,
                "default_branch": result.default_branch,
                "description": result.description,
                "private": result.private,
                "stars": result.stars,
                "open_issues": result.open_issues,
            }
        except GitHubServiceError as exc:
            return {"error": str(exc)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="github_read_repository",
            description="Read GitHub repository metadata",
            input_schema={"owner": {"type": "string"}, "repo": {"type": "string"}},
            output_schema={"type": "object"},
            permissions=["github_read"],
        )


class GitHubReadFileTool(Tool):
    def __init__(self, service: GitHubService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(self, owner: str = "", repo: str = "", path: str = "", ref: str = "") -> Dict[str, Any]:
        try:
            result = self._service.get_file(owner, repo, path, ref)
            return {"path": result.path, "content": result.content, "sha": result.sha, "size": result.size}
        except GitHubServiceError as exc:
            return {"error": str(exc)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="github_read_file",
            description="Read a file from a GitHub repository",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "path": {"type": "string"}, "ref": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["github_read"],
        )


class GitHubCreateBranchTool(Tool):
    def __init__(self, service: GitHubService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(self, owner: str = "", repo: str = "", branch: str = "", from_sha: str = "") -> Dict[str, Any]:
        try:
            result = self._service.create_branch(owner, repo, branch, from_sha)
            return {"name": result.name, "sha": result.sha}
        except GitHubServiceError as exc:
            return {"error": str(exc)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="github_create_branch",
            description="Create a new branch in a GitHub repository",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "branch": {"type": "string"}, "from_sha": {"type": "string"},
            },
            output_schema={"type": "object"},
            permissions=["github_write"],
        )


class GitHubCommitTool(Tool):
    def __init__(self, service: GitHubService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(
        self, owner: str = "", repo: str = "", path: str = "",
        content: str = "", message: str = "", sha: str = "", branch: str = "",
    ) -> Dict[str, Any]:
        try:
            result = self._service.update_file(owner, repo, path, content, message, sha, branch)
            commit = result.get("commit", {})
            return {
                "success": True,
                "commit_sha": commit.get("sha", ""),
                "path": path,
            }
        except GitHubServiceError as exc:
            return {"error": str(exc)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="github_commit",
            description="Create or update a file in a GitHub repository",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "path": {"type": "string"}, "content": {"type": "string"},
                "message": {"type": "string"}, "sha": {"type": "string"},
                "branch": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["github_write"],
        )


class GitHubCreatePullRequestTool(Tool):
    def __init__(self, service: GitHubService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(
        self, owner: str = "", repo: str = "", title: str = "",
        head: str = "", base: str = "main", body: str = "",
    ) -> Dict[str, Any]:
        try:
            result = self._service.create_pull_request(owner, repo, title, head, base, body)
            return {
                "number": result.number,
                "title": result.title,
                "url": result.url,
                "state": result.state,
            }
        except GitHubServiceError as exc:
            return {"error": str(exc)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="github_create_pull_request",
            description="Create a pull request in a GitHub repository",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "title": {"type": "string"}, "head": {"type": "string"},
                "base": {"type": "string"}, "body": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["github_write"],
        )


class GitHubListIssuesTool(Tool):
    def __init__(self, service: GitHubService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(self, owner: str = "", repo: str = "", state: str = "open") -> List[Dict[str, Any]]:
        try:
            issues = self._service.list_issues(owner, repo, state)
            return [{"number": i.number, "title": i.title, "state": i.state} for i in issues]
        except GitHubServiceError as exc:
            return [{"error": str(exc)}]

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="github_list_issues",
            description="List issues in a GitHub repository",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "state": {"type": "string", "optional": True},
            },
            output_schema={"type": "array"},
            permissions=["github_read"],
        )


# ---------------------------------------------------------------------------
# Vercel tools
# ---------------------------------------------------------------------------


class VercelListProjectsTool(Tool):
    def __init__(self, service: VercelService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(self, per_page: int = 20) -> List[Dict[str, Any]]:
        try:
            projects = self._service.list_projects(per_page)
            return [{"id": p.id, "name": p.name, "framework": p.framework} for p in projects]
        except VercelServiceError as exc:
            return [{"error": str(exc)}]

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="vercel_list_projects",
            description="List Vercel projects",
            input_schema={"per_page": {"type": "integer", "optional": True}},
            output_schema={"type": "array"},
            permissions=["vercel_read"],
        )


class VercelCreateDeploymentTool(Tool):
    def __init__(self, service: VercelService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(
        self, project_id: str = "", name: str = "", target: str = "preview",
        git_branch: str = "", git_repo: str = "",
    ) -> Dict[str, Any]:
        try:
            git_source = None
            if git_repo and git_branch:
                git_source = {"type": "github", "ref": git_branch, "repoId": git_repo}

            if target == "production":
                result = self._service.create_production_deployment(project_id, name, git_source)
            else:
                result = self._service.create_preview_deployment(project_id, name, git_source)
            return {
                "id": result.id,
                "url": result.url,
                "state": result.state.value,
                "environment": result.environment.value,
            }
        except VercelServiceError as exc:
            return {"error": str(exc)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="vercel_create_deployment",
            description="Create a Vercel deployment (preview or production)",
            input_schema={
                "project_id": {"type": "string"}, "name": {"type": "string"},
                "target": {"type": "string", "optional": True},
                "git_branch": {"type": "string", "optional": True},
                "git_repo": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["vercel_write"],
        )


class VercelGetDeploymentTool(Tool):
    def __init__(self, service: VercelService):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=False,
        ))
        self._service = service

    def execute(self, deployment_id: str = "") -> Dict[str, Any]:
        try:
            result = self._service.get_deployment(deployment_id)
            return {
                "id": result.id,
                "url": result.url,
                "state": result.state.value,
                "environment": result.environment.value,
                "branch": result.branch,
            }
        except VercelServiceError as exc:
            return {"error": str(exc)}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="vercel_get_deployment",
            description="Get Vercel deployment status",
            input_schema={"deployment_id": {"type": "string"}},
            output_schema={"type": "object"},
            permissions=["vercel_read"],
        )


# ---------------------------------------------------------------------------
# Workspace tools
# ---------------------------------------------------------------------------


class WorkspaceCloneTool(Tool):
    def __init__(self, workspace_root: str, path_security: PathSecurity):
        super().__init__(ToolSecurityProfile(
            network_access=True, filesystem_read=False, filesystem_write=True,
        ))
        self._workspace_root = workspace_root
        self._path_security = path_security

    def execute(self, owner: str = "", repo: str = "", clone_url: str = "", branch: str = "main") -> Dict[str, Any]:
        ws = RepositoryWorkspace(self._workspace_root, owner, repo, branch)
        result = ws.clone(clone_url)
        result["repo_dir"] = ws.repo_dir
        return result

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="workspace_clone",
            description="Clone a GitHub repository into the workspace",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "clone_url": {"type": "string"}, "branch": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["workspace_write"],
        )


class WorkspaceDiffTool(Tool):
    def __init__(self, workspace_root: str):
        super().__init__(ToolSecurityProfile(
            filesystem_read=True, filesystem_write=False, command_execution=True,
        ))
        self._workspace_root = workspace_root

    def execute(self, owner: str = "", repo: str = "", branch: str = "main") -> Dict[str, Any]:
        ws = RepositoryWorkspace(self._workspace_root, owner, repo, branch)
        return {"diff": ws.diff(), "status": ws.status()}

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="workspace_diff",
            description="Show git diff and status for a repository workspace",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "branch": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["workspace_read"],
        )


class WorkspaceTestTool(Tool):
    def __init__(self, workspace_root: str):
        super().__init__(ToolSecurityProfile(
            filesystem_read=True, filesystem_write=False, command_execution=True,
        ))
        self._workspace_root = workspace_root

    def execute(self, owner: str = "", repo: str = "", branch: str = "main", test_command: str = "pytest") -> Dict[str, Any]:
        ws = RepositoryWorkspace(self._workspace_root, owner, repo, branch)
        return ws.run_tests(test_command)

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="workspace_test",
            description="Run tests in a repository workspace",
            input_schema={
                "owner": {"type": "string"}, "repo": {"type": "string"},
                "branch": {"type": "string", "optional": True},
                "test_command": {"type": "string", "optional": True},
            },
            output_schema={"type": "object"},
            permissions=["workspace_command"],
        )

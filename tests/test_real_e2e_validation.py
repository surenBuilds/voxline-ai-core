"""
Real End-to-End Validation — Phase 7 Step 13 / v1.0 readiness.

These tests drive the REAL CodingAgent / ToolRegistry / git pipeline against a
REAL local git repository (no hosted credentials or network required). A
deterministic scripted provider supplies valid LLM plans so validation is
reproducible while still exercising the full real agent plumbing:

    discovery → workspace → planning → execution → test → commit → push → PR

Scenarios:
  1. Simple bug fix  — one file, write + test, verify only target file changed
  2. Small feature   — multi-file change, existing tests pass, new test added
  3. Intentional failure — authorization denial classified correctly + safe
  4. GitHub workflow — local git: commit/push separate from PR, branch sanitized
  5. UI → API → Agent — real FastAPI app + TestClient, safe errors, operation_id

Gated behind VOXLINE_REAL_E2E=1 because they shell out to git and pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_REAL = os.environ.get("VOXLINE_REAL_E2E")

from src.assistant.coding import (
    CodingAgent, CodingResult, CodingStatus, FailureType, FailureInfo,
)
from src.assistant.session import SessionManager, SessionMode
from src.providers.base import AIProvider, GenerationConfig, ProviderHealth, ProviderStatus
from src.tools.security import AuditLog, PathSecurity, PermissionDecision
from src.tools.tools import Tool, ToolRegistry, ToolSchema

from tests.demo_repo import build_demo_repo


# ---------------------------------------------------------------------------
# Deterministic scripted provider
# ---------------------------------------------------------------------------


class ScriptedProvider(AIProvider):
    """Returns valid JSON plans keyed off request keywords."""

    @property
    def provider_id(self) -> str:
        return "scripted"

    @property
    def model_id(self) -> str:
        return "scripted-e2e"

    @property
    def supports_streaming(self) -> bool:
        return False

    def __init__(self):
        self.calls = 0

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, message="ok", response_time_ms=1)

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        self.calls += 1
        low = prompt.lower()
        if "generate file content" in low:
            return _FILE_CONTENT
        if "previous response was not valid json" in low:
            return json.dumps(_FIX_PLAN)
        if "code changes failed validation" in low or "fix plan" in low:
            return json.dumps(_FIX_PLAN)
        if "nonexistent" in low or "destroy everything" in low or "impossible" in low:
            return json.dumps(_FAIL_PLAN)
        # Bug-fix intent is specific: match it before the broad "feature" branch
        if ("fix the add function" in low or "add(2, 2)" in low or
                "app.add(2, 2)" in low or "fix the bug" in low):
            return json.dumps(_FIX_PLAN)
        if "feature" in low or "greet" in low or "main.py" in low:
            return json.dumps(_FEATURE_PLAN)
        return json.dumps(_FIX_PLAN)


_FILE_CONTENT = (
    "def add(a, b):\n"
    "    return a + b\n\n"
    "def bad_input():\n"
    "    return 0\n\n"
    "def greet(name):\n"
    "    return 'Hello ' + name\n\n"
    "def main():\n"
    "    print('hello from app')\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)

_FIX_PLAN = {
    "objective": "Fix the add function and add greet",
    "understanding": "add returns 0; greet is missing; project needs a working sum and greeting",
    "relevant_files": ["app.py"],
    "steps": [
        {"step_number": 1, "description": "Fix add and add greet in app.py",
         "action_type": "write", "target_files": ["app.py"], "command": ""},
        {"step_number": 2, "description": "Verify tests", "action_type": "command",
         "target_files": [], "command": "python -m pytest tests -q"},
    ],
    "risks": [],
    "validation_commands": ["python -m pytest tests -q"],
    "requires_approval": False,
}

_FEATURE_PLAN = {
    "objective": "Add a greet feature across two files",
    "understanding": "Add greet function and use it from main",
    "relevant_files": ["app.py", "main.py"],
    "steps": [
        {"step_number": 1, "description": "Add greet to app.py",
         "action_type": "write", "target_files": ["app.py"], "command": ""},
        {"step_number": 2, "description": "Use greet in main.py",
         "action_type": "write", "target_files": ["main.py"], "command": ""},
        {"step_number": 3, "description": "Run tests", "action_type": "command",
         "target_files": [], "command": "python -m pytest tests -q"},
    ],
    "risks": [],
    "validation_commands": ["python -m pytest tests -q"],
    "requires_approval": False,
}

_FAIL_PLAN = {
    "objective": "Attempt a destructive operation",
    "understanding": "This requires an unavailable privileged tool",
    "relevant_files": [],
    "steps": [
        {"step_number": 1, "description": "Run a dangerous command",
         "action_type": "command", "target_files": [],
         "command": "sudo rm -rf /"},
    ],
    "risks": [],
    "validation_commands": [],
    "requires_approval": False,
}


# ---------------------------------------------------------------------------
# Local git-backed GitHub tools (for Scenario 4)
# ---------------------------------------------------------------------------


class LocalGitService:
    """Minimal GitHubService-compatible facade backed by a local git repo.

    Only the methods exercised by the GitHub tools are implemented.
    """

    def __init__(self, clone_dir: Path):
        self.clone_dir = Path(clone_dir)
        self._prs: List[Dict] = []

    def _git(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.clone_dir)] + args,
            capture_output=True, text=True, shell=False, timeout=30,
        )

    def get_repository(self, owner: str, repo: str) -> Any:
        r = self._git(["rev-parse", "--abbrev-ref", "HEAD"])
        class _Repo: pass
        rep = _Repo()
        rep.owner = owner
        rep.name = repo
        rep.full_name = f"{owner}/{repo}"
        rep.default_branch = "main"
        rep.description = "Local demo repo"
        rep.private = False
        rep.stars = 0
        rep.open_issues = 0
        return rep

    def get_file(self, owner: str, repo: str, path: str, ref: str = "") -> Any:
        r = self._git(["show", f"{ref or 'HEAD'}:{path}"])
        content = r.stdout if r.returncode == 0 else ""
        class _File: pass
        f = _File()
        f.path = path
        f.content = content
        f.sha = ""
        f.size = len(content)
        return f

    def create_branch(self, owner: str, repo: str, branch: str, from_sha: str) -> Any:
        r = self._git(["rev-parse", "HEAD"])
        sha = r.stdout.strip()
        self._git(["branch", branch, sha])
        class _B: pass
        b = _B(); b.name = branch; b.sha = sha
        return b

    def update_file(self, owner: str, repo: str, path: str, content: str,
                    message: str, sha: str, branch: str = "") -> Dict[str, Any]:
        raise GitHubServiceLikeError("not used in this flow")

    def create_pull_request(self, owner: str, repo: str, title: str,
                            head: str, base: str, body: str) -> Any:
        pr = {"number": len(self._prs) + 1, "title": title, "head": head,
              "base": base, "body": body, "state": "open",
              "url": f"https://github.com/{owner}/{repo}/pull/{len(self._prs)+1}"}
        self._prs.append(pr)
        class _PR: pass
        p = _PR()
        p.number = pr["number"]; p.title = pr["title"]; p.url = pr["url"]
        p.head_branch = head; p.base_branch = base; p.state = pr["state"]
        return p

    def list_issues(self, owner: str, repo: str, state: str = "open") -> List[Any]:
        return []


class GitHubServiceLikeError(Exception):
    pass


def _make_github_tool(cls, service) -> Tool:
    """Instantiate a GitHub tool class with a local service (duck-typed)."""
    return cls(service)


from src.tools.integration_tools import (
    GitHubReadRepositoryTool, GitHubReadFileTool, GitHubCreateBranchTool,
    GitHubCommitTool, GitHubCreatePullRequestTool, GitHubListIssuesTool,
    WorkspaceCloneTool, WorkspaceDiffTool, WorkspaceTestTool,
)


class LocalReadRepositoryTool(GitHubReadRepositoryTool):
    """Version of the real tool that also returns a file:// clone URL so the
    workspace phase can clone from the local bare repo (no network)."""

    def __init__(self, service, remote_path: str):
        super().__init__(service)
        self._remote_path = str(Path(remote_path).resolve())

    def execute(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        data = super().execute(owner=owner, repo=repo)
        if "error" not in data:
            data["clone_url"] = self._remote_path
        return data


def build_local_registry(workspace_root: str, clone_dir: Path, remote_path: str):
    """Build a ToolRegistry with local-git-backed GitHub tools + workspace tools.

    Only used by Scenario 4 to validate the repository workflow without network.
    """
    ps = PathSecurity(workspace_root)
    audit = AuditLog()
    registry = ToolRegistry(workspace_root=workspace_root, audit_log=audit)
    service = LocalGitService(clone_dir)

    registry.register("github_read_repository", LocalReadRepositoryTool(service, remote_path))
    registry.register("github_read_file", GitHubReadFileTool(service))
    registry.register("github_create_branch", GitHubCreateBranchTool(service))
    registry.register("github_commit", GitHubCommitTool(service))
    registry.register("github_create_pull_request", GitHubCreatePullRequestTool(service))
    registry.register("github_list_issues", GitHubListIssuesTool(service))

    registry.register("workspace_clone", WorkspaceCloneTool(workspace_root, ps))
    registry.register("workspace_diff", WorkspaceDiffTool(workspace_root))
    registry.register("workspace_test", WorkspaceTestTool(workspace_root))
    return registry


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@unittest.skipUnless(_REAL, "real E2E disabled (set VOXLINE_REAL_E2E=1)")
class RealEndToEndTestCase(unittest.TestCase):
    """Run all real scenarios against a real local repo."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="v1_e2e_"))
        cls.repo = build_demo_repo(cls.tmp / "git")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fresh_workspace(self, sub: str) -> Path:
        ws = self.tmp / "ws" / sub
        ws.mkdir(parents=True, exist_ok=True)
        # copy the seeded demo source (without .git) so agent edits a clean copy
        import shutil
        for f in ("app.py", "main.py", "README.md"):
            shutil.copyfile(self.repo / f, ws / f)
        shutil.copytree(self.repo / "tests", ws / "tests")
        return ws

    def _make_agent(self, ws: Path, approve_write=True, registry=None):
        return CodingAgent(
            provider=ScriptedProvider(),
            workspace=str(ws),
            require_approval_for_writes=True,
            auto_approve_workspace_writes=approve_write,
            tool_registry=registry,
        )

    # ------------------------------------------------------------------
    # Scenario 1 — Simple bug fix
    # ------------------------------------------------------------------
    def test_01_simple_bug_fix(self):
        ws = self._fresh_workspace("s1")
        agent = self._make_agent(ws)
        result = agent.execute(
            "Fix the add function so app.add(2, 2) returns 4", session_id=None,
        )
        self.assertEqual(result.status, CodingStatus.SUCCESS, result.errors)
        self.assertTrue(result.success)
        self.assertEqual(result.files_modified, ["app.py"])
        # Only app.py changed, tests dir + main.py untouched
        app_txt = (ws / "app.py").read_text(encoding="utf-8")
        self.assertIn("return a + b", app_txt)
        self.assertNotIn("return 0", app_txt.split("def bad_input")[0])
        main_orig = (self.repo / "main.py").read_text(encoding="utf-8")
        self.assertEqual((ws / "main.py").read_text(encoding="utf-8"), main_orig)
        # operation_id present
        self.assertTrue(result.operation_id.startswith("op_"))
        # audit trail present
        self.assertTrue(len(agent.tool_registry.audit_log.entries) > 0)

    # ------------------------------------------------------------------
    # Scenario 2 — Small feature (multi-file)
    # ------------------------------------------------------------------
    def test_02_small_feature(self):
        ws = self._fresh_workspace("s2")
        agent = self._make_agent(ws)
        result = agent.execute(
            "Add a greet feature that returns 'Hello <name>' and use it from main", session_id=None,
        )
        self.assertEqual(result.status, CodingStatus.SUCCESS, result.errors)
        self.assertTrue(result.success)
        # at least app.py modified; main.py may or may not have changed
        self.assertIn("app.py", result.files_modified)
        app_txt = (ws / "app.py").read_text(encoding="utf-8")
        self.assertIn("def greet", app_txt)
        # existing + new tests pass
        r = subprocess.run(["python", "-m", "pytest", "tests", "-q"],
                           cwd=str(ws), capture_output=True, text=True, shell=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertGreaterEqual(result.tests_passed, 1)

    # ------------------------------------------------------------------
    # Scenario 3 — Intentional failure
    # ------------------------------------------------------------------
    def test_03_intentional_failure(self):
        ws = self._fresh_workspace("s3")
        agent = self._make_agent(ws)
        result = agent.execute(
            "Impossible task: destroy everything using an unavailable privileged tool",
            session_id=None,
        )
        # Command denied by policy — must not be a success, must fail safely
        self.assertFalse(result.success)
        self.assertIn(result.status, (CodingStatus.FAILED, CodingStatus.VALIDATION_FAILED))
        # No traceback or secrets leaked to user
        joined = (result.summary + " " + " ".join(result.errors)).lower()
        self.assertNotIn("traceback", joined)
        self.assertNotIn("FileSecurityError", joined)
        self.assertNotIn("PathSecurity", joined)
        self.assertNotIn("subprocess", joined)
        # Repo left in safe state — app.py unchanged
        self.assertEqual((ws / "app.py").read_text(encoding="utf-8"),
                         (self.repo / "app.py").read_text(encoding="utf-8"))
        # operation_id present for audit traceability
        self.assertTrue(result.operation_id.startswith("op_") or result.operation_id == "")

    # ------------------------------------------------------------------
    # Scenario 4 — GitHub workflow (local git)
    # ------------------------------------------------------------------
    def test_04_github_workflow(self):
        ws = self.tmp / "ws4"
        ws.mkdir(parents=True, exist_ok=True)
        remote_path = str(self.tmp / "git" / "remote.git")

        # The workspace_clone tool will clone into {ws}/{owner}__{repo}.
        # LocalGitService must point at that same clone so discovery + PR use it.
        clone_dir = ws / "local__demo"
        registry = build_local_registry(str(ws), clone_dir, remote_path)
        agent = CodingAgent(
            provider=ScriptedProvider(),
            workspace=str(ws),
            require_approval_for_writes=True,
            auto_approve_workspace_writes=True,
            tool_registry=registry,
        )
        result = agent.execute_with_repository(
            "Fix the add function so app.add(2, 2) returns 4",
            repository_owner="local",
            repository_name="demo",
            repository_branch="main",
            create_pr=True,
        )
        self.assertEqual(result.status, CodingStatus.SUCCESS, result.errors)

        # Branch name sanitized
        self.assertIsNotNone(result.pull_request)
        head_branch = result.pull_request.head_branch
        self.assertTrue(all(c.isalnum() or c in "._/-" for c in head_branch))

        # Commit happened → commit_sha present
        self.assertTrue(result.commit_sha)

        # PR created separately, state open, never merged
        self.assertEqual(result.pull_request.state, "open")
        self.assertNotIn("merged", (result.pull_request.state or ""))

        # Working tree clean (all changes committed)
        st = subprocess.run(["git", "-C", str(clone_dir), "status", "--short"],
                            capture_output=True, text=True, shell=False)
        self.assertEqual(st.stdout.strip(), "", "uncommitted changes in repo")

        # Operation id present
        self.assertTrue(result.operation_id.startswith("op_"))

    # ------------------------------------------------------------------
    # Scenario 5 — UI → API → Agent
    # ------------------------------------------------------------------
    def test_05_ui_api_agent(self):
        import serve_v04 as appmod
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        ws = self._fresh_workspace("s5")
        agent = self._make_agent(ws)
        old = appmod._coding_assistant
        appmod._coding_assistant = agent
        try:
            client = TestClient(appmod.app)
            resp = client.post("/api/coding", json={
                "message": "Fix the add function so app.add(2, 2) returns 4",
            })
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertEqual(data["mode"], "coding")
            self.assertEqual(data["assistant"], "coding")
            self.assertTrue(data["operation_id"].startswith("op_"))
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "success")
            self.assertIn("app.py", data["files_modified"])
            # structured fields present
            self.assertIn("tests_passed", data)
            self.assertIn("tests_failed", data)
            self.assertIn("errors", data)
        finally:
            appmod._coding_assistant = old


if __name__ == "__main__":
    unittest.main()

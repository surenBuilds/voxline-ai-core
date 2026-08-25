"""
Tests for GitHub integration.

Covers:
  MODELS (tests 1-5)
  CREDENTIALS (tests 6-8)
  PERMISSION POLICY (tests 9-14)
  CLIENT (tests 15-19)
  SERVICE (tests 20-26)
  TOOLS (tests 27-31)
  REPOSITORY WORKSPACE (tests 32-35)
  CODING AGENT INTEGRATION (tests 36-37)
  SECURITY (tests 38-40)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integrations.credentials import EnvironmentCredentialProvider
from src.integrations.github.models import (
    GitHubBranch,
    GitHubCommit,
    GitHubFile,
    GitHubIssue,
    GitHubPermission,
    GitHubPullRequest,
    GitHubRepository,
    GitHubWorkflowRun,
)
from src.integrations.github.client import GitHubClient, GitHubClientError
from src.integrations.github.security import GitHubOperation, GitHubPermissionPolicy
from src.integrations.github.service import GitHubService, GitHubServiceError
from src.tools.integration_tools import (
    GitHubReadRepositoryTool,
    GitHubReadFileTool,
    GitHubCreateBranchTool,
    GitHubCommitTool,
    GitHubCreatePullRequestTool,
    GitHubListIssuesTool,
    RepositoryWorkspace,
)
from src.tools.security import PathSecurity, AuditLog


# ---------------------------------------------------------------------------
# Mock GitHub API responses
# ---------------------------------------------------------------------------

MOCK_REPO_DATA = {
    "name": "test-repo",
    "full_name": "testowner/test-repo",
    "owner": {"login": "testowner"},
    "default_branch": "main",
    "description": "A test repository",
    "private": False,
    "html_url": "https://github.com/testowner/test-repo",
    "clone_url": "https://github.com/testowner/test-repo.git",
    "ssh_url": "git@github.com:testowner/test-repo.git",
    "language": "Python",
    "stargazers_count": 42,
    "open_issues_count": 3,
}

MOCK_BRANCH_DATA = {
    "name": "main",
    "commit": {"sha": "abc123"},
    "protected": False,
}

MOCK_PR_DATA = {
    "number": 1,
    "title": "Fix authentication bug",
    "state": "open",
    "head": {"ref": "fix-auth"},
    "base": {"ref": "main"},
    "html_url": "https://github.com/testowner/test-repo/pull/1",
    "body": "Fixed the auth bug",
    "user": {"login": "testowner"},
    "mergeable": True,
}

MOCK_ISSUE_DATA = {
    "number": 42,
    "title": "Bug in login",
    "state": "open",
    "html_url": "https://github.com/testowner/test-repo/issues/42",
    "body": "Login is broken",
    "user": {"login": "reporter"},
    "labels": [{"name": "bug"}],
}

MOCK_COMMIT_DATA = {
    "sha": "def456",
    "commit": {
        "message": "Fix auth",
        "author": {"name": "dev", "date": "2026-01-01"},
    },
    "html_url": "https://github.com/testowner/test-repo/commit/def456",
}

MOCK_FILE_DATA = {
    "path": "src/auth.py",
    "content": "def login(): pass",
    "sha": "filesha123",
    "size": 100,
    "encoding": "utf-8",
}


# =========================================================================
# MODELS (tests 1-5)
# =========================================================================


class TestGitHubModels(unittest.TestCase):
    """Tests 1-5: GitHub model creation and parsing."""

    def test_01_repository_from_api(self):
        repo = GitHubRepository.from_api(MOCK_REPO_DATA)
        self.assertEqual(repo.owner, "testowner")
        self.assertEqual(repo.name, "test-repo")
        self.assertEqual(repo.full_name, "testowner/test-repo")
        self.assertEqual(repo.default_branch, "main")
        self.assertFalse(repo.private)
        self.assertEqual(repo.stars, 42)

    def test_02_branch_from_api(self):
        branch = GitHubBranch.from_api(MOCK_BRANCH_DATA)
        self.assertEqual(branch.name, "main")
        self.assertEqual(branch.sha, "abc123")
        self.assertFalse(branch.protected)

    def test_03_pull_request_from_api(self):
        pr = GitHubPullRequest.from_api(MOCK_PR_DATA)
        self.assertEqual(pr.number, 1)
        self.assertEqual(pr.title, "Fix authentication bug")
        self.assertEqual(pr.state, "open")
        self.assertEqual(pr.head_branch, "fix-auth")
        self.assertEqual(pr.base_branch, "main")
        self.assertEqual(pr.author, "testowner")

    def test_04_issue_from_api(self):
        issue = GitHubIssue.from_api(MOCK_ISSUE_DATA)
        self.assertEqual(issue.number, 42)
        self.assertEqual(issue.title, "Bug in login")
        self.assertEqual(issue.state, "open")
        self.assertIn("bug", issue.labels)

    def test_05_commit_from_api(self):
        commit = GitHubCommit.from_api(MOCK_COMMIT_DATA)
        self.assertEqual(commit.sha, "def456")
        self.assertEqual(commit.message, "Fix auth")
        self.assertEqual(commit.author, "dev")


# =========================================================================
# CREDENTIALS (tests 6-8)
# =========================================================================


class TestCredentials(unittest.TestCase):
    """Tests 6-8: Credential provider behavior."""

    def test_06_credential_provider_availability(self):
        provider = EnvironmentCredentialProvider()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123456789"}):
            self.assertTrue(provider.is_available("github"))

    def test_07_credential_provider_missing(self):
        provider = EnvironmentCredentialProvider()
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(provider.is_available("github"))

    def test_08_credential_redaction(self):
        provider = EnvironmentCredentialProvider()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_secrettoken123"}):
            redacted = provider.redact("Token is ghp_secrettoken123 here")
            self.assertNotIn("ghp_secrettoken123", redacted)
            self.assertIn("***", redacted)


# =========================================================================
# PERMISSION POLICY (tests 9-14)
# =========================================================================


class TestGitHubPermissionPolicy(unittest.TestCase):
    """Tests 9-14: GitHub permission policy."""

    def test_09_read_allowed(self):
        policy = GitHubPermissionPolicy()
        result = policy.check(GitHubOperation.GET_REPOSITORY)
        self.assertTrue(result.allowed)
        self.assertFalse(result.requires_approval)

    def test_10_write_requires_approval(self):
        policy = GitHubPermissionPolicy()
        result = policy.check(GitHubOperation.CREATE_BRANCH)
        self.assertTrue(result.allowed)
        self.assertTrue(result.requires_approval)

    def test_11_destructive_requires_approval(self):
        policy = GitHubPermissionPolicy()
        result = policy.check(GitHubOperation.MERGE_PULL_REQUEST)
        self.assertTrue(result.allowed)
        self.assertTrue(result.requires_approval)

    def test_12_repository_denied(self):
        policy = GitHubPermissionPolicy(denied_repositories={"bad/repo"})
        result = policy.check(GitHubOperation.GET_REPOSITORY, "bad/repo")
        self.assertFalse(result.allowed)

    def test_13_repository_not_in_allowed(self):
        policy = GitHubPermissionPolicy(allowed_repositories={"good/repo"})
        result = policy.check(GitHubOperation.GET_REPOSITORY, "other/repo")
        self.assertFalse(result.allowed)

    def test_14_repository_in_allowed(self):
        policy = GitHubPermissionPolicy(allowed_repositories={"good/repo"})
        result = policy.check(GitHubOperation.GET_REPOSITORY, "good/repo")
        self.assertTrue(result.allowed)


# =========================================================================
# CLIENT (tests 15-19)
# =========================================================================


class TestGitHubClient(unittest.TestCase):
    """Tests 15-19: GitHub client (mocked HTTP)."""

    def test_15_client_requires_token(self):
        with self.assertRaises(ValueError):
            GitHubClient("")

    def test_16_client_get_repository(self):
        client = GitHubClient("ghp_fake")
        with patch("urllib.request.urlopen") as mock_open:
            resp = MagicMock()
            resp.read.return_value = json.dumps(MOCK_REPO_DATA).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = resp
            repo = client.get_repository("testowner", "test-repo")
            self.assertEqual(repo.owner, "testowner")
            self.assertEqual(repo.name, "test-repo")

    def test_17_client_list_branches(self):
        client = GitHubClient("ghp_fake")
        with patch("urllib.request.urlopen") as mock_open:
            resp = MagicMock()
            resp.read.return_value = json.dumps([MOCK_BRANCH_DATA]).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = resp
            branches = client.list_branches("testowner", "test-repo")
            self.assertEqual(len(branches), 1)
            self.assertEqual(branches[0].name, "main")

    def test_18_client_create_pull_request(self):
        client = GitHubClient("ghp_fake")
        with patch("urllib.request.urlopen") as mock_open:
            resp = MagicMock()
            resp.read.return_value = json.dumps(MOCK_PR_DATA).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = resp
            pr = client.create_pull_request(
                "testowner", "test-repo", "Fix auth", "fix-auth", "main", "Fixed"
            )
            self.assertEqual(pr.number, 1)

    def test_19_client_handles_http_error(self):
        client = GitHubClient("ghp_fake")
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = GitHubClientError("Not found", status_code=404)
            with self.assertRaises(GitHubClientError):
                client.get_repository("testowner", "nonexistent")


# =========================================================================
# SERVICE (tests 20-26)
# =========================================================================


class TestGitHubService(unittest.TestCase):
    """Tests 20-26: GitHub service with mocked client."""

    def _make_service(self, **policy_kwargs):
        from src.integrations.credentials import CredentialProvider

        class MockCreds(CredentialProvider):
            def get_token(self, service):
                return "ghp_fake"
            def is_available(self, service):
                return True
            def redact(self, text):
                return text

        return GitHubService(MockCreds(), GitHubPermissionPolicy(**policy_kwargs))

    def test_20_service_get_repository(self):
        service = self._make_service()
        with patch.object(GitHubClient, "get_repository") as mock_get:
            mock_get.return_value = GitHubRepository(
                owner="testowner", name="test-repo",
                full_name="testowner/test-repo", default_branch="main",
            )
            service._client = GitHubClient("ghp_fake")
            repo = service.get_repository("testowner", "test-repo")
            self.assertEqual(repo.owner, "testowner")

    def test_21_service_denied_operation(self):
        service = self._make_service(denied_repositories={"bad/repo"})
        with self.assertRaises(GitHubServiceError):
            service.get_repository("bad", "repo")

    def test_22_service_is_authenticated(self):
        service = self._make_service()
        self.assertTrue(service.is_authenticated)

    def test_23_service_status(self):
        service = self._make_service()
        status = service.get_status()
        self.assertTrue(status["authenticated"])

    def test_24_service_create_pull_request(self):
        service = self._make_service()
        service._client = GitHubClient("ghp_fake")
        with patch.object(GitHubClient, "create_pull_request") as mock_create:
            mock_create.return_value = GitHubPullRequest(
                number=1, title="Fix", state="open",
                head_branch="fix", base_branch="main",
            )
            pr = service.create_pull_request(
                "testowner", "test-repo", "Fix", "fix", "main"
            )
            self.assertEqual(pr.number, 1)

    def test_25_service_unauthenticated(self):
        from src.integrations.credentials import CredentialProvider

        class NoCreds(CredentialProvider):
            def get_token(self, service):
                return None
            def is_available(self, service):
                return False
            def redact(self, text):
                return text

        service = GitHubService(NoCreds())
        with self.assertRaises(GitHubServiceError):
            service.get_repository("test", "repo")

    def test_26_service_list_issues(self):
        service = self._make_service()
        service._client = GitHubClient("ghp_fake")
        with patch.object(GitHubClient, "list_issues") as mock_list:
            mock_list.return_value = [GitHubIssue(
                number=1, title="Bug", state="open",
            )]
            issues = service.list_issues("testowner", "test-repo")
            self.assertEqual(len(issues), 1)


# =========================================================================
# TOOLS (tests 27-31)
# =========================================================================


class TestGitHubTools(unittest.TestCase):
    """Tests 27-31: GitHub tools for ToolRegistry."""

    def _make_service(self):
        from src.integrations.credentials import CredentialProvider

        class MockCreds(CredentialProvider):
            def get_token(self, service):
                return "ghp_fake"
            def is_available(self, service):
                return True
            def redact(self, text):
                return text

        return GitHubService(MockCreds())

    def test_27_read_repository_tool(self):
        service = self._make_service()
        service._client = GitHubClient("ghp_fake")
        tool = GitHubReadRepositoryTool(service)
        with patch.object(GitHubClient, "get_repository") as mock_get:
            mock_get.return_value = GitHubRepository(
                owner="testowner", name="test-repo",
                full_name="testowner/test-repo", default_branch="main",
            )
            result = tool.execute(owner="testowner", repo="test-repo")
            self.assertEqual(result["owner"], "testowner")

    def test_28_create_branch_tool(self):
        service = self._make_service()
        service._client = GitHubClient("ghp_fake")
        tool = GitHubCreateBranchTool(service)
        with patch.object(GitHubClient, "create_branch") as mock_create:
            mock_create.return_value = GitHubBranch(name="feature", sha="sha123")
            result = tool.execute(
                owner="testowner", repo="test-repo",
                branch="feature", from_sha="sha123",
            )
            self.assertEqual(result["name"], "feature")

    def test_29_create_pr_tool(self):
        service = self._make_service()
        service._client = GitHubClient("ghp_fake")
        tool = GitHubCreatePullRequestTool(service)
        with patch.object(GitHubClient, "create_pull_request") as mock_create:
            mock_create.return_value = GitHubPullRequest(
                number=1, title="Fix", state="open",
                head_branch="fix", base_branch="main",
            )
            result = tool.execute(
                owner="testowner", repo="test-repo",
                title="Fix", head="fix", base="main",
            )
            self.assertEqual(result["number"], 1)

    def test_30_list_issues_tool(self):
        service = self._make_service()
        service._client = GitHubClient("ghp_fake")
        tool = GitHubListIssuesTool(service)
        with patch.object(GitHubClient, "list_issues") as mock_list:
            mock_list.return_value = [GitHubIssue(number=1, title="Bug", state="open")]
            result = tool.execute(owner="testowner", repo="test-repo")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["title"], "Bug")

    def test_31_tool_error_handling(self):
        service = self._make_service()
        service._client = GitHubClient("ghp_fake")
        tool = GitHubReadRepositoryTool(service)
        with patch.object(GitHubClient, "get_repository") as mock_get:
            mock_get.side_effect = GitHubClientError("Not found", 404)
            result = tool.execute(owner="bad", repo="repo")
            self.assertIn("error", result)


# =========================================================================
# REPOSITORY WORKSPACE (tests 32-35)
# =========================================================================


class TestRepositoryWorkspace(unittest.TestCase):
    """Tests 32-35: Repository workspace operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_32_workspace_init(self):
        ws = RepositoryWorkspace(self.tmpdir, "owner", "repo")
        self.assertEqual(ws.owner, "owner")
        self.assertEqual(ws.repo, "repo")
        self.assertIn("owner__repo", ws.repo_dir)

    def test_33_workspace_diff_empty(self):
        ws = RepositoryWorkspace(self.tmpdir, "owner", "repo")
        ws.repo_dir = self.tmpdir
        diff = ws.diff()
        self.assertIsInstance(diff, str)

    def test_34_workspace_status_empty(self):
        ws = RepositoryWorkspace(self.tmpdir, "owner", "repo")
        ws.repo_dir = self.tmpdir
        status = ws.status()
        self.assertIsInstance(status, str)

    def test_35_workspace_run_tests_nonexistent(self):
        ws = RepositoryWorkspace(self.tmpdir, "owner", "repo")
        ws.repo_dir = self.tmpdir
        result = ws.run_tests("nonexistent_command")
        self.assertFalse(result["success"])


# =========================================================================
# CODING AGENT INTEGRATION (tests 36-37)
# =========================================================================


class TestCodingAgentIntegration(unittest.TestCase):
    """Tests 36-37: CodingAgent repository context."""

    def test_36_repository_context_on_task(self):
        from src.assistant.coding import RepositoryContext
        ctx = RepositoryContext(
            owner="testowner", name="test-repo",
            branch="main", default_branch="main",
        )
        self.assertEqual(ctx.owner, "testowner")
        self.assertEqual(ctx.name, "test-repo")

    def test_37_pull_request_info(self):
        from src.assistant.coding import PullRequestInfo
        pr = PullRequestInfo(
            number=1, title="Fix auth", url="https://github.com/test/repo/pull/1",
            head_branch="fix", base_branch="main", state="open",
        )
        self.assertEqual(pr.number, 1)
        self.assertEqual(pr.head_branch, "fix")


# =========================================================================
# SECURITY (tests 38-40)
# =========================================================================


class TestGitHubSecurity(unittest.TestCase):
    """Tests 38-40: Security requirements."""

    def test_38_tokens_never_in_logs(self):
        from src.integrations.credentials import EnvironmentCredentialProvider
        provider = EnvironmentCredentialProvider()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_SUPER_SECRET_12345"}):
            redacted = provider.redact("Connection with ghp_SUPER_SECRET_12345 to github.com")
            self.assertNotIn("ghp_SUPER_SECRET_12345", redacted)

    def test_39_client_never_logs_token(self):
        import logging
        with patch("src.integrations.github.client.logger") as mock_logger:
            client = GitHubClient("ghp_SECRET")
            mock_logger.warning.assert_not_called()

    def test_40_default_policy_blocks_destructive(self):
        policy = GitHubPermissionPolicy()
        result = policy.check(GitHubOperation.DELETE_BRANCH)
        self.assertTrue(result.requires_approval)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    unittest.main()

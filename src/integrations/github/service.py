"""GitHub service layer — business logic that coordinates client + security.

The service sits between the CodingAgent/tools and the raw GitHub client.
Every operation is permission-checked before execution.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.integrations.credentials import CredentialProvider
from src.integrations.github.client import GitHubClient, GitHubClientError
from src.integrations.github.models import (
    GitHubBranch,
    GitHubCommit,
    GitHubFile,
    GitHubIssue,
    GitHubPullRequest,
    GitHubRepository,
    GitHubWorkflowRun,
)
from src.integrations.github.security import GitHubOperation, GitHubPermissionPolicy

logger = logging.getLogger(__name__)


class GitHubServiceError(Exception):
    def __init__(self, message: str, operation: str = "", status_code: int = 0):
        super().__init__(message)
        self.operation = operation
        self.status_code = status_code


class GitHubService:
    """High-level GitHub operations with permission enforcement.

    Args:
        credential_provider: Source of GitHub tokens.
        policy: Permission policy for operations.
    """

    def __init__(
        self,
        credential_provider: CredentialProvider,
        policy: Optional[GitHubPermissionPolicy] = None,
    ):
        self._credentials = credential_provider
        self._policy = policy or GitHubPermissionPolicy()
        self._client: Optional[GitHubClient] = None

    @property
    def is_authenticated(self) -> bool:
        return self._credentials.is_available("github")

    def _get_client(self) -> GitHubClient:
        if self._client is None:
            token = self._credentials.get_token("github")
            if not token:
                raise GitHubServiceError("GitHub not authenticated — no token available")
            self._client = GitHubClient(token)
        return self._client

    def _authorize(self, operation: GitHubOperation, repository: str = "") -> None:
        result = self._policy.check(operation, repository)
        if not result.allowed:
            raise GitHubServiceError(
                f"Operation denied by policy: {operation.value} — {result.reason}",
                operation=operation.value,
            )
        if result.requires_approval:
            logger.info(
                "GitHub operation %s requires approval for repo=%s",
                operation.value, repository,
            )

    def _repo_str(self, owner: str, repo: str) -> str:
        return f"{owner}/{repo}"

    # ---- Repository operations -------------------------------------------

    def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        self._authorize(GitHubOperation.GET_REPOSITORY, self._repo_str(owner, repo))
        try:
            return self._get_client().get_repository(owner, repo)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to get repository: {exc}",
                operation="get_repository",
                status_code=exc.status_code,
            ) from exc

    def list_repositories(self, per_page: int = 30) -> List[GitHubRepository]:
        self._authorize(GitHubOperation.LIST_REPOSITORIES)
        try:
            return self._get_client().list_repositories(per_page)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to list repositories: {exc}",
                operation="list_repositories",
            ) from exc

    # ---- Branch operations -----------------------------------------------

    def list_branches(self, owner: str, repo: str) -> List[GitHubBranch]:
        self._authorize(GitHubOperation.LIST_BRANCHES, self._repo_str(owner, repo))
        try:
            return self._get_client().list_branches(owner, repo)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to list branches: {exc}",
                operation="list_branches",
            ) from exc

    def get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        self._authorize(GitHubOperation.GET_BRANCH, self._repo_str(owner, repo))
        try:
            return self._get_client().get_branch(owner, repo, branch)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to get branch: {exc}",
                operation="get_branch",
            ) from exc

    def create_branch(self, owner: str, repo: str, branch: str, from_sha: str) -> GitHubBranch:
        self._authorize(GitHubOperation.CREATE_BRANCH, self._repo_str(owner, repo))
        try:
            result = self._get_client().create_branch(owner, repo, branch, from_sha)
            logger.info("Created branch %s on %s/%s", branch, owner, repo)
            return result
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to create branch: {exc}",
                operation="create_branch",
            ) from exc

    # ---- File operations -------------------------------------------------

    def get_file(self, owner: str, repo: str, path: str, ref: str = "") -> GitHubFile:
        self._authorize(GitHubOperation.GET_FILE, self._repo_str(owner, repo))
        try:
            return self._get_client().get_file(owner, repo, path, ref)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to get file {path}: {exc}",
                operation="get_file",
            ) from exc

    def update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        sha: str,
        branch: str = "",
    ) -> Dict[str, Any]:
        self._authorize(GitHubOperation.UPDATE_FILE, self._repo_str(owner, repo))
        try:
            result = self._get_client().update_file(owner, repo, path, content, message, sha, branch)
            logger.info("Updated file %s on %s/%s (branch=%s)", path, owner, repo, branch)
            return result
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to update file {path}: {exc}",
                operation="update_file",
            ) from exc

    # ---- Commit operations -----------------------------------------------

    def list_commits(self, owner: str, repo: str, sha: str = "") -> List[GitHubCommit]:
        self._authorize(GitHubOperation.LIST_COMMITS, self._repo_str(owner, repo))
        try:
            return self._get_client().list_commits(owner, repo, sha)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to list commits: {exc}",
                operation="list_commits",
            ) from exc

    # ---- Pull request operations -----------------------------------------

    def list_pull_requests(self, owner: str, repo: str, state: str = "open") -> List[GitHubPullRequest]:
        self._authorize(GitHubOperation.LIST_PULL_REQUESTS, self._repo_str(owner, repo))
        try:
            return self._get_client().list_pull_requests(owner, repo, state)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to list pull requests: {exc}",
                operation="list_pull_requests",
            ) from exc

    def get_pull_request(self, owner: str, repo: str, number: int) -> GitHubPullRequest:
        self._authorize(GitHubOperation.GET_PULL_REQUEST, self._repo_str(owner, repo))
        try:
            return self._get_client().get_pull_request(owner, repo, number)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to get pull request #{number}: {exc}",
                operation="get_pull_request",
            ) from exc

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
    ) -> GitHubPullRequest:
        self._authorize(GitHubOperation.CREATE_PULL_REQUEST, self._repo_str(owner, repo))
        try:
            result = self._get_client().create_pull_request(owner, repo, title, head, base, body)
            logger.info("Created PR #%d on %s/%s: %s", result.number, owner, repo, title)
            return result
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to create pull request: {exc}",
                operation="create_pull_request",
            ) from exc

    # ---- Issue operations ------------------------------------------------

    def list_issues(self, owner: str, repo: str, state: str = "open") -> List[GitHubIssue]:
        self._authorize(GitHubOperation.LIST_ISSUES, self._repo_str(owner, repo))
        try:
            return self._get_client().list_issues(owner, repo, state)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to list issues: {exc}",
                operation="list_issues",
            ) from exc

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
    ) -> GitHubIssue:
        self._authorize(GitHubOperation.CREATE_ISSUE, self._repo_str(owner, repo))
        try:
            return self._get_client().create_issue(owner, repo, title, body, labels)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to create issue: {exc}",
                operation="create_issue",
            ) from exc

    # ---- Workflow operations ---------------------------------------------

    def list_workflow_runs(self, owner: str, repo: str) -> List[GitHubWorkflowRun]:
        self._authorize(GitHubOperation.GET_WORKFLOW_RUNS, self._repo_str(owner, repo))
        try:
            return self._get_client().list_workflow_runs(owner, repo)
        except GitHubClientError as exc:
            raise GitHubServiceError(
                f"Failed to list workflow runs: {exc}",
                operation="list_workflow_runs",
            ) from exc

    # ---- Authentication status -------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "authenticated": self.is_authenticated,
        }

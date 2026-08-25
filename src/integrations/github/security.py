"""GitHub permission policy — controls what operations are allowed.

Every GitHub operation is classified as READ, WRITE, or DESTRUCTIVE.
Defaults: READ allowed, WRITE configurable, DESTRUCTIVE requires approval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from src.integrations.github.models import GitHubPermission

logger = logging.getLogger(__name__)


class GitHubOperation(Enum):
    GET_REPOSITORY = "get_repository"
    LIST_REPOSITORIES = "list_repositories"
    GET_BRANCH = "get_branch"
    LIST_BRANCHES = "list_branches"
    GET_FILE = "get_file"
    LIST_COMMITS = "list_commits"
    GET_COMMIT = "get_commit"
    GET_PULL_REQUEST = "get_pull_request"
    LIST_PULL_REQUESTS = "list_pull_requests"
    GET_ISSUE = "get_issue"
    LIST_ISSUES = "list_issues"
    GET_WORKFLOW_RUNS = "get_workflow_runs"

    CREATE_BRANCH = "create_branch"
    UPDATE_FILE = "update_file"
    CREATE_COMMIT = "create_commit"
    CREATE_PULL_REQUEST = "create_pull_request"
    CREATE_ISSUE = "create_issue"
    ADD_COMMENT = "add_comment"

    DELETE_BRANCH = "delete_branch"
    MERGE_PULL_REQUEST = "merge_pull_request"
    FORCE_PUSH = "force_push"
    DELETE_FILE = "delete_file"


_OPERATION_CLASSIFICATION: Dict[GitHubOperation, GitHubPermission] = {
    # READ
    GitHubOperation.GET_REPOSITORY: GitHubPermission.READ,
    GitHubOperation.LIST_REPOSITORIES: GitHubPermission.READ,
    GitHubOperation.GET_BRANCH: GitHubPermission.READ,
    GitHubOperation.LIST_BRANCHES: GitHubPermission.READ,
    GitHubOperation.GET_FILE: GitHubPermission.READ,
    GitHubOperation.LIST_COMMITS: GitHubPermission.READ,
    GitHubOperation.GET_COMMIT: GitHubPermission.READ,
    GitHubOperation.GET_PULL_REQUEST: GitHubPermission.READ,
    GitHubOperation.LIST_PULL_REQUESTS: GitHubPermission.READ,
    GitHubOperation.GET_ISSUE: GitHubPermission.READ,
    GitHubOperation.LIST_ISSUES: GitHubPermission.READ,
    GitHubOperation.GET_WORKFLOW_RUNS: GitHubPermission.READ,
    # WRITE
    GitHubOperation.CREATE_BRANCH: GitHubPermission.WRITE,
    GitHubOperation.UPDATE_FILE: GitHubPermission.WRITE,
    GitHubOperation.CREATE_COMMIT: GitHubPermission.WRITE,
    GitHubOperation.CREATE_PULL_REQUEST: GitHubPermission.WRITE,
    GitHubOperation.CREATE_ISSUE: GitHubPermission.WRITE,
    GitHubOperation.ADD_COMMENT: GitHubPermission.WRITE,
    # DESTRUCTIVE
    GitHubOperation.DELETE_BRANCH: GitHubPermission.DESTRUCTIVE,
    GitHubOperation.MERGE_PULL_REQUEST: GitHubPermission.DESTRUCTIVE,
    GitHubOperation.FORCE_PUSH: GitHubPermission.DESTRUCTIVE,
    GitHubOperation.DELETE_FILE: GitHubPermission.DESTRUCTIVE,
}


@dataclass
class GitHubPermissionPolicy:
    """Controls which GitHub operations are permitted.

    Defaults:
        - READ: always allowed
        - WRITE: configurable (default: requires approval)
        - DESTRUCTIVE: always requires explicit approval
    """

    read_allowed: bool = True
    write_requires_approval: bool = True
    destructive_requires_approval: bool = True
    allowed_repositories: Optional[Set[str]] = None
    denied_repositories: Optional[Set[str]] = None

    def classify(self, operation: GitHubOperation) -> GitHubPermission:
        return _OPERATION_CLASSIFICATION.get(operation, GitHubPermission.READ)

    def check(
        self,
        operation: GitHubOperation,
        repository: str = "",
    ) -> _GitHubPolicyResult:
        """Evaluate whether an operation is permitted."""
        if self.denied_repositories and repository in self.denied_repositories:
            return _GitHubPolicyResult(
                allowed=False,
                reason=f"Repository explicitly denied: {repository}",
                permission=GitHubPermission.READ,
                requires_approval=False,
            )

        if self.allowed_repositories and repository not in self.allowed_repositories:
            return _GitHubPolicyResult(
                allowed=False,
                reason=f"Repository not in allowed list: {repository}",
                permission=GitHubPermission.READ,
                requires_approval=False,
            )

        permission = self.classify(operation)

        if permission == GitHubPermission.READ:
            if not self.read_allowed:
                return _GitHubPolicyResult(
                    allowed=False,
                    reason="Read operations disabled",
                    permission=permission,
                    requires_approval=False,
                )
            return _GitHubPolicyResult(
                allowed=True,
                reason="Read allowed",
                permission=permission,
                requires_approval=False,
            )

        if permission == GitHubPermission.WRITE:
            return _GitHubPolicyResult(
                allowed=True,
                reason="Write allowed with approval" if self.write_requires_approval else "Write allowed",
                permission=permission,
                requires_approval=self.write_requires_approval,
            )

        if permission == GitHubPermission.DESTRUCTIVE:
            return _GitHubPolicyResult(
                allowed=True,
                reason="Destructive operations require explicit approval",
                permission=permission,
                requires_approval=True,
            )

        return _GitHubPolicyResult(
            allowed=False,
            reason="Unknown permission level",
            permission=permission,
            requires_approval=False,
        )


@dataclass(frozen=True)
class _GitHubPolicyResult:
    allowed: bool
    reason: str
    permission: GitHubPermission
    requires_approval: bool

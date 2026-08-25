"""Voxline Integrations — GitHub, Vercel, and external service abstractions.

All external service access flows through typed clients, credential providers,
and permission layers. The LLM never receives credentials directly.
"""

from src.integrations.credentials import (
    CredentialProvider,
    EnvironmentCredentialProvider,
)
from src.integrations.github.models import (
    GitHubRepository,
    GitHubBranch,
    GitHubCommit,
    GitHubPullRequest,
    GitHubIssue,
    GitHubWorkflowRun,
    GitHubFile,
    GitHubPermission,
)
from src.integrations.github.client import GitHubClient
from src.integrations.github.security import GitHubPermissionPolicy
from src.integrations.vercel.models import (
    VercelProject,
    VercelDeployment,
    VercelDomain,
    VercelDeploymentStatus,
)
from src.integrations.vercel.client import VercelClient
from src.integrations.vercel.security import VercelPermissionPolicy

__all__ = [
    "CredentialProvider",
    "EnvironmentCredentialProvider",
    "GitHubRepository",
    "GitHubBranch",
    "GitHubCommit",
    "GitHubPullRequest",
    "GitHubIssue",
    "GitHubWorkflowRun",
    "GitHubFile",
    "GitHubPermission",
    "GitHubClient",
    "GitHubPermissionPolicy",
    "VercelProject",
    "VercelDeployment",
    "VercelDomain",
    "VercelDeploymentStatus",
    "VercelClient",
    "VercelPermissionPolicy",
]

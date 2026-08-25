"""ToolRegistry bootstrap — builds a fully configured registry from config.

Conditionally registers:
  - Core tools (filesystem, commands, calculator)
  - GitHub integration tools (if enabled + authenticated)
  - Vercel integration tools (if enabled + authenticated)
  - Workspace tools (clone, diff, test)

Never crashes if optional integrations are unavailable.
Never exposes credentials to the registry or tool schemas.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config.settings import VoxlineConfig
from src.integrations.credentials import (
    CredentialProvider,
    EnvironmentCredentialProvider,
)
from src.tools.security import AuditLog, PathSecurity
from src.tools.tools import ToolRegistry

logger = logging.getLogger(__name__)


def build_tool_registry(
    config: Optional[VoxlineConfig] = None,
    credential_provider: Optional[CredentialProvider] = None,
    workspace_root: Optional[str] = None,
    audit_log: Optional[AuditLog] = None,
) -> ToolRegistry:
    """Build a fully configured ToolRegistry from config.

    Core tools are always registered.
    Integration tools are registered only when enabled AND authenticated.
    Failures in optional integrations are logged and skipped — never fatal.

    Args:
        config: VoxlineConfig instance (uses global config if None).
        credential_provider: Credential source (uses env provider if None).
        workspace_root: Workspace root path (from config if None).
        audit_log: Shared audit log (creates new if None).

    Returns:
        Configured ToolRegistry with all available tools registered.
    """
    if config is None:
        from src.config.settings import get_config
        config = get_config()

    if credential_provider is None:
        credential_provider = EnvironmentCredentialProvider()

    if workspace_root is None:
        workspace_root = str(config.workspace_root)

    if audit_log is None:
        audit_log = AuditLog()

    allowed_commands = config.coding_allowed_commands

    registry = ToolRegistry(
        workspace_root=workspace_root,
        max_file_size_bytes=config.coding_max_file_size_mb * 1024 * 1024,
        max_output_bytes=config.coding_max_output_bytes,
        command_timeout=config.agent_step_timeout,
        allowed_commands=allowed_commands,
        audit_log=audit_log,
    )

    _register_github_tools(registry, config, credential_provider)
    _register_vercel_tools(registry, config, credential_provider)
    _register_workspace_tools(registry, workspace_root)

    tool_count = len(registry.tools)
    categories = _categorize_tools(registry)
    logger.info(
        "ToolRegistry built: %d tools (%s)",
        tool_count,
        ", ".join(f"{cat}: {len(tools)}" for cat, tools in categories.items()),
    )

    return registry


def _register_github_tools(
    registry: ToolRegistry,
    config: VoxlineConfig,
    credentials: CredentialProvider,
) -> None:
    """Register GitHub tools if enabled and authenticated."""
    if not config.github_enabled:
        logger.debug("GitHub integration disabled — skipping tool registration")
        return

    if not credentials.is_available("github"):
        logger.debug("GitHub credentials unavailable — skipping tool registration")
        return

    try:
        from src.integrations.github.security import GitHubPermissionPolicy
        from src.integrations.github.service import GitHubService
        from src.tools.integration_tools import (
            GitHubCommitTool,
            GitHubCreateBranchTool,
            GitHubCreatePullRequestTool,
            GitHubListIssuesTool,
            GitHubReadFileTool,
            GitHubReadRepositoryTool,
        )

        allowed_repos = config.github_allowed_repositories
        policy = GitHubPermissionPolicy(
            allowed_repositories=allowed_repos if allowed_repos else None,
        )
        service = GitHubService(
            credential_provider=credentials,
            policy=policy,
        )

        registry.register("github_read_repository", GitHubReadRepositoryTool(service))
        registry.register("github_read_file", GitHubReadFileTool(service))
        registry.register("github_create_branch", GitHubCreateBranchTool(service))
        registry.register("github_commit", GitHubCommitTool(service))
        registry.register("github_create_pull_request", GitHubCreatePullRequestTool(service))
        registry.register("github_list_issues", GitHubListIssuesTool(service))

        logger.info("GitHub tools registered (6 tools)")
    except Exception as exc:
        logger.warning("Failed to register GitHub tools: %s", exc)


def _register_vercel_tools(
    registry: ToolRegistry,
    config: VoxlineConfig,
    credentials: CredentialProvider,
) -> None:
    """Register Vercel tools if enabled and authenticated."""
    if not config.vercel_enabled:
        logger.debug("Vercel integration disabled — skipping tool registration")
        return

    if not credentials.is_available("vercel"):
        logger.debug("Vercel credentials unavailable — skipping tool registration")
        return

    try:
        from src.integrations.vercel.security import VercelPermissionPolicy
        from src.integrations.vercel.service import VercelService
        from src.tools.integration_tools import (
            VercelCreateDeploymentTool,
            VercelGetDeploymentTool,
            VercelListProjectsTool,
        )

        allowed_projects = config.vercel_allowed_projects
        policy = VercelPermissionPolicy(
            allowed_projects=allowed_projects if allowed_projects else None,
        )
        service = VercelService(
            credential_provider=credentials,
            policy=policy,
        )

        registry.register("vercel_list_projects", VercelListProjectsTool(service))
        registry.register("vercel_create_deployment", VercelCreateDeploymentTool(service))
        registry.register("vercel_get_deployment", VercelGetDeploymentTool(service))

        logger.info("Vercel tools registered (3 tools)")
    except Exception as exc:
        logger.warning("Failed to register Vercel tools: %s", exc)


def _register_workspace_tools(
    registry: ToolRegistry,
    workspace_root: str,
) -> None:
    """Register workspace tools (clone, diff, test)."""
    try:
        from src.tools.integration_tools import (
            WorkspaceCloneTool,
            WorkspaceDiffTool,
            WorkspaceTestTool,
        )

        path_security = PathSecurity(workspace_root)

        registry.register("workspace_clone", WorkspaceCloneTool(workspace_root, path_security))
        registry.register("workspace_diff", WorkspaceDiffTool(workspace_root))
        registry.register("workspace_test", WorkspaceTestTool(workspace_root))

        logger.info("Workspace tools registered (3 tools)")
    except Exception as exc:
        logger.warning("Failed to register workspace tools: %s", exc)


def _categorize_tools(registry: ToolRegistry) -> dict:
    """Group registered tools by category for reporting."""
    categories: dict = {}
    for name in registry.tools:
        if name.startswith("github_"):
            cat = "github"
        elif name.startswith("vercel_"):
            cat = "vercel"
        elif name.startswith("workspace_"):
            cat = "workspace"
        else:
            cat = "core"
        categories.setdefault(cat, []).append(name)
    return categories

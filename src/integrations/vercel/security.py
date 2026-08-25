"""Vercel permission policy — controls deployment operations.

PREVIEW deployments may be created automatically.
PRODUCTION deployments require explicit authorization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set

from src.integrations.vercel.models import VercelEnvironment

logger = logging.getLogger(__name__)


class VercelOperation(Enum):
    LIST_PROJECTS = "list_projects"
    GET_PROJECT = "get_project"
    LIST_DEPLOYMENTS = "list_deployments"
    GET_DEPLOYMENT = "get_deployment"
    CREATE_PREVIEW = "create_preview"
    CREATE_PRODUCTION = "create_production"


_OPERATION_REQUIRES_AUTHORIZATION = {
    VercelOperation.LIST_PROJECTS: False,
    VercelOperation.GET_PROJECT: False,
    VercelOperation.LIST_DEPLOYMENTS: False,
    VercelOperation.GET_DEPLOYMENT: False,
    VercelOperation.CREATE_PREVIEW: False,
    VercelOperation.CREATE_PRODUCTION: True,
}


@dataclass
class VercelPermissionPolicy:
    """Controls which Vercel operations are permitted.

    Defaults:
        - Read: always allowed
        - Preview deployment: allowed (no approval needed)
        - Production deployment: requires explicit approval
    """

    require_production_approval: bool = True
    allowed_projects: Optional[Set[str]] = None
    denied_projects: Optional[Set[str]] = None

    def check(
        self,
        operation: VercelOperation,
        project_id: str = "",
    ) -> _VercelPolicyResult:
        if self.denied_projects and project_id in self.denied_projects:
            return _VercelPolicyResult(
                allowed=False,
                reason=f"Project explicitly denied: {project_id}",
                requires_approval=False,
            )

        if self.allowed_projects and project_id not in self.allowed_projects:
            return _VercelPolicyResult(
                allowed=False,
                reason=f"Project not in allowed list: {project_id}",
                requires_approval=False,
            )

        requires_auth = _OPERATION_REQUIRES_AUTHORIZATION.get(operation, False)
        if requires_auth and self.require_production_approval:
            return _VercelPolicyResult(
                allowed=True,
                reason="Production deployment requires explicit authorization",
                requires_approval=True,
            )

        return _VercelPolicyResult(
            allowed=True,
            reason=f"Operation allowed: {operation.value}",
            requires_approval=False,
        )


@dataclass(frozen=True)
class _VercelPolicyResult:
    allowed: bool
    reason: str
    requires_approval: bool

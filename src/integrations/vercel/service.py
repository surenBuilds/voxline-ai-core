"""Vercel service layer — business logic that coordinates client + security."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.integrations.credentials import CredentialProvider
from src.integrations.vercel.client import VercelClient, VercelClientError
from src.integrations.vercel.models import VercelDeployment, VercelProject
from src.integrations.vercel.security import VercelOperation, VercelPermissionPolicy

logger = logging.getLogger(__name__)


class VercelServiceError(Exception):
    def __init__(self, message: str, operation: str = "", status_code: int = 0):
        super().__init__(message)
        self.operation = operation
        self.status_code = status_code


class VercelService:
    """High-level Vercel operations with permission enforcement."""

    def __init__(
        self,
        credential_provider: CredentialProvider,
        policy: Optional[VercelPermissionPolicy] = None,
    ):
        self._credentials = credential_provider
        self._policy = policy or VercelPermissionPolicy()
        self._client: Optional[VercelClient] = None

    @property
    def is_authenticated(self) -> bool:
        return self._credentials.is_available("vercel")

    def _get_client(self) -> VercelClient:
        if self._client is None:
            token = self._credentials.get_token("vercel")
            if not token:
                raise VercelServiceError("Vercel not authenticated — no token available")
            self._client = VercelClient(token)
        return self._client

    def _authorize(self, operation: VercelOperation, project_id: str = "") -> None:
        result = self._policy.check(operation, project_id)
        if not result.allowed:
            raise VercelServiceError(
                f"Operation denied by policy: {operation.value} — {result.reason}",
                operation=operation.value,
            )
        if result.requires_approval:
            logger.info(
                "Vercel operation %s requires approval for project=%s",
                operation.value, project_id,
            )

    # ---- Project operations ----------------------------------------------

    def list_projects(self, per_page: int = 20) -> List[VercelProject]:
        self._authorize(VercelOperation.LIST_PROJECTS)
        try:
            return self._get_client().list_projects(per_page)
        except VercelClientError as exc:
            raise VercelServiceError(
                f"Failed to list projects: {exc}", operation="list_projects",
            ) from exc

    def get_project(self, project_id: str) -> VercelProject:
        self._authorize(VercelOperation.GET_PROJECT, project_id)
        try:
            return self._get_client().get_project(project_id)
        except VercelClientError as exc:
            raise VercelServiceError(
                f"Failed to get project: {exc}", operation="get_project",
            ) from exc

    # ---- Deployment operations -------------------------------------------

    def list_deployments(
        self,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[VercelDeployment]:
        self._authorize(VercelOperation.LIST_DEPLOYMENTS, project_id or "")
        try:
            return self._get_client().list_deployments(project_id, limit)
        except VercelClientError as exc:
            raise VercelServiceError(
                f"Failed to list deployments: {exc}", operation="list_deployments",
            ) from exc

    def get_deployment(self, deployment_id: str) -> VercelDeployment:
        self._authorize(VercelOperation.GET_DEPLOYMENT)
        try:
            return self._get_client().get_deployment(deployment_id)
        except VercelClientError as exc:
            raise VercelServiceError(
                f"Failed to get deployment: {exc}", operation="get_deployment",
            ) from exc

    def create_preview_deployment(
        self,
        project_id: str,
        name: str,
        git_source: Optional[Dict[str, str]] = None,
    ) -> VercelDeployment:
        self._authorize(VercelOperation.CREATE_PREVIEW, project_id)
        try:
            result = self._get_client().create_deployment(
                project_id, name, git_source, target="preview",
            )
            logger.info("Created preview deployment for project %s", project_id)
            return result
        except VercelClientError as exc:
            raise VercelServiceError(
                f"Failed to create preview deployment: {exc}",
                operation="create_preview",
            ) from exc

    def create_production_deployment(
        self,
        project_id: str,
        name: str,
        git_source: Optional[Dict[str, str]] = None,
    ) -> VercelDeployment:
        self._authorize(VercelOperation.CREATE_PRODUCTION, project_id)
        try:
            result = self._get_client().create_deployment(
                project_id, name, git_source, target="production",
            )
            logger.info("Created production deployment for project %s", project_id)
            return result
        except VercelClientError as exc:
            raise VercelServiceError(
                f"Failed to create production deployment: {exc}",
                operation="create_production",
            ) from exc

    # ---- Authentication status -------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "authenticated": self.is_authenticated,
        }

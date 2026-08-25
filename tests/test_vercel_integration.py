"""
Tests for Vercel integration.

Covers:
  MODELS (tests 1-4)
  PERMISSION POLICY (tests 5-9)
  CLIENT (tests 10-13)
  SERVICE (tests 14-18)
  TOOLS (tests 19-22)
  SECURITY (tests 23-25)
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integrations.credentials import EnvironmentCredentialProvider
from src.integrations.vercel.models import (
    VercelDeployment,
    VercelDeploymentStatus,
    VercelDomain,
    VercelEnvironment,
    VercelProject,
)
from src.integrations.vercel.client import VercelClient, VercelClientError
from src.integrations.vercel.security import VercelOperation, VercelPermissionPolicy
from src.integrations.vercel.service import VercelService, VercelServiceError
from src.tools.integration_tools import (
    VercelListProjectsTool,
    VercelCreateDeploymentTool,
    VercelGetDeploymentTool,
)


# ---------------------------------------------------------------------------
# Mock Vercel API responses
# ---------------------------------------------------------------------------

MOCK_PROJECT_DATA = {
    "id": "prj_abc123",
    "name": "my-app",
    "framework": "nextjs",
    "link": {"repo": "testowner/test-repo", "type": "github"},
    "createdAt": "2026-01-01",
    "updatedAt": "2026-01-02",
}

MOCK_DEPLOYMENT_DATA = {
    "uid": "dpl_xyz789",
    "url": "my-app-abc123.vercel.app",
    "name": "my-app",
    "state": "READY",
    "target": "preview",
    "branch": "fix-auth",
    "commit": "def456",
    "created": "2026-01-01T00:00:00Z",
    "readyState": "READY",
    "meta": {"branch": "fix-auth", "githubCommitSha": "def456"},
}


# =========================================================================
# MODELS (tests 1-4)
# =========================================================================


class TestVercelModels(unittest.TestCase):
    """Tests 1-4: Vercel model creation and parsing."""

    def test_01_project_from_api(self):
        project = VercelProject.from_api(MOCK_PROJECT_DATA)
        self.assertEqual(project.id, "prj_abc123")
        self.assertEqual(project.name, "my-app")
        self.assertEqual(project.framework, "nextjs")

    def test_02_deployment_from_api(self):
        deploy = VercelDeployment.from_api(MOCK_DEPLOYMENT_DATA)
        self.assertEqual(deploy.id, "dpl_xyz789")
        self.assertEqual(deploy.url, "my-app-abc123.vercel.app")
        self.assertEqual(deploy.state, VercelDeploymentStatus.READY)
        self.assertEqual(deploy.environment, VercelEnvironment.PREVIEW)
        self.assertEqual(deploy.branch, "fix-auth")

    def test_03_deployment_states(self):
        for state_str, expected in [
            ("QUEUED", VercelDeploymentStatus.QUEUED),
            ("BUILDING", VercelDeploymentStatus.BUILDING),
            ("READY", VercelDeploymentStatus.READY),
            ("ERROR", VercelDeploymentStatus.ERROR),
        ]:
            data = {**MOCK_DEPLOYMENT_DATA, "state": state_str}
            deploy = VercelDeployment.from_api(data)
            self.assertEqual(deploy.state, expected)

    def test_04_domain_from_api(self):
        domain = VercelDomain.from_api({"name": "my-app.vercel.app", "verified": True})
        self.assertEqual(domain.name, "my-app.vercel.app")
        self.assertTrue(domain.verified)


# =========================================================================
# PERMISSION POLICY (tests 5-9)
# =========================================================================


class TestVercelPermissionPolicy(unittest.TestCase):
    """Tests 5-9: Vercel permission policy."""

    def test_05_read_allowed(self):
        policy = VercelPermissionPolicy()
        result = policy.check(VercelOperation.LIST_PROJECTS)
        self.assertTrue(result.allowed)
        self.assertFalse(result.requires_approval)

    def test_06_preview_no_approval(self):
        policy = VercelPermissionPolicy()
        result = policy.check(VercelOperation.CREATE_PREVIEW)
        self.assertTrue(result.allowed)
        self.assertFalse(result.requires_approval)

    def test_07_production_requires_approval(self):
        policy = VercelPermissionPolicy()
        result = policy.check(VercelOperation.CREATE_PRODUCTION)
        self.assertTrue(result.allowed)
        self.assertTrue(result.requires_approval)

    def test_08_project_denied(self):
        policy = VercelPermissionPolicy(denied_projects={"bad-project"})
        result = policy.check(VercelOperation.LIST_PROJECTS, "bad-project")
        self.assertFalse(result.allowed)

    def test_09_project_not_in_allowed(self):
        policy = VercelPermissionPolicy(allowed_projects={"good-project"})
        result = policy.check(VercelOperation.LIST_PROJECTS, "other-project")
        self.assertFalse(result.allowed)


# =========================================================================
# CLIENT (tests 10-13)
# =========================================================================


class TestVercelClient(unittest.TestCase):
    """Tests 10-13: Vercel client (mocked HTTP)."""

    def test_10_client_requires_token(self):
        with self.assertRaises(ValueError):
            VercelClient("")

    def test_11_client_list_projects(self):
        client = VercelClient("fake_token")
        with patch("urllib.request.urlopen") as mock_open:
            resp = MagicMock()
            resp.read.return_value = json.dumps({"projects": [MOCK_PROJECT_DATA]}).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = resp
            projects = client.list_projects()
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0].name, "my-app")

    def test_12_client_get_deployment(self):
        client = VercelClient("fake_token")
        with patch("urllib.request.urlopen") as mock_open:
            resp = MagicMock()
            resp.read.return_value = json.dumps(MOCK_DEPLOYMENT_DATA).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = resp
            deploy = client.get_deployment("dpl_xyz789")
            self.assertEqual(deploy.id, "dpl_xyz789")

    def test_13_client_handles_http_error(self):
        client = VercelClient("fake_token")
        with patch("urllib.request.urlopen") as mock_open:
            import urllib.error
            http_err = urllib.error.HTTPError(
                url="https://api.vercel.com", code=404,
                msg="Not found", hdrs=None, fp=MagicMock(),
            )
            http_err.read = MagicMock(return_value=b'{"error":"not found"}')
            mock_open.side_effect = http_err
            with self.assertRaises(VercelClientError):
                client.get_deployment("nonexistent")


# =========================================================================
# SERVICE (tests 14-18)
# =========================================================================


class TestVercelService(unittest.TestCase):
    """Tests 14-18: Vercel service with mocked client."""

    def _make_service(self, **policy_kwargs):
        from src.integrations.credentials import CredentialProvider

        class MockCreds(CredentialProvider):
            def get_token(self, service):
                return "fake_token"
            def is_available(self, service):
                return True
            def redact(self, text):
                return text

        return VercelService(MockCreds(), VercelPermissionPolicy(**policy_kwargs))

    def test_14_service_list_projects(self):
        service = self._make_service()
        service._client = VercelClient("fake_token")
        with patch.object(VercelClient, "list_projects") as mock_list:
            mock_list.return_value = [VercelProject(id="1", name="app")]
            projects = service.list_projects()
            self.assertEqual(len(projects), 1)

    def test_15_service_create_preview(self):
        service = self._make_service()
        service._client = VercelClient("fake_token")
        with patch.object(VercelClient, "create_deployment") as mock_create:
            mock_create.return_value = VercelDeployment(
                id="dpl1", url="app.vercel.app", name="app",
                state=VercelDeploymentStatus.READY,
                environment=VercelEnvironment.PREVIEW,
            )
            deploy = service.create_preview_deployment("prj1", "app")
            self.assertEqual(deploy.id, "dpl1")

    def test_16_service_create_production_requires_approval(self):
        service = self._make_service(require_production_approval=True)
        with self.assertRaises(VercelServiceError):
            service.create_production_deployment("prj1", "app")

    def test_17_service_unauthenticated(self):
        from src.integrations.credentials import CredentialProvider

        class NoCreds(CredentialProvider):
            def get_token(self, service):
                return None
            def is_available(self, service):
                return False
            def redact(self, text):
                return text

        service = VercelService(NoCreds())
        with self.assertRaises(VercelServiceError):
            service.list_projects()

    def test_18_service_status(self):
        service = self._make_service()
        status = service.get_status()
        self.assertTrue(status["authenticated"])


# =========================================================================
# TOOLS (tests 19-22)
# =========================================================================


class TestVercelTools(unittest.TestCase):
    """Tests 19-22: Vercel tools for ToolRegistry."""

    def _make_service(self):
        from src.integrations.credentials import CredentialProvider

        class MockCreds(CredentialProvider):
            def get_token(self, service):
                return "fake_token"
            def is_available(self, service):
                return True
            def redact(self, text):
                return text

        return VercelService(MockCreds())

    def test_19_list_projects_tool(self):
        service = self._make_service()
        service._client = VercelClient("fake_token")
        tool = VercelListProjectsTool(service)
        with patch.object(VercelClient, "list_projects") as mock_list:
            mock_list.return_value = [VercelProject(id="1", name="app", framework="nextjs")]
            result = tool.execute()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["name"], "app")

    def test_20_create_deployment_tool(self):
        service = self._make_service()
        service._client = VercelClient("fake_token")
        tool = VercelCreateDeploymentTool(service)
        with patch.object(VercelClient, "create_deployment") as mock_create:
            mock_create.return_value = VercelDeployment(
                id="dpl1", url="app.vercel.app", name="app",
                state=VercelDeploymentStatus.BUILDING,
                environment=VercelEnvironment.PREVIEW,
            )
            result = tool.execute(project_id="prj1", name="app", target="preview")
            self.assertEqual(result["id"], "dpl1")
            self.assertEqual(result["state"], "building")

    def test_21_get_deployment_tool(self):
        service = self._make_service()
        service._client = VercelClient("fake_token")
        tool = VercelGetDeploymentTool(service)
        with patch.object(VercelClient, "get_deployment") as mock_get:
            mock_get.return_value = VercelDeployment(
                id="dpl1", url="app.vercel.app", name="app",
                state=VercelDeploymentStatus.READY,
                environment=VercelEnvironment.PREVIEW,
            )
            result = tool.execute(deployment_id="dpl1")
            self.assertEqual(result["state"], "ready")

    def test_22_tool_error_handling(self):
        service = self._make_service()
        service._client = VercelClient("fake_token")
        tool = VercelListProjectsTool(service)
        with patch.object(VercelClient, "list_projects") as mock_list:
            mock_list.side_effect = VercelClientError("Auth failed", 401)
            result = tool.execute()
            self.assertIn("error", result[0])


# =========================================================================
# SECURITY (tests 23-25)
# =========================================================================


class TestVercelSecurity(unittest.TestCase):
    """Tests 23-25: Vercel security requirements."""

    def test_23_tokens_never_exposed(self):
        provider = EnvironmentCredentialProvider()
        with patch.dict(os.environ, {"VERCEL_TOKEN": "vrcl_SECRET_12345"}):
            redacted = provider.redact("Token: vrcl_SECRET_12345")
            self.assertNotIn("vrcl_SECRET_12345", redacted)

    def test_24_production_always_requires_approval(self):
        policy = VercelPermissionPolicy()
        result = policy.check(VercelOperation.CREATE_PRODUCTION)
        self.assertTrue(result.requires_approval)

    def test_25_preview_does_not_require_approval(self):
        policy = VercelPermissionPolicy()
        result = policy.check(VercelOperation.CREATE_PREVIEW)
        self.assertFalse(result.requires_approval)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    unittest.main()

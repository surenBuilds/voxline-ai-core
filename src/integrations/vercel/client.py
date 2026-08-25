"""Vercel API client — typed HTTP interface to the Vercel REST API.

Uses only stdlib (urllib.request). Never logs tokens.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from src.integrations.vercel.models import VercelDeployment, VercelProject

logger = logging.getLogger(__name__)

_VERCEL_API = "https://api.vercel.com"


class VercelClientError(Exception):
    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class VercelClient:
    """Low-level Vercel REST API client."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("Vercel token must not be empty")
        self._token = token

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{_VERCEL_API}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v)
            if query:
                url = f"{url}?{query}"

        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                if not raw.strip():
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise VercelClientError(
                f"Vercel API error {exc.code}: {exc.reason}",
                status_code=exc.code,
                response_body=body_text,
            ) from exc
        except urllib.error.URLError as exc:
            raise VercelClientError(f"Vercel API connection error: {exc.reason}") from exc

    # ---- Project operations ----------------------------------------------

    def list_projects(self, per_page: int = 20) -> List[VercelProject]:
        data = self._request("GET", "/v9/projects", {"limit": str(per_page)})
        projects = data.get("projects", []) if isinstance(data, dict) else []
        return [VercelProject.from_api(p) for p in projects]

    def get_project(self, project_id: str) -> VercelProject:
        data = self._request("GET", f"/v9/projects/{project_id}")
        return VercelProject.from_api(data)

    # ---- Deployment operations -------------------------------------------

    def list_deployments(
        self,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[VercelDeployment]:
        params: Dict[str, str] = {"limit": str(limit)}
        if project_id:
            params["projectId"] = project_id
        data = self._request("GET", "/v6/deployments", params)
        deployments = data.get("deployments", []) if isinstance(data, dict) else []
        return [VercelDeployment.from_api(d) for d in deployments]

    def get_deployment(self, deployment_id: str) -> VercelDeployment:
        data = self._request("GET", f"/v13/deployments/{deployment_id}")
        return VercelDeployment.from_api(data)

    def create_deployment(
        self,
        project_id: str,
        name: str,
        git_source: Optional[Dict[str, str]] = None,
        target: str = "preview",
    ) -> VercelDeployment:
        body: Dict[str, Any] = {
            "name": name,
            "projectId": project_id,
            "target": target,
        }
        if git_source:
            body["gitSource"] = git_source
        data = self._request("POST", "/v13/deployments", body=body)
        return VercelDeployment.from_api(data)

    # ---- Authentication status -------------------------------------------

    def verify_authentication(self) -> Dict[str, Any]:
        return self._request("GET", "/v2/user")

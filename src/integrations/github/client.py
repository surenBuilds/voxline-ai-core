"""GitHub API client — typed HTTP interface to the GitHub REST API.

Uses only stdlib (urllib.request). No third-party HTTP libraries.
Never logs tokens. Never exposes raw credentials.

The client is a transport layer. Authorization is checked by the service layer.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from src.integrations.github.models import (
    GitHubBranch,
    GitHubCommit,
    GitHubFile,
    GitHubIssue,
    GitHubPullRequest,
    GitHubRepository,
    GitHubWorkflowRun,
)

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


class GitHubClientError(Exception):
    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GitHubClient:
    """Low-level GitHub REST API client.

    Args:
        token: GitHub personal access token (or App installation token).
               Stored only in memory, never logged.
    """

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("GitHub token must not be empty")
        self._token = token

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{_GITHUB_API}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v)
            if query:
                url = f"{url}?{query}"

        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data:
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
            raise GitHubClientError(
                f"GitHub API error {exc.code}: {exc.reason}",
                status_code=exc.code,
                response_body=body_text,
            ) from exc
        except urllib.error.URLError as exc:
            raise GitHubClientError(f"GitHub API connection error: {exc.reason}") from exc

    def _request_list(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        result = self._request(method, path, params=params)
        if isinstance(result, list):
            return result
        return result.get("items", result.get("workflows", []))

    def _paginate(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        current_params = dict(params or {})
        for _ in range(max_pages):
            url = f"{_GITHUB_API}{path}"
            if current_params:
                query = "&".join(f"{k}={v}" for k, v in current_params.items() if v)
                if query:
                    url = f"{url}?{query}"
            data = None
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Authorization", f"Bearer {self._token}")
            req.add_header("Accept", "application/vnd.github+json")
            req.add_header("X-GitHub-Api-Version", "2022-11-28")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    result = json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError:
                break
            except urllib.error.URLError:
                break

            if isinstance(result, list):
                all_items.extend(result)
                break
            items = result.get("items", result.get("workflows", []))
            all_items.extend(items)

            link_header = ""
            try:
                link_header = resp.headers.get("Link", "")
            except Exception:
                pass
            if 'rel="next"' not in link_header:
                break
            break

        return all_items

    # ---- Repository operations -------------------------------------------

    def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        data = self._request("GET", f"/repos/{owner}/{repo}")
        return GitHubRepository.from_api(data)

    def list_repositories(self, per_page: int = 30) -> List[GitHubRepository]:
        data = self._request_list("GET", "/user/repos", {"per_page": str(per_page)})
        return [GitHubRepository.from_api(item) for item in data]

    # ---- Branch operations -----------------------------------------------

    def list_branches(self, owner: str, repo: str) -> List[GitHubBranch]:
        data = self._request_list("GET", f"/repos/{owner}/{repo}/branches")
        return [GitHubBranch.from_api(item) for item in data]

    def get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        data = self._request("GET", f"/repos/{owner}/{repo}/branches/{branch}")
        return GitHubBranch.from_api(data)

    def create_branch(self, owner: str, repo: str, branch: str, from_sha: str) -> GitHubBranch:
        data = self._request("POST", f"/repos/{owner}/{repo}/git/refs", {
            "ref": f"refs/heads/{branch}",
            "sha": from_sha,
        })
        return GitHubBranch(
            name=branch,
            sha=from_sha,
        )

    # ---- File operations -------------------------------------------------

    def get_file(self, owner: str, repo: str, path: str, ref: str = "") -> GitHubFile:
        params = {"ref": ref} if ref else None
        data = self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)
        return GitHubFile.from_api(data)

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
        import base64
        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        body: Dict[str, Any] = {
            "message": message,
            "content": encoded,
            "sha": sha,
        }
        if branch:
            body["branch"] = branch
        return self._request("PUT", f"/repos/{owner}/{repo}/contents/{path}", body=body)

    # ---- Commit operations -----------------------------------------------

    def list_commits(self, owner: str, repo: str, sha: str = "", per_page: int = 10) -> List[GitHubCommit]:
        params: Dict[str, str] = {"per_page": str(per_page)}
        if sha:
            params["sha"] = sha
        data = self._request_list("GET", f"/repos/{owner}/{repo}/commits", params)
        return [GitHubCommit.from_api(item) for item in data]

    def get_commit(self, owner: str, repo: str, sha: str) -> GitHubCommit:
        data = self._request("GET", f"/repos/{owner}/{repo}/commits/{sha}")
        return GitHubCommit.from_api(data)

    # ---- Pull request operations -----------------------------------------

    def list_pull_requests(self, owner: str, repo: str, state: str = "open") -> List[GitHubPullRequest]:
        data = self._request_list("GET", f"/repos/{owner}/{repo}/pulls", {"state": state})
        return [GitHubPullRequest.from_api(item) for item in data]

    def get_pull_request(self, owner: str, repo: str, number: int) -> GitHubPullRequest:
        data = self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}")
        return GitHubPullRequest.from_api(data)

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
    ) -> GitHubPullRequest:
        data = self._request("POST", f"/repos/{owner}/{repo}/pulls", {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        })
        return GitHubPullRequest.from_api(data)

    # ---- Issue operations ------------------------------------------------

    def list_issues(self, owner: str, repo: str, state: str = "open") -> List[GitHubIssue]:
        data = self._request_list("GET", f"/repos/{owner}/{repo}/issues", {"state": state})
        return [GitHubIssue.from_api(item) for item in data]

    def get_issue(self, owner: str, repo: str, number: int) -> GitHubIssue:
        data = self._request("GET", f"/repos/{owner}/{repo}/issues/{number}")
        return GitHubIssue.from_api(data)

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
    ) -> GitHubIssue:
        payload: Dict[str, Any] = {"title": title}
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        data = self._request("POST", f"/repos/{owner}/{repo}/issues", body=payload)
        return GitHubIssue.from_api(data)

    def add_comment(self, owner: str, repo: str, issue_number: int, body: str) -> Dict[str, Any]:
        return self._request("POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {
            "body": body,
        })

    # ---- Workflow operations ---------------------------------------------

    def list_workflow_runs(self, owner: str, repo: str, per_page: int = 10) -> List[GitHubWorkflowRun]:
        data = self._request("GET", f"/repos/{owner}/{repo}/actions/runs", {"per_page": str(per_page)})
        runs = data.get("workflow_runs", []) if isinstance(data, dict) else []
        return [GitHubWorkflowRun.from_api(item) for item in runs]

    # ---- Auth verification ------------------------------------------------

    def verify_authentication(self) -> Dict[str, Any]:
        return self._request("GET", "/user")

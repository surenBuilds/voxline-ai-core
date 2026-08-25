"""GitHub data models — typed representations of GitHub entities.

All models are pure data. No credentials, no API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GitHubPermission(Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class GitHubRepository:
    owner: str
    name: str
    full_name: str
    default_branch: str
    description: str = ""
    private: bool = False
    url: str = ""
    clone_url: str = ""
    ssh_url: str = ""
    language: Optional[str] = None
    stars: int = 0
    open_issues: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> GitHubRepository:
        owner = data.get("owner", {})
        owner_login = owner.get("login", "") if isinstance(owner, dict) else ""
        return cls(
            owner=owner_login,
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            default_branch=data.get("default_branch", "main"),
            description=data.get("description") or "",
            private=data.get("private", False),
            url=data.get("html_url", ""),
            clone_url=data.get("clone_url", ""),
            ssh_url=data.get("ssh_url", ""),
            language=data.get("language"),
            stars=data.get("stargazers_count", 0),
            open_issues=data.get("open_issues_count", 0),
        )


@dataclass(frozen=True)
class GitHubBranch:
    name: str
    sha: str
    is_default: bool = False
    protected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> GitHubBranch:
        commit = data.get("commit", {})
        sha = commit.get("sha", "") if isinstance(commit, dict) else ""
        return cls(
            name=data.get("name", ""),
            sha=sha,
            protected=data.get("protected", False),
        )


@dataclass(frozen=True)
class GitHubCommit:
    sha: str
    message: str
    author: str = ""
    date: str = ""
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> GitHubCommit:
        commit_data = data.get("commit", data)
        author_data = commit_data.get("author", {})
        return cls(
            sha=data.get("sha", ""),
            message=commit_data.get("message", ""),
            author=author_data.get("name", "") if isinstance(author_data, dict) else "",
            date=author_data.get("date", "") if isinstance(author_data, dict) else "",
            url=data.get("html_url", ""),
        )


@dataclass(frozen=True)
class GitHubFile:
    path: str
    content: str
    sha: str
    size: int = 0
    encoding: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> GitHubFile:
        import base64
        content_b64 = data.get("content", "")
        encoding = data.get("encoding", "")
        content = ""
        if encoding == "base64" and content_b64:
            try:
                content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            except Exception:
                content = ""
        return cls(
            path=data.get("path", ""),
            content=content,
            sha=data.get("sha", ""),
            size=data.get("size", 0),
            encoding=encoding,
        )


@dataclass(frozen=True)
class GitHubPullRequest:
    number: int
    title: str
    state: str
    head_branch: str
    base_branch: str
    url: str = ""
    body: str = ""
    author: str = ""
    mergeable: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> GitHubPullRequest:
        head = data.get("head", {})
        base = data.get("base", {})
        user = data.get("user", {})
        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            state=data.get("state", ""),
            head_branch=head.get("ref", "") if isinstance(head, dict) else "",
            base_branch=base.get("ref", "") if isinstance(base, dict) else "",
            url=data.get("html_url", ""),
            body=data.get("body") or "",
            author=user.get("login", "") if isinstance(user, dict) else "",
            mergeable=data.get("mergeable"),
        )


@dataclass(frozen=True)
class GitHubIssue:
    number: int
    title: str
    state: str
    url: str = ""
    body: str = ""
    author: str = ""
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> GitHubIssue:
        user = data.get("user", {})
        labels = [l.get("name", "") for l in data.get("labels", []) if isinstance(l, dict)]
        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            state=data.get("state", ""),
            url=data.get("html_url", ""),
            body=data.get("body") or "",
            author=user.get("login", "") if isinstance(user, dict) else "",
            labels=labels,
        )


@dataclass(frozen=True)
class GitHubWorkflowRun:
    id: int
    name: str
    status: str
    conclusion: Optional[str]
    url: str = ""
    head_branch: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> GitHubWorkflowRun:
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            status=data.get("status", ""),
            conclusion=data.get("conclusion"),
            url=data.get("html_url", ""),
            head_branch=data.get("head_branch", ""),
        )

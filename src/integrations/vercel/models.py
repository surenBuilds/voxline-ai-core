"""Vercel data models — typed representations of Vercel entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VercelDeploymentStatus(Enum):
    QUEUED = "queued"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"
    CANCELED = "canceled"


class VercelEnvironment(Enum):
    PREVIEW = "preview"
    PRODUCTION = "production"


@dataclass(frozen=True)
class VercelDomain:
    name: str
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> VercelDomain:
        return cls(
            name=data.get("name", ""),
            verified=data.get("verified", False),
        )


@dataclass(frozen=True)
class VercelProject:
    id: str
    name: str
    framework: Optional[str] = None
    link: Optional[Dict[str, Any]] = None
    domains: List[VercelDomain] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> VercelProject:
        domains = [VercelDomain.from_api(d) for d in data.get("targets", {}).get("production", {}).get("domains", []) if isinstance(d, dict)]
        link = data.get("link") or data.get("latestDeployments", [{}])[0].get("meta", {}).get("githubUrl") if data.get("latestDeployments") else None
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            framework=data.get("framework"),
            link=data.get("link") if isinstance(data.get("link"), dict) else None,
            domains=domains,
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
        )


@dataclass(frozen=True)
class VercelDeployment:
    id: str
    url: str
    name: str
    state: VercelDeploymentStatus
    environment: VercelEnvironment = VercelEnvironment.PREVIEW
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    created_at: str = ""
    ready_state: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> VercelDeployment:
        state_str = data.get("state", "QUEUED")
        try:
            state = VercelDeploymentStatus(state_str.lower())
        except ValueError:
            state = VercelDeploymentStatus.QUEUED

        env_str = data.get("target") or data.get("meta", {}).get("target", "preview")
        try:
            env = VercelEnvironment(env_str.lower())
        except ValueError:
            env = VercelEnvironment.PREVIEW

        meta = data.get("meta", {})
        return cls(
            id=data.get("uid", data.get("id", "")),
            url=data.get("url", ""),
            name=data.get("name", ""),
            state=state,
            environment=env,
            branch=meta.get("branch") or data.get("branch"),
            commit_sha=meta.get("githubCommitSha") or data.get("commit"),
            created_at=data.get("created", data.get("createdAt", "")),
            ready_state=data.get("readyState", ""),
        )

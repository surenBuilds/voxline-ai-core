"""Credential providers for external service authentication.

Design:
    - CodingAgent NEVER reads environment variables for tokens.
    - CredentialProvider is the ONLY source of credentials.
    - Tokens are NEVER logged, returned in API responses, or stored in code.
    - Tokens are NEVER passed to the LLM context.
"""

from abc import ABC, abstractmethod
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CredentialProvider(ABC):
    """Abstract source of service credentials."""

    @abstractmethod
    def get_token(self, service: str) -> Optional[str]:
        """Return the credential for a service, or None if unavailable."""

    @abstractmethod
    def is_available(self, service: str) -> bool:
        """Check if credentials are available for a service without exposing them."""

    @abstractmethod
    def redact(self, text: str) -> str:
        """Remove any credential traces from text for safe logging."""


class EnvironmentCredentialProvider(CredentialProvider):
    """Reads credentials from environment variables.

    Supported services and their env vars:
        - github: GITHUB_TOKEN
        - vercel: VERCEL_TOKEN
    """

    _ENV_MAP = {
        "github": "GITHUB_TOKEN",
        "vercel": "VERCEL_TOKEN",
    }

    def get_token(self, service: str) -> Optional[str]:
        env_key = self._ENV_MAP.get(service.lower())
        if not env_key:
            logger.warning("Unknown service for credential lookup: %s", service)
            return None
        token = os.environ.get(env_key)
        if not token:
            logger.debug("No credential found for %s (env: %s)", service, env_key)
        return token

    def is_available(self, service: str) -> bool:
        return self.get_token(service) is not None

    def redact(self, text: str) -> str:
        result = text
        for service, env_key in self._ENV_MAP.items():
            token = os.environ.get(env_key, "")
            if token and len(token) > 4:
                result = result.replace(token, f"***{service.upper()}***")
        return result

"""
OpenAI-compatible hosted provider.

A clean AIProvider adapter that talks to any OpenAI-compatible
``/chat/completions`` endpoint (OpenAI, or a vLLM/Ollama/LM Studio /
OpenAI-compatible gateway the developer points at).

Design / secrets:
    - The API key comes ONLY from the environment (``AI_API_KEY``) via
      ``VoxlineConfig`` (or is injected for tests). It is never hardcoded,
      never committed, never returned to a browser, and never logged.
    - ``health_check`` / ``chat`` errors never echo the key.
    - A missing key raises a clear, safe error instead of pretending to work.

This provider is client-side HTTP only (``httpx``) — it does NOT pull in
torch / transformers, so it can run inside a lean Vercel serverless function.
The existing local providers (``qwen``, ``native``) are untouched and continue
to work locally.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from src.config.settings import VoxlineConfig
from src.providers.base import (
    AIProvider,
    GenerationConfig,
    ModelInfo,
    ProviderHealth,
    ProviderStatus,
)


class OpenAICompatProvider(AIProvider):
    """AIProvider adapter for any OpenAI-compatible chat-completions API."""

    provider_id_value = "openai"

    def __init__(
        self,
        config: Optional[VoxlineConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 60.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        if config is None:
            from src.config.settings import get_config

            config = get_config()

        self._api_key = (
            api_key if api_key is not None else (config.ai_api_key or "")
        )
        self._base_url = (
            (base_url or config.ai_base_url or "https://api.openai.com/v1")
            .rstrip("/")
        )
        self._model = model or config.ai_model or "gpt-3.5-turbo"
        self._timeout = timeout_seconds
        if transport is not None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, transport=transport
            )
        else:
            self._client = httpx.AsyncClient(timeout=self._timeout)

    # ------------------------------------------------------------------
    # AIProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return self.provider_id_value

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def supports_streaming(self) -> bool:
        return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            provider_id=self.provider_id,
            model_type="api",
            supports_streaming=False,
            extra={"base_url": self._base_url, "endpoint": "chat/completions"},
        )

    def _require_key(self) -> None:
        """Raise a safe, generic error if no API key is configured."""
        if not self._api_key:
            raise ValueError(
                "Hosted provider not configured: AI_API_KEY is not set. "
                "Set AI_API_KEY (server-side only) to enable the hosted provider."
            )

    async def health_check(self) -> ProviderHealth:
        """Check connectivity/authorization without exposing the key."""
        if not self._api_key:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message="Hosted provider not configured (missing AI_API_KEY)",
            )
        start = time.time()
        try:
            # A minimal 1-token request verifies auth + connectivity.
            await self._chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            elapsed = (time.time() - start) * 1000
            return ProviderHealth(
                status=ProviderStatus.HEALTHY,
                message="Hosted provider reachable and authorized",
                response_time_ms=elapsed,
            )
        except httpx.HTTPStatusError as exc:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message=f"Hosted provider returned HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message=f"Hosted provider unreachable: {exc.request.url.host}",
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message=f"Hosted provider health check failed: {type(exc).__name__}",
            )

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate a completion from a bare prompt."""
        return await self.chat(
            [{"role": "user", "content": prompt}], config
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> str:
        """Generate a response from OpenAI-style messages (hosted)."""
        self._require_key()
        payload = await self._chat_completion(
            messages=messages,
            max_tokens=config.max_tokens or 100,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        return payload.get("choices", [{}])[0].get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            body["top_p"] = top_p

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """Close the underlying HTTP client (used by tests / lifecycle)."""
        await self._client.aclose()

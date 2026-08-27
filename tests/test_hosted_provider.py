"""Tests for the OpenAI-compatible hosted provider (src/providers/hosted.py).

Uses httpx.MockTransport so no real network / no API key is required.
Validates: request shape, response parsing, missing-key safety, and that
secrets never leak into errors.

Each provider is created AND closed inside a single event loop (asyncio.run),
because httpx.AsyncClient binds loop-bound resources that must not be reused
across loops.
"""

import asyncio

import httpx
import pytest

from src.providers.hosted import OpenAICompatProvider
from src.providers.base import GenerationConfig, ProviderStatus

EXAMPLE_KEY = "sk-test-thisikkeyneverleaks"
EXAMPLE_BASE = "https://example-llm.example.com/v1"
EXAMPLE_MODEL = "my-hosted-model"


def _ok_transport():
    """Return a MockTransport that answers chat/completions with one message."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == f"Bearer {EXAMPLE_KEY}"
        payload = {
            "id": "cmpl-test",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hello from hosted"},
                 "finish_reason": "stop"}
            ],
        }
        return httpx.Response(200, json=payload, request=request)
    return httpx.MockTransport(handler)


def _unauthorized_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "invalid key"}, request=request)
    return httpx.MockTransport(handler)


def _make(transport=None, api_key=EXAMPLE_KEY):
    return OpenAICompatProvider(
        api_key=api_key,
        base_url=EXAMPLE_BASE,
        model=EXAMPLE_MODEL,
        transport=transport,
        timeout_seconds=5.0,
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------


def test_provider_identity():
    async def t():
        p = _make(_ok_transport())
        try:
            assert p.provider_id == "openai"
            assert p.model_id == EXAMPLE_MODEL
            assert p.get_model_info().model_type == "api"
        finally:
            await p.aclose()
    run(t())


def test_chat_success_returns_content_and_sends_key():
    async def t():
        p = _make(_ok_transport())
        try:
            out = await p.chat([{"role": "user", "content": "hi"}], GenerationConfig(max_tokens=50))
            assert out == "hello from hosted"
        finally:
            await p.aclose()
    run(t())


def test_generate_delegates_to_chat():
    async def t():
        p = _make(_ok_transport())
        try:
            out = await p.generate("some prompt", GenerationConfig(max_tokens=50))
            assert out == "hello from hosted"
        finally:
            await p.aclose()
    run(t())


def test_health_check_healthy():
    async def t():
        p = _make(_ok_transport())
        try:
            health = await p.health_check()
            assert health.status == ProviderStatus.HEALTHY
            assert health.response_time_ms >= 0
        finally:
            await p.aclose()
    run(t())


def test_health_check_reports_http_error():
    async def t():
        p = _make(_unauthorized_transport())
        try:
            health = await p.health_check()
            assert health.status == ProviderStatus.UNAVAILABLE
            assert "403" in health.message
        finally:
            await p.aclose()
    run(t())


def test_missing_key_chat_raises_safe_error_and_never_leaks():
    async def t():
        p = _make(_ok_transport(), api_key="")
        try:
            with pytest.raises(ValueError) as excinfo:
                await p.chat([{"role": "user", "content": "x"}], GenerationConfig(max_tokens=1))
            msg = str(excinfo.value)
            assert "AI_API_KEY" in msg
            assert EXAMPLE_KEY not in msg
        finally:
            await p.aclose()
    run(t())


def test_missing_key_health_is_unavailable_and_never_leaks():
    async def t():
        p = _make(_ok_transport(), api_key="")
        try:
            health = await p.health_check()
            assert health.status == ProviderStatus.UNAVAILABLE
            assert EXAMPLE_KEY not in health.message
        finally:
            await p.aclose()
    run(t())


def test_error_never_contains_secret_on_unauthorized():
    async def t():
        p = _make(_unauthorized_transport())
        try:
            health = await p.health_check()
            assert EXAMPLE_KEY not in health.message
        finally:
            await p.aclose()
    run(t())

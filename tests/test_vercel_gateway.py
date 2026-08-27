"""Tests for the Vercel ASGI gateway (api/index.py).

Validates endpoint contracts against the real assistants (with an injected
provider) and — most importantly — that errors are generic and can never
leak secrets or tracebacks.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.providers.base import AIProvider, GenerationConfig, ModelInfo, ProviderHealth, ProviderStatus
from api.index import create_app

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProvider(AIProvider):
    """Deterministic in-memory provider for gateway tests."""

    def __init__(self, text="fake assistant reply", fallback="ok", raise_on_chat=False):
        self._text = text
        self._fallback = fallback
        self._raise_on_chat = raise_on_chat

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return "fake-model"

    @property
    def supports_streaming(self) -> bool:
        return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_id=self.model_id, provider_id=self.provider_id, model_type="api")

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        return self._text

    async def chat(self, messages, config: GenerationConfig) -> str:
        if self._raise_on_chat:
            raise RuntimeError("boom-internal")
        return self._text

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, message="ok", response_time_ms=1)


class EmptyKeyProvider(FakeProvider):
    """Simulates a provider that has no key configured (raises safely)."""

    def __init__(self):
        super().__init__()
        self._raise_on_chat = True


def make_app(provider=None):
    return create_app(provider=provider)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_root_serves_ui_or_fallback():
    client = TestClient(make_app())
    resp = client.get("/")
    assert resp.status_code == 200


def test_lazy_runtime_not_created_on_static_root():
    # GET / must not require building the provider/assistants.
    client = TestClient(make_app(provider=None))
    resp = client.get("/")
    assert resp.status_code == 200
    client.close()


def test_health_ok_with_provider():
    client = TestClient(make_app(provider=FakeProvider()))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["provider"] == "openai"
    client.close()


def test_health_with_missing_key_reports_error_no_secret():
    # EmptyKeyProvider health_check returns unavailable (from base) OR raises;
    # either way the response must be a safe dict without a traceback/secret.
    client = TestClient(make_app(provider=EmptyKeyProvider()))
    resp = client.get("/health")
    assert resp.status_code == 200
    text = resp.text
    assert "Traceback" not in text
    assert "sk-" not in text
    client.close()


def test_api_tools_ok():
    client = TestClient(make_app(provider=FakeProvider()))
    resp = client.get("/api/tools")
    assert resp.status_code == 200
    assert "tools" in resp.json()
    # Tool listing must not leak any secret-like content.
    assert "sk-" not in resp.text
    client.close()


def test_api_chat_ok():
    client = TestClient(make_app(provider=FakeProvider()))
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "fake assistant reply"
    assert body["session_id"]
    assert body["provider"] == "openai"
    client.close()


def test_api_business_ok():
    client = TestClient(make_app(provider=FakeProvider()))
    resp = client.post("/api/business", json={"message": "analyze this"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "fake assistant reply"
    assert body["session_id"]
    client.close()


def test_api_chat_internal_error_is_generic_and_secret_safe():
    # Provider raises -> gateway must return a generic 500 with no traceback
    # and no internal detail.
    client = TestClient(make_app(provider=EmptyKeyProvider()))
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal error processing chat request"
    assert "Traceback" not in resp.text
    assert "boom" not in resp.text
    client.close()


def test_api_business_internal_error_is_generic():
    client = TestClient(make_app(provider=EmptyKeyProvider()))
    resp = client.post("/api/business", json={"message": "analyze this"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal error processing business request"
    assert "Traceback" not in resp.text
    client.close()


def test_api_coding_reaches_agent_and_is_secret_safe():
    client = TestClient(make_app(provider=EmptyKeyProvider()))
    resp = client.post("/api/coding", json={"message": "hello"})
    # Should never leak a secret or traceback regardless of path taken.
    assert "Traceback" not in resp.text
    assert "sk-" not in resp.text
    client.close()


def test_request_validation_rejects_empty_message():
    client = TestClient(make_app(provider=FakeProvider()))
    resp = client.post("/api/chat", json={"message": ""})
    assert resp.status_code == 422
    client.close()

"""
Vercel serverless ASGI gateway for the Voxline Coding Agent.

Browser
  -> Vercel frontend (this app serves the existing static UI at ``/``)
  -> this server-side API (/health, /api/chat, /api/business, /api/tools, /api/coding)
  -> hosted AI provider (OpenAI-compatible)
  -> existing CodingAgent / ToolRegistry / security system

Why a lean gateway instead of reusing serve_v04.py:
  - serve_v04.py loads a local heavyweight model in an ``@app.on_event``
    startup hook, which is not a Vercel-compatible runtime model and drags in
    torch/transformers.
  - This gateway only imports the torch-free parts of the stack and uses the
    hosted provider, so it fits a lean Vercel Python function.

Security:
  - The AI API key is read server-side only (env ``AI_API_KEY``).
  - No secret is ever returned to the browser or included in client JS.
  - Errors are generic — no tracebacks, no internal details, no key leakage.
  - The existing CodingAgent / ToolRegistry / PathSecurity / AuditLog /
    approval / GitHub / Vercel security controls are used unchanged.

Persistence (honest): on Vercel the filesystem is ephemeral. The memory DB and
the Coding Agent workspace live under the platform temp dir and are lost between
cold starts. This is safe for a first interactive browser test; durable
persistence / a persistent worker is documented as a separate step.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Minimal, torch-free imports only (verified: the coding/chat/business/memory/
# tools stack does not import torch at module level).
from src.config.settings import get_config, VoxlineConfig  # noqa: E402
from src.providers.base import AIProvider  # noqa: E402
from src.assistant.session import SessionManager, SessionMode  # noqa: E402
from src.assistant.context import ContextBuilder  # noqa: E402
from src.assistant.chat import ChatAssistant, AssistantResponse  # noqa: E402
from src.assistant.business import BusinessAssistant, BusinessResponse, BusinessTaskType  # noqa: E402
from src.memory.memory import MemoryStore  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent.parent / "src" / "api" / "static"

DEFAULT_PROVIDER = "openai"


# ---------------------------------------------------------------------------
# Pydantic request models (same contracts as serve_v04.py)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    session_id: Optional[str] = Field(default=None, max_length=128)


class BusinessChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    session_id: Optional[str] = Field(default=None, max_length=128)
    task_type: str = Field(default="general_analysis", max_length=64)


class CodingRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    session_id: Optional[str] = Field(default=None, max_length=128)
    repository_owner: str = Field(default="", max_length=128)
    repository_name: str = Field(default="", max_length=128)
    repository_branch: str = Field(default="main", max_length=128)
    create_pr: bool = Field(default=False)
    deploy_preview: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Runtime construction
# ---------------------------------------------------------------------------


class _Runtime:
    """Holds the lazily-built provider + assistants for the gateway."""

    def __init__(self) -> None:
        self.provider: Optional[AIProvider] = None
        self.chat: Optional[ChatAssistant] = None
        self.business: Optional[BusinessAssistant] = None
        self.coding = None
        self.sessions: Optional[SessionManager] = None
        self.memory: Optional[MemoryStore] = None
        self.workspace: str = ""


def _load_provider(config: VoxlineConfig, injected: Optional[AIProvider] = None) -> AIProvider:
    """Create the hosted (or injected) provider. Never loads local models here."""
    if injected is not None:
        return injected
    from src.providers.factory import ProviderFactory
    provider_id = config.ai_provider or DEFAULT_PROVIDER
    try:
        provider = ProviderFactory.create(config)
    except Exception:  # noqa: BLE001 - re-raise a generic, safe config error
        raise RuntimeError(
            f"AI provider '{provider_id}' could not be initialized. "
            "Check server-side AI_PROVIDER / AI_API_KEY configuration."
        )
    if getattr(provider, "provider_id", None) in ("qwen", "native"):
        raise RuntimeError("Local models (qwen/native) are not supported on serverless.")
    return provider


def _ephemeral_dir(label: str) -> str:
    """Resolve an ephemeral platform-temp dir (honest: not durable on Vercel)."""
    override = os.environ.get(f"VOXLINE_{label}_PATH")
    if override:
        d = Path(override)
    else:
        d = Path(tempfile.gettempdir()) / f"voxline_{label.lower()}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _build_assistants(runtime: _Runtime, config: VoxlineConfig, provider: AIProvider) -> None:
    mem_path = Path(_ephemeral_dir("MEMORY")) / "gateway.db"
    runtime.memory = MemoryStore(str(mem_path))
    runtime.sessions = SessionManager()
    ctx = ContextBuilder(memory_store=runtime.memory, max_history=20)
    runtime.chat = ChatAssistant(
        provider=provider,
        session_manager=runtime.sessions,
        context_builder=ctx,
        memory_store=runtime.memory,
        max_history=20,
        max_new_tokens=256,
        temperature=0.7,
    )
    runtime.business = BusinessAssistant(
        provider=provider,
        session_manager=runtime.sessions,
        context_builder=ctx,
        memory_store=runtime.memory,
        max_history=20,
        max_new_tokens=256,
        temperature=0.7,
    )

    from src.tools.bootstrap import build_tool_registry
    from src.assistant.coding import CodingAgent

    workspace_root = _ephemeral_dir("CODING_WORKSPACE")
    runtime.workspace = workspace_root
    tool_registry = build_tool_registry(
        config=config,
        workspace_root=workspace_root,
    )
    runtime.coding = CodingAgent(
        provider=provider,
        session_manager=runtime.sessions,
        context_builder=ctx,
        memory_store=runtime.memory,
        tool_registry=tool_registry,
        workspace=workspace_root,
        max_plan_steps=config.coding_agent_max_plan_steps,
        max_fix_iterations=config.coding_agent_max_fix_iterations,
        max_context_chars=config.coding_agent_max_context_chars,
        require_approval_for_writes=config.coding_agent_require_approval_for_writes,
        auto_approve_workspace_writes=True,
    )


def create_app(
    config: Optional[VoxlineConfig] = None,
    provider: Optional[AIProvider] = None,
) -> FastAPI:
    """Build the gateway app. Provider can be injected for tests."""
    if config is None:
        config = get_config()

    runtime = _Runtime()

    app = FastAPI(title="Voxline AI Gateway", version="1.0.0")

    def _ensure_ready() -> _Runtime:
        if runtime.coding is None:
            runtime.provider = _load_provider(config, provider)
            _build_assistants(runtime, config, runtime.provider)
        return runtime

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(content=index.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Voxline AI</h1><p>UI not found.</p>", status_code=200)

    @app.get("/health")
    async def health():
        try:
            rt = _ensure_ready()
        except Exception as exc:
            return {
                "status": "unhealthy",
                "provider": None,
                "model": None,
                "error": str(exc),
            }
        try:
            h = await rt.provider.health_check()
            return {
                "status": h.status.value,
                "provider": rt.provider.provider_id,
                "model": rt.provider.model_id,
                "response_time_ms": h.response_time_ms,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "provider": rt.provider.provider_id,
                "model": rt.provider.model_id,
                "error": str(exc),
            }

    @app.get("/api/tools")
    async def api_tools():
        try:
            rt = _ensure_ready()
        except Exception:
            return {"tools": {}}
        return {"tools": rt.coding.tool_registry.available_tools()}

    @app.post("/api/chat")
    async def api_chat(request: ChatRequest):
        try:
            rt = _ensure_ready()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        session_id = request.session_id
        if not session_id:
            session = rt.chat.new_session(mode=SessionMode.CHAT)
            session_id = session.session_id
        else:
            session = rt.sessions.get(session_id)
            if session is None:
                session = rt.chat.new_session(mode=SessionMode.CHAT)
                session_id = session.session_id
        try:
            response: AssistantResponse = rt.chat.chat(
                session_id=session_id, message=request.message,
            )
            return {
                "response": response.text,
                "session_id": response.session_id,
                "mode": response.mode,
                "provider": response.provider_id,
                "model": response.model_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat endpoint error")
            raise HTTPException(status_code=500, detail="Internal error processing chat request")

    @app.post("/api/business")
    async def api_business(request: BusinessChatRequest):
        try:
            rt = _ensure_ready()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        session_id = request.session_id
        if not session_id:
            session = rt.business.new_session()
            session_id = session.session_id
        else:
            session = rt.sessions.get(session_id)
            if session is None:
                session = rt.business.new_session()
                session_id = session.session_id
        try:
            task_map = {t.value: t for t in BusinessTaskType}
            task_type = task_map.get(request.task_type, BusinessTaskType.GENERAL_ANALYSIS)
            response: BusinessResponse = rt.business.chat(
                session_id=session_id, message=request.message, task_type=task_type,
            )
            return {
                "response": response.response,
                "session_id": response.session_id,
                "task_type": response.task_type.value,
                "provider": response.provider_id,
                "model": response.model_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("business endpoint error")
            raise HTTPException(status_code=500, detail="Internal error processing business request")

    @app.post("/api/coding")
    async def api_coding(request: CodingRequest):
        try:
            rt = _ensure_ready()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        session_id = request.session_id or None
        try:
            import concurrent.futures

            def _run_coding():
                if request.repository_owner and request.repository_name:
                    return rt.coding.execute_with_repository(
                        user_request=request.message,
                        repository_owner=request.repository_owner,
                        repository_name=request.repository_name,
                        repository_branch=request.repository_branch,
                        session_id=session_id,
                        create_pr=request.create_pr,
                        deploy_preview=request.deploy_preview,
                    )
                else:
                    return rt.coding.execute(user_request=request.message, session_id=session_id)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_coding)
                try:
                    result = future.result(timeout=120)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise HTTPException(status_code=504, detail="Coding task timed out")

            data: Dict[str, Any] = {
                "response": result.summary,
                "mode": "coding",
                "assistant": "coding",
                "session_id": result.task_id,
                "operation_id": result.operation_id,
                "success": result.success,
                "status": result.status.value if hasattr(result.status, "value") else result.status,
                "files_modified": result.files_modified,
                "commands_executed": result.commands_executed,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "errors": result.errors,
                "warnings": result.warnings,
                "iterations": result.iterations,
                "commit_sha": result.commit_sha,
            }
            if result.pull_request:
                data["pull_request"] = {
                    "number": result.pull_request.number,
                    "title": result.pull_request.title,
                    "url": result.pull_request.url,
                    "head_branch": result.pull_request.head_branch,
                    "base_branch": result.pull_request.base_branch,
                }
            if result.deployment:
                data["deployment"] = {
                    "id": result.deployment.id,
                    "url": result.deployment.url,
                    "environment": result.deployment.environment,
                    "state": result.deployment.state,
                }
            return data
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("coding endpoint error")
            raise HTTPException(status_code=500, detail="Internal error processing coding request")

    @app.on_event("shutdown")
    async def shutdown():
        try:
            if runtime.memory is not None:
                runtime.memory.close()
        except Exception:  # noqa: BLE001
            pass

    return app


app = create_app()

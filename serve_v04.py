#!/usr/bin/env python3
"""Voxline AI v0.4 — Web UI Server

Launches the full Voxline assistant stack with a browser-accessible Chat UI.

Usage:
    python serve_v04.py                          # default (qwen)
    python serve_v04.py --provider qwen          # Qwen2.5-0.5B-Instruct
    python serve_v04.py --provider native        # native Voxline model
    python serve_v04.py --host 127.0.0.1 --port 8000
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional

from src.providers.base import AIProvider
from src.config.settings import VoxlineConfig, get_config
from src.assistant.session import SessionManager, SessionMode
from src.assistant.context import ContextBuilder
from src.assistant.chat import ChatAssistant, AssistantResponse
from src.assistant.business import (
    BusinessAssistant, BusinessResponse,
    BusinessTaskType,
)
from src.memory.memory import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "src" / "api" / "static"

app = FastAPI(title="Voxline AI v0.4", version="0.4.0")

_provider: Optional[AIProvider] = None
_chat_assistant: Optional[ChatAssistant] = None
_business_assistant: Optional[BusinessAssistant] = None
_coding_assistant = None
_session_manager: Optional[SessionManager] = None
_memory_store: Optional[MemoryStore] = None
_provider_name: str = "qwen"


# ---------------------------------------------------------------------------
# Pydantic request models
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
# Provider loaders
# ---------------------------------------------------------------------------

def load_qwen_provider() -> AIProvider:
    """Load Qwen provider from local model."""
    from src.providers.qwen_provider import QwenProvider
    model_path = "models/Qwen2.5-0.5B-Instruct"
    provider = QwenProvider(model_path=model_path, device="cpu")
    info = provider.get_model_info()
    logger.info("Qwen loaded: %s params", f"{info.parameters:,}" if info.parameters else "?")
    return provider


def load_native_provider() -> AIProvider:
    """Load native Voxline provider from checkpoint."""
    import torch
    import yaml
    from src.model.transformer import VoxlineTransformer
    from src.tokenizer.bpe import BPETokenizer
    from src.providers.local_voxline import LocalVoxlineProvider
    from src.config.model_config import ModelConfig

    config_path = Path("configs/model_configs.yaml")
    with open(config_path) as f:
        model_config_dict = yaml.safe_load(f)["v0_4_small"]["model"]

    checkpoint_dir = Path("checkpoints/v0_4")
    tokenizer = BPETokenizer(vocab_size=model_config_dict["vocab_size"])
    tokenizer.load(str(checkpoint_dir / "tokenizer.json"))

    model = VoxlineTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_config_dict["d_model"],
        num_layers=model_config_dict["num_layers"],
        num_heads=model_config_dict["num_heads"],
        d_ff=model_config_dict["d_ff"],
        max_seq_len=model_config_dict["max_seq_len"],
        dropout=model_config_dict["dropout"],
    )

    checkpoint = torch.load(
        checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_config = ModelConfig(
        model_type="voxline_transformer",
        model_version="0.4.0",
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_config_dict["d_model"],
        max_seq_len=model_config_dict["max_seq_len"],
        num_layers=model_config_dict["num_layers"],
        num_heads=model_config_dict["num_heads"],
        d_ff=model_config_dict["d_ff"],
        dropout=model_config_dict["dropout"],
    )

    provider = LocalVoxlineProvider(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        device="cpu",
    )
    n = model.get_num_parameters()
    logger.info("Native Voxline loaded: %s params", f"{n:,}")
    return provider


def load_provider(name: str) -> AIProvider:
    loaders = {"qwen": load_qwen_provider, "native": load_native_provider}
    if name not in loaders:
        raise ValueError(f"Unknown provider: {name}. Available: {list(loaders.keys())}")
    return loaders[name]()


# ---------------------------------------------------------------------------
# Assistant factory
# ---------------------------------------------------------------------------

def _build_assistants(provider: AIProvider) -> None:
    global _chat_assistant, _business_assistant, _coding_assistant
    global _session_manager, _memory_store
    _session_manager = SessionManager()
    _memory_store = MemoryStore("memory/voxline_web.db")
    ctx = ContextBuilder(memory_store=_memory_store, max_history=20)
    _chat_assistant = ChatAssistant(
        provider=provider,
        session_manager=_session_manager,
        context_builder=ctx,
        memory_store=_memory_store,
        max_history=20,
        max_new_tokens=256,
        temperature=0.7,
    )
    _business_assistant = BusinessAssistant(
        provider=provider,
        session_manager=_session_manager,
        context_builder=ctx,
        memory_store=_memory_store,
        max_history=20,
        max_new_tokens=256,
        temperature=0.7,
    )

    from src.tools.bootstrap import build_tool_registry
    from src.assistant.coding import CodingAgent
    config = get_config()
    tool_registry = build_tool_registry(config=config)
    _coding_assistant = CodingAgent(
        provider=provider,
        session_manager=_session_manager,
        context_builder=ctx,
        memory_store=_memory_store,
        tool_registry=tool_registry,
        max_plan_steps=config.coding_agent_max_plan_steps,
        max_fix_iterations=config.coding_agent_max_fix_iterations,
        max_context_chars=config.coding_agent_max_context_chars,
        require_approval_for_writes=config.coding_agent_require_approval_for_writes,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Voxline AI</h1><p>UI not found.</p>", status_code=200)


@app.get("/health")
async def health():
    if _provider is None:
        return {"status": "unhealthy", "provider": None, "model": None}
    try:
        h = await _provider.health_check()
        return {
            "status": h.status.value,
            "provider": _provider.provider_id,
            "model": _provider.model_id,
            "response_time_ms": h.response_time_ms,
        }
    except Exception as exc:
        return {"status": "error", "provider": _provider.provider_id, "model": _provider.model_id, "error": str(exc)}


@app.get("/api/tools")
async def api_tools():
    if _coding_assistant is None:
        return {"tools": {}}
    return {"tools": _coding_assistant.tool_registry.available_tools()}


@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    if _chat_assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not loaded")

    session_id = request.session_id
    if not session_id:
        session = _chat_assistant.new_session(mode=SessionMode.CHAT)
        session_id = session.session_id
    else:
        session = _session_manager.get(session_id)
        if session is None:
            session = _chat_assistant.new_session(mode=SessionMode.CHAT)
            session_id = session.session_id

    try:
        response: AssistantResponse = _chat_assistant.chat(
            session_id=session_id,
            message=request.message,
        )
        return {
            "response": response.text,
            "session_id": response.session_id,
            "mode": response.mode,
            "provider": response.provider_id,
            "model": response.model_id,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/business")
async def api_business(request: BusinessChatRequest):
    if _business_assistant is None:
        raise HTTPException(status_code=503, detail="Business assistant not loaded")

    session_id = request.session_id
    if not session_id:
        session = _business_assistant.new_session()
        session_id = session.session_id
    else:
        session = _session_manager.get(session_id)
        if session is None:
            session = _business_assistant.new_session()
            session_id = session.session_id

    try:
        task_map = {t.value: t for t in BusinessTaskType}
        task_type = task_map.get(request.task_type, BusinessTaskType.GENERAL_ANALYSIS)

        response: BusinessResponse = _business_assistant.chat(
            session_id=session_id,
            message=request.message,
            task_type=task_type,
        )
        return {
            "response": response.response,
            "session_id": response.session_id,
            "task_type": response.task_type.value,
            "provider": response.provider_id,
            "model": response.model_id,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/coding")
async def api_coding(request: CodingRequest):
    if _coding_assistant is None:
        raise HTTPException(status_code=503, detail="Coding assistant not loaded")

    session_id = request.session_id or None

    try:
        if request.repository_owner and request.repository_name:
            result = _coding_assistant.execute_with_repository(
                user_request=request.message,
                repository_owner=request.repository_owner,
                repository_name=request.repository_name,
                repository_branch=request.repository_branch,
                session_id=session_id,
                create_pr=request.create_pr,
                deploy_preview=request.deploy_preview,
            )
        else:
            result = _coding_assistant.execute(
                user_request=request.message,
                session_id=session_id,
            )

        response_data = {
            "response": result.summary,
            "mode": "coding",
            "assistant": "coding",
            "session_id": result.task_id,
            "success": result.success,
            "files_modified": result.files_modified,
            "commands_executed": result.commands_executed,
            "test_results": result.test_results,
            "errors": result.errors,
            "warnings": result.warnings,
            "iterations": result.iterations,
        }
        if result.pull_request:
            response_data["pull_request"] = {
                "number": result.pull_request.number,
                "title": result.pull_request.title,
                "url": result.pull_request.url,
                "head_branch": result.pull_request.head_branch,
                "base_branch": result.pull_request.base_branch,
            }
        if result.deployment:
            response_data["deployment"] = {
                "id": result.deployment.id,
                "url": result.deployment.url,
                "environment": result.deployment.environment,
                "state": result.deployment.state,
            }
        return response_data
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/api/integrations")
async def api_integrations():
    from src.integrations.credentials import EnvironmentCredentialProvider
    from src.config.settings import get_config
    config = get_config()
    creds = EnvironmentCredentialProvider()
    return {
        "github": {
            "enabled": config.github_enabled,
            "authenticated": creds.is_available("github"),
        },
        "vercel": {
            "enabled": config.vercel_enabled,
            "authenticated": creds.is_available("vercel"),
        },
    }


@app.on_event("startup")
async def startup():
    global _provider, _provider_name
    _provider = load_provider(_provider_name)
    _build_assistants(_provider)


@app.on_event("shutdown")
async def shutdown():
    if _memory_store:
        _memory_store.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BANNER = """
========================================
VOXLINE AI
========================================
Status: RUNNING
Provider: {provider}
Model: {model}

Local URL:
http://{host}:{port}

Browser:
http://{host}:{port}

========================================
"""


def main():
    global _provider_name

    parser = argparse.ArgumentParser(description="Voxline AI v0.4 Server")
    parser.add_argument(
        "--provider", choices=["native", "qwen"], default="qwen",
        help="AI provider to use (default: qwen)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _provider_name = args.provider

    import uvicorn

    def _print_banner():
        prov = _provider
        provider_name = prov.provider_id if prov else _provider_name
        model_name = prov.model_id if prov else "loading..."
        print(BANNER.format(provider=provider_name, model=model_name, host=args.host, port=args.port))

    @app.on_event("startup")
    async def _banner():
        _print_banner()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

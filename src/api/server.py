"""Local HTTP API for the Voxline Business Agent.

Run locally with:
    uvicorn src.api.server:app --host 127.0.0.1 --port 8000
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.business.agent import BusinessAgent
from src.config.settings import VoxlineConfig
from src.memory.memory import MemoryStore
from src.providers.base import GenerationConfig
from src.providers.local_transformers import LocalTransformersProvider


class KnowledgeRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    tags: List[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=2_000)
    context: Optional[str] = Field(default=None, max_length=10_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    max_tokens: int = Field(default=256, ge=16, le=1_024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


def create_app(memory_path: str = "memory/voxline_business.db", config: Optional[VoxlineConfig] = None) -> FastAPI:
    """Create an app instance; a separate factory keeps tests isolated."""
    app = FastAPI(title="Voxline Business Agent", version="0.1.0")
    memory_store = MemoryStore(memory_path)
    agent = BusinessAgent(memory_store)
    config = config or VoxlineConfig()
    provider: Optional[LocalTransformersProvider] = None
    provider_error: Optional[str] = None

    def get_provider() -> LocalTransformersProvider:
        """Load the heavy local model only when the first chat request arrives."""
        nonlocal provider, provider_error
        if provider is not None:
            return provider
        if config.ai_provider != "local_hf":
            provider_error = "AI_PROVIDER must be local_hf. Hosted AI providers are disabled."
            raise RuntimeError(provider_error)
        try:
            provider = LocalTransformersProvider(config.ai_model_path, config.ai_device)
            provider_error = None
            return provider
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            provider_error = str(error)
            raise RuntimeError(provider_error) from error

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "healthy",
            "mode": "local-first",
            "external_ai_api": False,
            "model": provider.model_id if provider else None,
            "model_status": "loaded" if provider else "loads on first chat request",
            "detail": provider_error,
        }

    @app.post("/chat")
    async def chat(request: ChatRequest) -> dict:
        try:
            active_provider = get_provider()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        knowledge = agent.search_knowledge(request.message, limit=3)
        context = "\n".join(f"- {item['content']}" for item in knowledge)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Voxline, the user's private business AI. Reply in the user's language. "
                    "Be accurate, practical and concise. Do not claim to have performed actions you did not perform."
                ),
            },
        ]
        if context:
            messages.append({"role": "system", "content": f"Relevant local business knowledge:\n{context}"})
        messages.append({"role": "user", "content": request.message})
        response = active_provider.generate_chat(
            messages,
            GenerationConfig(max_tokens=request.max_tokens, temperature=request.temperature, top_p=0.9),
        )
        agent.remember(request.message, tags=["conversation", "user"])
        agent.remember(response, tags=["conversation", "assistant"])
        return {"response": response, "model": active_provider.model_id, "knowledge_items_used": len(knowledge)}

    @app.post("/knowledge")
    def add_knowledge(request: KnowledgeRequest) -> dict:
        return {"id": agent.remember(request.content, request.tags)}

    @app.get("/knowledge/search")
    def search_knowledge(query: str, limit: int = 5) -> dict:
        return {"items": agent.search_knowledge(query, max(1, min(limit, 20)))}

    @app.post("/plans")
    def create_plan(request: PlanRequest) -> dict:
        return agent.create_plan(request.goal, request.context).to_dict()

    @app.get("/plans/{plan_id}")
    def get_plan(plan_id: str) -> dict:
        plan = agent.get_plan(plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="Plan not found")
        return plan.to_dict()

    @app.on_event("shutdown")
    def close_memory() -> None:
        memory_store.close()

    return app


app = create_app()

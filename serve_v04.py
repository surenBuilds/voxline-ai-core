#!/usr/bin/env python3
"""Voxline AI v0.4 — API Server

Provider-configurable API server.

Usage:
    python serve_v04.py                          # default (native)
    python serve_v04.py --provider native        # native Voxline model
    python serve_v04.py --provider qwen          # Qwen2.5-0.5B-Instruct
    python serve_v04.py --host 0.0.0.0 --port 8000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from src.providers.base import AIProvider, GenerationConfig, ModelInfo
from src.providers.factory import ProviderFactory
from src.config.settings import VoxlineConfig

app = FastAPI(title="Voxline AI v0.4", version="0.4.0")

_provider: Optional[AIProvider] = None
_provider_name: str = "native"


def load_native_provider() -> AIProvider:
    """Load native Voxline provider from checkpoint."""
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
    print(f"Native Voxline loaded: {n:,} params")
    return provider


def load_qwen_provider() -> AIProvider:
    """Load Qwen provider from local model."""
    from src.providers.qwen_provider import QwenProvider

    model_path = "models/Qwen2.5-0.5B-Instruct"
    provider = QwenProvider(model_path=model_path, device="cpu")
    print(f"Qwen loaded: {provider.get_model_info().parameters:,} params")
    return provider


def load_provider(provider_name: str) -> AIProvider:
    """Load the requested provider."""
    loaders = {
        "native": load_native_provider,
        "qwen": load_qwen_provider,
    }
    if provider_name not in loaders:
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {list(loaders.keys())}"
        )
    return loaders[provider_name]()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=5000)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


@app.on_event("startup")
def startup():
    global _provider, _provider_name
    _provider = load_provider(_provider_name)


@app.get("/health")
async def health():
    if _provider is None:
        return {"status": "unhealthy", "model_loaded": False}
    h = await _provider.health_check()
    return {
        "status": h.status.value,
        "version": "0.4.0",
        "model_loaded": True,
        "provider": _provider.provider_id,
        "model": _provider.model_id,
        "response_time_ms": h.response_time_ms,
    }


@app.get("/models")
async def models():
    if _provider is None:
        raise HTTPException(status_code=503, detail="No provider loaded")
    info = _provider.get_model_info()
    return {
        "provider": info.provider_id,
        "model_id": info.model_id,
        "model_type": info.model_type,
        "parameters": info.parameters,
        "vocab_size": info.vocab_size,
        "max_context_length": info.max_context_length,
        "device": info.device,
        "supports_streaming": info.supports_streaming,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if _provider is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=0.9,
        do_sample=True,
    )

    messages = [{"role": "user", "content": request.message}]
    response = await _provider.chat(messages, config)

    return {"response": response, "model": _provider.model_id}


@app.post("/generate")
async def generate(request: GenerateRequest):
    if _provider is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=True,
    )
    text = await _provider.generate(request.prompt, config)
    return {"text": text, "model": _provider.model_id}


def main():
    global _provider_name

    parser = argparse.ArgumentParser(description="Voxline AI v0.4 Server")
    parser.add_argument(
        "--provider",
        choices=["native", "qwen"],
        default="native",
        help="AI provider to use (default: native)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _provider_name = args.provider

    import uvicorn
    print(f"Starting Voxline AI v0.4 server on {args.host}:{args.port}")
    print(f"Provider: {_provider_name}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Voxline AI v0.4 — API Server

Usage:
    python serve_v04.py
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
from typing import Optional

from src.model.transformer import VoxlineTransformer
from src.tokenizer.bpe import BPETokenizer
from src.inference.generator import TextGenerator, GenerationConfig

app = FastAPI(title="Voxline AI v0.4", version="0.4.0")

_model = None
_tokenizer = None
_generator = None
_model_config = None


def load_model():
    global _model, _tokenizer, _generator, _model_config

    config_path = Path("configs/model_configs.yaml")
    with open(config_path) as f:
        _model_config = yaml.safe_load(f)["v0_4_small"]["model"]

    checkpoint_dir = Path("checkpoints/v0_4")

    _tokenizer = BPETokenizer(vocab_size=_model_config["vocab_size"])
    _tokenizer.load(str(checkpoint_dir / "tokenizer.json"))

    _model = VoxlineTransformer(
        vocab_size=_tokenizer.get_vocab_size(),
        d_model=_model_config["d_model"],
        num_layers=_model_config["num_layers"],
        num_heads=_model_config["num_heads"],
        d_ff=_model_config["d_ff"],
        max_seq_len=_model_config["max_seq_len"],
        dropout=_model_config["dropout"],
    )

    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=False)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    _generator = TextGenerator(_model, _tokenizer, device="cpu")
    print(f"Model loaded: {sum(p.numel() for p in _model.parameters()):,} params")


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
    load_model()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "0.4.0",
        "model_loaded": _model is not None,
        "params": sum(p.numel() for p in _model.parameters()) if _model else 0,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = f"User: {request.message}\nAssistant:"
    config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=0.9,
        do_sample=True,
        eos_token_id=_tokenizer.vocab.get("<EOS>"),
        pad_token_id=_tokenizer.vocab.get("<PAD>"),
    )

    # Generate using token IDs to extract only new tokens
    prompt_ids = _tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    output_ids = _generator._generate_greedy(input_ids, config)
    new_token_ids = output_ids[0, len(prompt_ids):].tolist()
    response = _tokenizer.decode(new_token_ids)

    # Clean response
    for marker in ["Assistant:", "User:", "USER:"]:
        idx = response.find(marker)
        if idx != -1:
            response = response[idx + len(marker):]
    for token in ["<EOS>", "<BOS>", "<PAD>", "<UNK>", "<CLS>", "<SEP>"]:
        response = response.replace(token, "")
    response = response.strip()

    return {"response": response, "model": "voxline-v0.4"}


@app.post("/generate")
def generate(request: GenerateRequest):
    if _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=True,
        eos_token_id=_tokenizer.vocab.get("<EOS>"),
        pad_token_id=_tokenizer.vocab.get("<PAD>"),
    )
    text = _generator.generate(request.prompt, config, return_text=True)
    return {"text": text, "model": "voxline-v0.4"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    print(f"Starting Voxline AI v0.4 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Voxline AI Core - Text Generation Script

Usage:
    python generate.py --model checkpoints/best_model.pt --prompt "Voxline"
    python generate.py --model checkpoints/best_model.pt --prompt "The future of AI" --temperature 0.7
"""

import argparse
import sys
import torch
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import VoxlineTransformer
from src.inference.generator import TextGenerator, GenerationConfig


def load_checkpoint(model_path: str, tokenizer_path: str, device: str):
    """Load model and tokenizer from checkpoints."""
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint with training info
        model_state = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        # Just model state
        model_state = checkpoint
        config = {}

    # Recreate model
    model = VoxlineTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.get("d_model", 768),
        num_layers=config.get("num_layers", 12),
        num_heads=config.get("num_heads", 12),
        d_ff=config.get("d_ff", 3072),
        max_seq_len=config.get("max_seq_len", 2048),
    )

    model.load_state_dict(model_state)
    model = model.to(device)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with Voxline")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Infer tokenizer path if not provided
    if not args.tokenizer:
        checkpoint_dir = Path(args.model).parent
        for name in ["tokenizer.json", "voxline_tokenizer.json"]:
            if (checkpoint_dir / name).exists():
                args.tokenizer = checkpoint_dir / name
                break
        else:
            args.tokenizer = checkpoint_dir / "tokenizer.json"

    print("=" * 70)
    print("VOXLINE AI CORE - TEXT GENERATION")
    print("=" * 70)

    # Load model and tokenizer
    print(f"\nLoading model from {args.model}...")
    model, tokenizer = load_checkpoint(args.model, str(args.tokenizer), args.device)
    print(f"  Model loaded on {args.device}")

    # Create generator
    generator = TextGenerator(model, tokenizer, device=args.device)
    generator.set_seed(args.seed)

    # Create generation config
    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    # Generate
    print(f"\nGenerating from prompt: '{args.prompt}'")
    print("-" * 70)

    generated = generator.generate(args.prompt, config, return_text=True)

    # Clean special tokens
    for tok in ["<EOS>", "<BOS>", "<PAD>", "<UNK>", "<CLS>", "<SEP>"]:
        generated = generated.replace(tok, "")

    print(generated.strip())
    print("-" * 70)


if __name__ == "__main__":
    main()

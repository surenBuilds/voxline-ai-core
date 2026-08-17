#!/usr/bin/env python
"""
Voxline AI Core - Interactive Chat

Usage:
    python chat.py --model checkpoints/best_model.pt
    python chat.py --model checkpoints/best_model.pt --armenian
"""

import argparse
import torch
from pathlib import Path

from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import VoxlineTransformer
from src.api.chat import ConversationalAI
from src.memory.memory import MemoryStore


def load_checkpoint(model_path: str, tokenizer_path: str, device: str):
    """Load model and tokenizer from checkpoints."""
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        model_state = checkpoint
        config = {}

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
    parser = argparse.ArgumentParser(description="Chat with Voxline")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer checkpoint")
    parser.add_argument("--memory", type=str, help="Path to memory database")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--armenian", action="store_true", help="Enable Armenian language mode")

    args = parser.parse_args()

    # Infer paths if not provided
    if not args.tokenizer:
        checkpoint_dir = Path(args.model).parent
        args.tokenizer = checkpoint_dir / "voxline_tokenizer.json"

    if not args.memory:
        args.memory = "memory/chat_memory.db"

    print("=" * 70)
    print("VOXLINE AI CORE - INTERACTIVE CHAT")
    print("=" * 70)
    print("\nType 'exit' to quit, 'clear' to clear conversation, 'memory' to view memories")

    # Load model and tokenizer
    print(f"\nLoading model from {args.model}...")
    model, tokenizer = load_checkpoint(args.model, str(args.tokenizer), args.device)

    # Initialize memory
    memory_store = MemoryStore(args.memory)

    # Create chat system
    system_instruction = (
        "You are Voxline, a helpful and respectful AI assistant built with Voxline AI Core. "
        "You are able to have natural conversations, answer questions, and help with tasks. "
        "Be concise and direct in your responses."
    )

    if args.armenian:
        system_instruction += (
            " You can communicate fluently in both English and Armenian. "
            "When the user writes in Armenian, respond in Armenian."
        )

    chat = ConversationalAI(
        model,
        tokenizer,
        memory_store=memory_store,
        device=args.device,
        system_instruction=system_instruction,
    )

    print("\nReady for conversation. Type messages and press Enter.")
    print("-" * 70 + "\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                chat.clear_conversation()
                print("Conversation cleared.\n")
                continue

            if user_input.lower() == "memory":
                recent = chat.get_memory_store().get_recent_memories(limit=5)
                if recent:
                    print("\nRecent memories:")
                    for mem in recent:
                        print(f"  - {mem.content[:50]}...")
                else:
                    print("\nNo memories yet.")
                print()
                continue

            # Generate response
            print(f"\nVoxline: ", end="", flush=True)
            response = chat.chat(
                user_input,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

    # Clean up
    memory_store.close()


if __name__ == "__main__":
    main()

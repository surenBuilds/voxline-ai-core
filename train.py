#!/usr/bin/env python
"""
Voxline AI Core - Training Script

Usage:
    python train.py --config configs/small.yaml
    python train.py --vocab-size 50000 --num-layers 6
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import VoxlineTransformer
from src.training.trainer import (
    LanguageModelDataset,
    Trainer,
    TrainingConfig,
    collate_batch,
)
from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_corpus(data_dir: str = "data") -> list:
    """Load text corpus."""
    texts = []
    for file_path in Path(data_dir).glob("*.txt"):
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Train Voxline Language Model")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        model_config = config_dict.get("model", {})
        train_config = config_dict.get("training", {})
    else:
        model_config = {}
        train_config = {}

    # Create training config
    config = TrainingConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        **{k: v for k, v in model_config.items()},
        **{k: v for k, v in train_config.items()},
    )

    print("=" * 70)
    print("VOXLINE AI CORE - TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")

    # Load corpus
    print(f"\nLoading corpus from {args.data_dir}...")
    texts = load_corpus(args.data_dir)
    print(f"  Loaded {len(texts)} text samples")

    # Train tokenizer
    print(f"\nTraining tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.fit(texts, num_merges=config.vocab_size - 4)
    print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

    # Save tokenizer
    tokenizer_path = Path(config.checkpoint_dir) / "voxline_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"  Saved to {tokenizer_path}")

    # Create datasets
    print(f"\nCreating datasets...")
    train_texts = texts[: int(len(texts) * 0.8)]
    val_texts = texts[int(len(texts) * 0.8) :]

    train_dataset = LanguageModelDataset(
        train_texts,
        tokenizer,
        max_seq_len=512,
    )
    val_dataset = LanguageModelDataset(
        val_texts,
        tokenizer,
        max_seq_len=512,
    )

    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Create model
    print(f"\nCreating model...")
    model = VoxlineTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads or min(12, config.d_model // 64),
        d_ff=config.d_ff or config.d_model * 4,
        max_seq_len=512,
    )

    num_params = model.get_num_parameters()
    print(f"  Parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(model, config, tokenizer)

    # Train
    print(f"\nStarting training...")
    trainer.train(train_loader, val_loader, args.num_epochs)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

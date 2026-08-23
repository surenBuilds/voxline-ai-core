#!/usr/bin/env python3
"""Voxline AI v0.4 — Small Model Training Script

Trains a VoxlineTransformer with the v0.4 small config on the bilingual corpus.

Usage:
    python scripts/train_small.py
"""

import sys
import os
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.model.transformer import VoxlineTransformer
from src.tokenizer.bpe import BPETokenizer
from src.training.trainer import TrainingConfig, LanguageModelDataset, Trainer, collate_batch
from torch.utils.data import DataLoader, random_split


def main():
    # Load config
    config_path = Path("configs/model_configs.yaml")
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)

    cfg = all_configs["v0_4_small"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    print("=" * 60)
    print("Voxline AI v0.4 — Small Model Training")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Load corpus
    corpus_path = Path("data/voxline_corpus.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"Corpus: {len(texts)} lines")

    arm_count = sum(1 for t in texts if any("\u0561" <= c <= "\u0587" for c in t))
    eng_count = len(texts) - arm_count
    print(f"  Armenian: {arm_count} lines")
    print(f"  English:  {eng_count} lines")
    print()

    # Train BPE tokenizer (skip if already exists)
    tokenizer_path = Path("checkpoints/v0_4/tokenizer.json")
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer_path.exists():
        print("Loading existing tokenizer...")
        tokenizer = BPETokenizer(vocab_size=model_cfg["vocab_size"])
        tokenizer.load(str(tokenizer_path))
        if len(tokenizer.merges_list) > 1500:
            print("  Too many merges (%d), retraining..." % len(tokenizer.merges_list))
            tokenizer = BPETokenizer(vocab_size=model_cfg["vocab_size"])
            bpe_merges = train_cfg.get("bpe_merges", 1000)
            tokenizer.fit(texts, num_merges=bpe_merges)
            tokenizer.save(str(tokenizer_path))
    else:
        print("Training BPE tokenizer...")
        bpe_merges = train_cfg.get("bpe_merges", 1000)
        tokenizer = BPETokenizer(vocab_size=model_cfg["vocab_size"])
        tokenizer.fit(texts, num_merges=bpe_merges)
        tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved: {tokenizer_path}")
    print(f"  Actual vocab size: {tokenizer.get_vocab_size()}")
    print()

    # Create dataset (with caching)
    max_seq_len = model_cfg["max_seq_len"]
    cache_path = Path("checkpoints/v0_4/dataset_cache.pt")

    if cache_path.exists():
        print("Loading cached dataset...")
        cached = torch.load(cache_path, weights_only=False)
        all_tokens = cached["all_tokens"]
        print(f"  Cached tokens: {len(all_tokens)}")

        # Rebuild examples from cached tokens
        examples = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            example = all_tokens[i : i + max_seq_len + 1]
            if len(example) == max_seq_len + 1:
                examples.append(example)

        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, examples):
                self.examples = examples
            def __len__(self):
                return len(self.examples)
            def __getitem__(self, idx):
                ex = self.examples[idx]
                return torch.tensor(ex[:-1], dtype=torch.long), torch.tensor(ex[1:], dtype=torch.long)

        dataset = CachedDataset(examples)
    else:
        print("Tokenizing corpus (this may take a moment)...")
        import time
        t0 = time.time()
        dataset = LanguageModelDataset(texts, tokenizer, max_seq_len=max_seq_len)
        print(f"  Done in {time.time()-t0:.1f}s")

        # Cache the tokens for faster reloads
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            bos_id = tokenizer.token_to_id("<BOS>")
            eos_id = tokenizer.token_to_id("<EOS>")
            if bos_id is not None:
                all_tokens.append(bos_id)
            all_tokens.extend(tokens)
            if eos_id is not None:
                all_tokens.append(eos_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"all_tokens": all_tokens}, cache_path)
        print(f"  Dataset cached to {cache_path}")
    print(f"Dataset: {len(dataset)} examples (max_seq_len={max_seq_len})")

    if len(dataset) == 0:
        print("ERROR: No training examples. Corpus too short for max_seq_len.")
        sys.exit(1)

    # Split train/val (90/10)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"  Train: {train_size} examples")
    print(f"  Val:   {val_size} examples")
    print()

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Build model
    print("Building model...")
    model = VoxlineTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=max_seq_len,
        dropout=model_cfg["dropout"],
    )

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,} total, {trainable_params:,} trainable")
    print()

    # Configure trainer
    training_config = TrainingConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=max_seq_len,
        dropout=model_cfg["dropout"],
        batch_size=batch_size,
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_steps=train_cfg.get("save_steps", 100),
        checkpoint_dir="checkpoints/v0_4",
        num_epochs=train_cfg.get("num_epochs", 10),
        patience=train_cfg.get("patience", 3),
    )
    training_config.save("checkpoints/v0_4/config.json")

    trainer = Trainer(model, training_config, tokenizer=tokenizer)

    # Train
    print("Starting training...")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Patience: {training_config.patience}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Warmup steps: {training_config.warmup_steps}")
    print()

    trainer.train(train_loader, val_loader)

    # Final evaluation
    print()
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_ppl = trainer.compute_perplexity(val_loader)
    print(f"Final validation perplexity: {final_ppl:.4f}")

    # Save training history
    import json
    history_path = Path("checkpoints/v0_4/training_history.json")
    with open(history_path, "w") as f:
        json.dump(trainer.training_history, f, indent=2)
    print(f"Training history saved: {history_path}")
    print()
    print("Training complete!")


if __name__ == "__main__":
    main()

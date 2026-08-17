"""
Training infrastructure for Voxline AI Core

Includes:
- Dataset handling
- Training loop
- Validation
- Checkpointing
- Metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import json
import math
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Training configuration."""

    vocab_size: int = 50000
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Evaluation
    eval_steps: int = 500
    eval_batch_size: int = 32

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_steps: int = 500

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save(self, path: str):
        """Save config to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class LanguageModelDataset(Dataset):
    """Dataset for language modeling."""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_seq_len: int = 2048,
        pad_id: int = 0,
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            pad_id: Padding token ID
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        # Create examples by sliding window
        self.examples = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            example = all_tokens[i : i + max_seq_len + 1]
            if len(example) == max_seq_len + 1:
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = torch.tensor(example[:-1], dtype=torch.long)
        target_ids = torch.tensor(example[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate batch with padding."""
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)

    # Pad sequences
    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        padded_inputs.append(torch.nn.functional.pad(inp, (0, pad_len), value=0))
        padded_targets.append(torch.nn.functional.pad(tgt, (0, pad_len), value=-100))

    return torch.stack(padded_inputs), torch.stack(padded_targets)


class Trainer:
    """Training manager."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer=None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            tokenizer: Tokenizer instance
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits = self.model(input_ids)

            # Compute loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.6f} | Step: {self.global_step}"
                )

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def compute_perplexity(self, val_loader: DataLoader) -> float:
        """Compute perplexity on validation set."""
        val_loss = self.validate(val_loader)
        perplexity = math.exp(val_loss)
        return perplexity

    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": asdict(self.config),
            "training_history": self.training_history,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

        if is_best:
            best_path = str(Path(path).parent / "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to load checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.training_history = checkpoint.get("training_history", [])
        print(f"Checkpoint loaded from {path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)
            perplexity = self.compute_perplexity(val_loader)

            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "perplexity": perplexity,
                "global_step": self.global_step,
            }
            self.training_history.append(metrics)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Perplexity: {perplexity:.4f}")

            # Save checkpoint
            checkpoint_path = (
                Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            )
            self.save_checkpoint(str(checkpoint_path))

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
                self.save_checkpoint(str(best_path), is_best=True)

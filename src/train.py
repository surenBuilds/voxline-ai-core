from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import TrainingConfig
from .dataset import TextDataset
from .model import VoxlineModel


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(config: TrainingConfig):
    set_seed(config.seed)
    config.resolve_paths()

    dataset = TextDataset(config.data_dir, sequence_length=config.max_seq_length)
    if len(dataset) == 0:
        raise ValueError(f"No training data found in {config.data_dir}")

    device = torch.device(config.device)
    model = VoxlineModel(vocab_size=config.vocab_size, max_seq_length=config.max_seq_length)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            inputs = batch["input_ids"]
            labels = batch["labels"]

            input_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(input_tensor)
            loss = criterion(logits.view(-1, logits.size(-1)), label_tensor.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        checkpoint_path = Path(config.checkpoint_dir) / f"voxline_epoch_{epoch + 1}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "config": config,
        }, checkpoint_path)

    return model

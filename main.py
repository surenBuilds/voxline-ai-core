from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import SimpleTokenizer


# ============================================================
# VOXLINE AI CORE v0.3
# Text Dataset + Vocabulary + Tokenizer + Train/Val Split + Token Prediction
# ============================================================


class TextCorpus:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.texts = []
        self.load()

    def load(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for file_path in sorted(self.data_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line:
                    self.texts.append(line)

        if not self.texts:
            raise ValueError(f"No text corpus found in {self.data_dir}")


class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=8, pad_id=0):
        self.samples = []
        self.pad_id = pad_id

        for text in texts:
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                continue
            for i in range(len(token_ids) - 1):
                context = token_ids[max(0, i - seq_length + 1): i + 1]
                target = token_ids[i + 1]
                while len(context) < seq_length:
                    context = [pad_id] + context
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class NextTokenModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, seq_length=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(seq_length, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, vocab_size),
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        x = self.embedding(input_ids) + self.position(positions)
        pooled = x.mean(dim=1)
        return self.net(pooled)


def collate_fn(batch):
    contexts = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_len = max(len(x) for x in contexts)

    padded = []
    for context in contexts:
        context_list = context.tolist() if isinstance(context, torch.Tensor) else list(context)
        pad_len = max_len - len(context_list)
        padded.append(context_list + [0] * pad_len)

    return torch.tensor(padded, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = TextCorpus("data")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(corpus.texts)
    tokenizer.save("checkpoints/voxline_tokenizer_v0_3.json")

    train_texts = corpus.texts[:-2]
    val_texts = corpus.texts[-2:]

    train_dataset = NextTokenDataset(train_texts, tokenizer, seq_length=8, pad_id=tokenizer.vocab["<PAD>"])
    val_dataset = NextTokenDataset(val_texts, tokenizer, seq_length=8, pad_id=tokenizer.vocab["<PAD>"])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = NextTokenModel(vocab_size=len(tokenizer.vocab), seq_length=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("=" * 70)
    print("VOXLINE AI CORE v0.3")
    print("=" * 70)
    print("Text Dataset -> Vocabulary -> Tokenizer -> Encode -> Decode -> Train/Validation Split -> Token Prediction Dataset")

    sample_text = "Voxline creates language models"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nSample text: {sample_text}")
    print(f"Encoded tokens: {encoded}")
    print(f"Decoded text: {decoded}")

    best_val_loss = float("inf")
    print("\nStarting token prediction training...\n")

    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)

        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, "checkpoints/voxline_ai_v0_3.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/50 | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")

    model.eval()
    prompt = "Voxline creates"
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > 8:
        input_ids = input_ids[-8:]
    else:
        input_ids = [tokenizer.vocab["<PAD>"]] * (8 - len(input_ids)) + input_ids

    with torch.no_grad():
        logits = model(torch.tensor([input_ids], dtype=torch.long).to(device))
        top_idx = torch.argmax(logits, dim=1).item()
        predicted_token = tokenizer.inv_vocab.get(top_idx, "<UNK>")

    print(f"\nNext-token inference on prompt: '{prompt}'")
    print(f"Predicted next token: {predicted_token}")
    print("\n" + "=" * 70)
    print("VOXLINE AI CORE v0.3 READY")
    print("=" * 70)


if __name__ == "__main__":
    main()

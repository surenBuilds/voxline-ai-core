from pathlib import Path
from typing import Iterable, List, Tuple


class TextDataset:
    """Minimal dataset wrapper for plain-text training data."""

    def __init__(self, data_dir: str, sequence_length: int = 1024):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.examples: List[str] = []
        self._load()

    def _load(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        text_files = sorted(self.data_dir.glob("**/*.*"))
        for file_path in text_files:
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if content.strip():
                        self.examples.append(content)
                except Exception:
                    continue

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for text in self.examples:
            yield text

    def collate(self, batch: List[str], tokenizer):
        encoded = [tokenizer.encode(item) for item in batch]
        input_ids = []
        labels = []

        for ids in encoded:
            if len(ids) > self.sequence_length:
                ids = ids[: self.sequence_length]
            if len(ids) < self.sequence_length:
                pad_len = self.sequence_length - len(ids)
                ids = ids + [0] * pad_len
            input_ids.append(ids)
            labels.append(ids[:])

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

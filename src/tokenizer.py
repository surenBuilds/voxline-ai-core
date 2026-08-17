import json
import re
from pathlib import Path


class SimpleTokenizer:
    """Simple word-level tokenizer for early language modeling milestones."""

    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab = {}
        self.inv_vocab = {}
        self.pattern = re.compile(r"\w+|[.,!?;:()\"'/-]")

    def tokenize(self, text: str):
        text = text.strip()
        if not text:
            return []
        tokens = self.pattern.findall(text)
        return [token for token in tokens if token.strip()]

    def fit(self, texts):
        counts = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        ordered = self.special_tokens + sorted(counts.keys())
        self.vocab = {token: idx for idx, token in enumerate(ordered)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        return self

    def encode(self, text: str):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.inv_vocab.get(idx, "<UNK>") for idx in token_ids]
        return " ".join(tokens).replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vocab": self.vocab, "special_tokens": self.special_tokens}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.special_tokens = payload.get("special_tokens", ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        self.vocab = payload.get("vocab", {})
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        return self

    def __call__(self, text: str):
        return self.encode(text)

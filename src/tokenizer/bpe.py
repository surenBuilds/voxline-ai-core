"""
Byte-Pair Encoding (BPE) Tokenizer for Voxline AI Core

Implements a subword tokenization system supporting:
- English
- Armenian
- Mixed language text
- Punctuation
- Numbers
- Code
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """Byte-Pair Encoding tokenizer with vocabulary management."""

    def __init__(
        self,
        vocab_size: int = 50000,
        special_tokens: Optional[List[str]] = None,
        pattern: Optional[str] = None,
    ):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (after merges)
            special_tokens: List of special tokens to reserve
            pattern: Regex pattern for splitting text into tokens
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "<CLS>",
            "<SEP>",
        ]

        # Unicode-aware pattern supporting Armenian, English, punctuation, numbers
        self.pattern = (
            pattern
            or r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Use a simpler pattern that works with standard Python regex
        self.pattern = r"""[a-zA-Zա-ևᴀ-ᴫ]+|[0-9]+|[^\w\s]|\s+"""

        self.vocab = {}
        self.bpe_merges = {}
        self.inv_vocab = {}
        self.merges_list = []

        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inv_vocab[i] = token

    def _tokenize_word(self, word: str) -> List[str]:
        """Split word into characters (byte-level start)."""
        return list(word)

    def fit(self, texts: List[str], num_merges: Optional[int] = None) -> "BPETokenizer":
        """
        Train BPE tokenizer on corpus.

        Args:
            texts: List of training texts
            num_merges: Number of merge operations (if None, use vocab_size - len(special_tokens))

        Returns:
            Self
        """
        if num_merges is None:
            num_merges = self.vocab_size - len(self.special_tokens)

        # Tokenize all words
        word_tokens = defaultdict(int)

        for text in texts:
            # Split text into words using regex pattern
            words = re.findall(self.pattern, text)
            for word in words:
                word = word.strip()
                if word:
                    # Convert word to character sequence
                    char_seq = " ".join(list(word)) + " </w>"
                    word_tokens[char_seq] += 1

        # Build initial character vocabulary
        vocab_tokens = set()
        for word in word_tokens.keys():
            for token in word.split():
                vocab_tokens.add(token)

        # Initialize vocab with characters
        start_idx = len(self.special_tokens)
        for i, token in enumerate(sorted(vocab_tokens)):
            self.vocab[token] = start_idx + i
            self.inv_vocab[start_idx + i] = token

        # Perform BPE merges
        for merge_idx in range(num_merges):
            # Count adjacent pairs
            pairs = defaultdict(int)
            for word, freq in word_tokens.items():
                tokens = word.split()
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pairs[pair] += freq

            if not pairs:
                break

            # Find most common pair
            most_common = max(pairs.items(), key=lambda x: x[1])
            pair, count = most_common

            # Create new token for this pair
            new_token = pair[0] + pair[1]
            new_idx = len(self.vocab)
            self.vocab[new_token] = new_idx
            self.inv_vocab[new_idx] = new_token
            self.bpe_merges[pair] = new_token
            self.merges_list.append(pair)

            # Update word tokens by merging this pair
            new_word_tokens = {}
            for word, freq in word_tokens.items():
                tokens = word.split()
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_word = " ".join(new_tokens)
                new_word_tokens[new_word] = freq

            word_tokens = new_word_tokens

        # Add any remaining tokens to vocab
        for word, freq in word_tokens.items():
            for token in word.split():
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.inv_vocab[self.vocab[token]] = token

        return self

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokens = []

        # Split text into words
        words = re.findall(self.pattern, text)

        for word in words:
            word = word.strip()
            if not word:
                continue

            # Convert word to character sequence
            chars = list(word)
            word_tokens = chars + ["</w>"]

            # Apply BPE merges
            for pair in self.merges_list:
                new_token = self.bpe_merges.get(pair)
                if new_token:
                    i = 0
                    new_word_tokens = []
                    while i < len(word_tokens):
                        if (
                            i < len(word_tokens) - 1
                            and (word_tokens[i], word_tokens[i + 1]) == pair
                        ):
                            new_word_tokens.append(new_token)
                            i += 2
                        else:
                            new_word_tokens.append(word_tokens[i])
                            i += 1
                    word_tokens = new_word_tokens

            # Convert tokens to IDs
            for token in word_tokens:
                token_id = self.vocab.get(token, self.vocab.get("<UNK>"))
                if token_id is not None:
                    tokens.append(token_id)

        return tokens if tokens else [self.vocab.get("<UNK>")]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = [self.inv_vocab.get(tid, "<UNK>") for tid in token_ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

    def __call__(self, text: str) -> List[int]:
        """Alias for encode."""
        return self.encode(text)

    def save(self, path: str) -> None:
        """
        Save tokenizer to JSON.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges_list,
        }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> "BPETokenizer":
        """
        Load tokenizer from JSON.

        Args:
            path: Path to load file

        Returns:
            Self
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.vocab_size = payload.get("vocab_size", self.vocab_size)
        self.special_tokens = payload.get("special_tokens", self.special_tokens)
        self.vocab = payload.get("vocab", {})
        self.merges_list = payload.get("merges", [])

        # Reconstruct inverse vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Reconstruct bpe_merges
        self.bpe_merges = {}
        for i, pair in enumerate(self.merges_list):
            if isinstance(pair, list):
                pair = tuple(pair)
            new_token = pair[0] + pair[1]
            self.bpe_merges[pair] = new_token

        return self

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        return self.special_tokens

    def token_to_id(self, token: str) -> int:
        """Get token ID from token string."""
        return self.vocab.get(token, self.vocab.get("<UNK>"))

    def id_to_token(self, token_id: int) -> str:
        """Get token string from token ID."""
        return self.inv_vocab.get(token_id, "<UNK>")

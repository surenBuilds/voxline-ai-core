from __future__ import annotations

from .model import VoxlineModel


class ChatBot:
    """Simple chat wrapper around the language model."""

    def __init__(self, model: VoxlineModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, prompt: str, max_new_tokens: int = 50):
        input_ids = self.tokenizer.encode(prompt)
        generated = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(generated)

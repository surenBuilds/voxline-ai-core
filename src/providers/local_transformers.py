"""Provider for an instruction-tuned Hugging Face model stored on this machine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import AsyncIterator, List, Optional

import torch

from src.providers.base import AIProvider, GenerationConfig, ProviderHealth, ProviderStatus


class LocalTransformersProvider(AIProvider):
    """Run a locally downloaded chat model without any hosted AI API."""

    def __init__(self, model_path: str, device: str = "cpu"):
        # Importing Transformers is expensive on some CPU-only Windows setups.
        # Delay it until a chat model is actually requested.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = Path(model_path)
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Local model folder does not exist: {self.path}. "
                "Download a model before starting the chat service."
            )
        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            local_files_only=True,
            torch_dtype=torch.float32 if self.device == "cpu" else "auto",
        ).to(self.device)
        self.model.eval()

    @property
    def provider_id(self) -> str:
        return "local_transformers"

    @property
    def model_id(self) -> str:
        return self.path.name

    @property
    def supports_streaming(self) -> bool:
        return False

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, message="Local model is loaded")

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        messages = [
            {"role": "system", "content": "You are Voxline, a helpful business assistant. Reply in the user's language."},
            {"role": "user", "content": prompt},
        ]
        return self.generate_chat(messages, config)

    async def stream(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        yield await self.generate(prompt, config)

    def generate_chat(self, messages: List[dict], config: GenerationConfig) -> str:
        """Generate one response from OpenAI-style messages, locally."""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            prompt = "\n".join(f"{item['role'].title()}: {item['content']}" for item in messages) + "\nAssistant:"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=config.max_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

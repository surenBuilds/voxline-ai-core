"""
Qwen Provider — wraps a locally downloaded Qwen2.5 model.

Uses HuggingFace transformers for inference.
This is the primary external model backend for Voxline AI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import AsyncIterator, List, Dict, Optional

import torch

from src.providers.base import (
    AIProvider, GenerationConfig, ProviderHealth, ProviderStatus, ModelInfo,
)


class QwenProvider(AIProvider):
    """
    Provider for Qwen2.5 models stored locally.

    Loads the model via HuggingFace transformers and provides
    the AIProvider interface for the rest of Voxline.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = Path(model_path)
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Qwen model folder does not exist: {self.path}. "
                "Download with: python scripts/download_local_model.py"
            )
        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            local_files_only=True,
            dtype=torch.float32 if self.device == "cpu" else "auto",
        ).to(self.device)
        self.model.eval()
        self._param_count = sum(p.numel() for p in self.model.parameters())

    @property
    def provider_id(self) -> str:
        return "qwen"

    @property
    def model_id(self) -> str:
        return self.path.name

    @property
    def supports_streaming(self) -> bool:
        return False

    def get_model_info(self) -> ModelInfo:
        config = self.model.config
        return ModelInfo(
            model_id=self.model_id,
            provider_id=self.provider_id,
            model_type="huggingface",
            parameters=self._param_count,
            vocab_size=getattr(config, "vocab_size", None),
            max_context_length=getattr(config, "max_position_embeddings", None),
            device=str(self.device),
            supports_streaming=False,
            extra={
                "model_path": str(self.path),
                "torch_dtype": str(next(self.model.parameters()).dtype),
            },
        )

    async def health_check(self) -> ProviderHealth:
        try:
            start = time.time()
            test_input = self.tokenizer("hello", return_tensors="pt").to(self.device)
            with torch.inference_mode():
                _ = self.model(**test_input)
            elapsed = (time.time() - start) * 1000
            return ProviderHealth(
                status=ProviderStatus.HEALTHY,
                message="Qwen model is loaded and responsive",
                response_time_ms=elapsed,
            )
        except Exception as e:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message=f"Health check failed: {e}",
            )

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Voxline, a helpful bilingual AI assistant. "
                    "Reply in the user's language."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._generate_chat(messages, config)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> str:
        """Generate a response from multi-turn messages using Qwen's chat template."""
        system_msg = {
            "role": "system",
            "content": (
                "You are Voxline, a helpful bilingual AI assistant. "
                "Reply in the user's language."
            ),
        }
        full_messages = [system_msg] + messages
        return self._generate_chat(full_messages, config)

    def _generate_chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str:
        """Generate one response from OpenAI-style messages, locally."""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            chat_output = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(chat_output, "input_ids"):
                input_ids = chat_output.input_ids.to(self.device)
            else:
                input_ids = chat_output.to(self.device)
        else:
            prompt = "\n".join(
                f"{item['role'].title()}: {item['content']}" for item in messages
            ) + "\nAssistant:"
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

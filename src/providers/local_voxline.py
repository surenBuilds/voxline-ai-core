"""
Local Voxline AI Provider - wraps VoxlineTransformer model.

Does NOT include logic from VoxlineTransformer - only wraps it.
Model remains in src/model/transformer.py.
"""

import torch
import logging
from typing import AsyncIterator, List, Dict
import time

from src.providers.base import (
    AIProvider, GenerationConfig, ProviderHealth, ProviderStatus, ModelInfo,
)
from src.checkpoint import CheckpointLoader
from src.config.model_config import ModelConfig


logger = logging.getLogger(__name__)


class LocalVoxlineProvider(AIProvider):
    """
    Local provider using VoxlineTransformer.

    This provider wraps the locally trained VoxlineTransformer model
    and makes it available through the AIProvider interface.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_config: ModelConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.model.eval()

        logger.info(f"Initialized LocalVoxlineProvider on {device}")
        logger.info(f"Model: {model_config.model_type} v{model_config.model_version}")
        logger.info(f"Vocab: {model_config.vocab_size}, Dim: {model_config.d_model}")

    @property
    def provider_id(self) -> str:
        return "local_voxline"

    @property
    def model_id(self) -> str:
        return f"voxline_{self.model_config.model_version}"

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            provider_id=self.provider_id,
            model_type="native",
            parameters=self.model.get_num_parameters(),
            vocab_size=self.model_config.vocab_size,
            max_context_length=self.model_config.max_seq_len,
            device=self.device,
            supports_streaming=True,
            extra={
                "d_model": self.model_config.d_model,
                "num_layers": self.model_config.num_layers,
                "num_heads": self.model_config.num_heads,
                "d_ff": self.model_config.d_ff,
            },
        )

    async def health_check(self) -> ProviderHealth:
        try:
            start = time.time()
            test_prompt = "test"
            tokens = self.tokenizer.encode(test_prompt)
            if len(tokens) == 0:
                return ProviderHealth(
                    status=ProviderStatus.DEGRADED,
                    message="Tokenizer failed on test input",
                )
            input_ids = torch.tensor([tokens[:min(10, len(tokens))]], device=self.device)
            with torch.no_grad():
                _ = self.model(input_ids)
            elapsed = (time.time() - start) * 1000
            return ProviderHealth(
                status=ProviderStatus.HEALTHY,
                message="Provider is healthy",
                response_time_ms=elapsed,
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message=f"Health check failed: {str(e)}",
            )

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        full_response = ""
        async for token in self.stream(prompt, config):
            full_response += token
        return full_response

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> str:
        """Override chat to use the native prompt format."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        return await self.generate(prompt, config)

    async def stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        try:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                logger.warning(f"Prompt tokenized to empty: {prompt}")
                yield ""
                return

            input_ids = torch.tensor([tokens], device=self.device)

            with torch.no_grad():
                for _ in range(config.max_tokens):
                    logits = self.model(input_ids)
                    next_token_logits = logits[:, -1, :]

                    if config.temperature != 1.0:
                        next_token_logits = next_token_logits / config.temperature

                    if config.do_sample:
                        if config.top_k:
                            top_k_logits, top_k_indices = torch.topk(
                                next_token_logits, config.top_k
                            )
                            probs = torch.softmax(top_k_logits, dim=-1)
                            sample_idx = torch.multinomial(probs, 1)
                            next_token = top_k_indices.gather(-1, sample_idx)
                        elif config.top_p:
                            sorted_logits, sorted_indices = torch.sort(
                                next_token_logits, descending=True
                            )
                            sorted_probs = torch.softmax(sorted_logits, dim=-1)
                            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumsum_probs > config.top_p
                            sorted_indices_to_remove[..., 0] = False
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                -1, sorted_indices, sorted_indices_to_remove
                            )
                            next_token_logits[indices_to_remove] = float('-inf')
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    token_str = self.tokenizer.decode([next_token.item()])
                    yield token_str

                    input_ids = torch.cat([input_ids, next_token], dim=-1)

                    if input_ids.shape[1] >= self.model_config.max_seq_len:
                        break

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            yield f"[Error during generation: {str(e)}]"

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        tokenizer,
        model_class,
        device: str = "cpu",
    ) -> "LocalVoxlineProvider":
        """
        Load provider from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint
            tokenizer: Tokenizer instance
            model_class: Model class (VoxlineTransformer or NextTokenModel)
            device: Device to load on

        Returns:
            Initialized provider
        """
        from src.config.model_config import ModelType

        if "VoxlineTransformer" in model_class.__name__:
            expected_config = ModelConfig.for_voxline_transformer()
        elif "NextTokenModel" in model_class.__name__:
            expected_config = ModelConfig.for_next_token_model()
        else:
            raise ValueError(f"Unknown model class: {model_class.__name__}")

        state_dict, checkpoint_config = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            expected_config,
            device=device,
            strict=True,
        )

        model = model_class(**checkpoint_config.to_dict())
        model.load_state_dict(state_dict)
        model = model.to(device)

        logger.info(f"Loaded checkpoint into {model_class.__name__}")

        return LocalVoxlineProvider(
            model=model,
            tokenizer=tokenizer,
            model_config=checkpoint_config,
            device=device,
        )

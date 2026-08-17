"""
Inference engine for text generation

Supports:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Batch inference
- Streaming
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Callable
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Generation configuration."""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    do_sample: bool = True


class TextGenerator:
    """Text generation engine."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        """
        Initialize generator.

        Args:
            model: Language model
            tokenizer: Tokenizer instance
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        return_text: bool = True,
    ) -> str | List[int]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt text
            config: Generation configuration
            return_text: Whether to return text or token IDs

        Returns:
            Generated text or token IDs
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self._generate_greedy(input_ids, config)

        if return_text:
            return self.tokenizer.decode(output_ids[0].tolist())
        else:
            return output_ids[0].tolist()

    def _generate_greedy(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """
        Generate using greedy decoding.

        Args:
            input_ids: (batch_size, seq_len)
            config: Generation configuration

        Returns:
            (batch_size, seq_len + num_generated)
        """
        batch_size = input_ids.size(0)

        for _ in range(config.max_new_tokens):
            # Truncate to max sequence length if needed
            if input_ids.size(1) > self.model.max_seq_len:
                context_ids = input_ids[:, -self.model.max_seq_len :]
            else:
                context_ids = input_ids

            # Forward pass
            logits = self.model(context_ids)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature

            # Apply top-k filtering
            if config.top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, config.top_k)
                next_token_logits = torch.full_like(
                    next_token_logits, float("-inf")
                )
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) sampling
            if config.top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumsum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumsum_probs > config.top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float("-inf")

            # Sample or greedy
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS token
            if (
                config.eos_token_id is not None
                and (next_token == config.eos_token_id).all()
            ):
                break

        return input_ids

    def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        on_token: Callable[[str], None],
    ):
        """
        Generate text with streaming callback.

        Args:
            prompt: Input prompt
            config: Generation configuration
            on_token: Callback called on each generated token
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate and stream
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                # Truncate if needed
                if input_ids.size(1) > self.model.max_seq_len:
                    context_ids = input_ids[:, -self.model.max_seq_len :]
                else:
                    context_ids = input_ids

                # Forward pass
                logits = self.model(context_ids)
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature

                # Sample or greedy
                if config.do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Decode and callback
                token_text = self.tokenizer.id_to_token(next_token.item())
                on_token(token_text)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check for EOS
                if (
                    config.eos_token_id is not None
                    and (next_token == config.eos_token_id).all()
                ):
                    break

    def batch_generate(
        self,
        prompts: List[str],
        config: GenerationConfig,
        return_text: bool = True,
    ) -> List[str] | List[List[int]]:
        """
        Generate from multiple prompts.

        Args:
            prompts: List of prompts
            config: Generation configuration
            return_text: Whether to return text or token IDs

        Returns:
            List of generated texts or token lists
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, config, return_text=return_text)
            results.append(result)
        return results

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)

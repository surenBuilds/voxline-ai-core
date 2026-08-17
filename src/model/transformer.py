"""
Transformer Language Model for Voxline AI Core

Implements a decoder-only Transformer suitable for autoregressive language
modeling.

Architecture:
- Token embedding
- Positional encoding
- Transformer blocks (attention + feedforward)
- Output projection to vocabulary
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from src.attention.attention import CausalSelfAttention


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        # Attention
        self.attention = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residual connections."""
        # Self-attention with residual
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class VoxlineTransformer(nn.Module):
    """
    Voxline Transformer Language Model.

    Autoregressive decoder-only transformer for next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        """
        Initialize Transformer Language Model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feedforward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            tie_embeddings: Whether to tie embedding and output weights
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tie embeddings if specified
        self.tie_embeddings = tie_embeddings
        if tie_embeddings:
            self.output_proj.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) - unused for now

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()

        # Embed tokens
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply final layer norm
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: (batch_size, seq_len) - starting tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Filter to top-k tokens
            top_p: Nucleus sampling threshold
            pad_token_id: ID of padding token

        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate to max_seq_len if needed
                if input_ids.size(1) > self.max_seq_len:
                    input_ids_truncated = input_ids[:, -self.max_seq_len :]
                else:
                    input_ids_truncated = input_ids

                # Forward pass
                logits = self(input_ids_truncated)

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][
                        :, -1, None
                    ]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply nucleus (top-p) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float("-inf")

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
        }

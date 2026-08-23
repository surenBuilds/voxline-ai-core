"""
Model configuration with version tracking for checkpoint compatibility.

Each model architecture has a unique config ID to prevent loading incompatible checkpoints.
"""

from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum
import json
from pathlib import Path


class ModelType(Enum):
    """Model architecture types."""
    NEXT_TOKEN_MODEL = "next_token_model"  # v0.3 in main.py
    VOXLINE_TRANSFORMER = "voxline_transformer"  # Full transformer


@dataclass
class ModelConfig:
    """Complete model architecture configuration with versioning."""
    
    model_type: str  # ModelType.value
    model_version: str  # e.g. "0.3.0", "1.0.0"
    
    # Architecture
    vocab_size: int
    d_model: int
    max_seq_len: int
    
    # Transformer-specific
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    d_ff: Optional[int] = None
    dropout: float = 0.1
    
    # Legacy/SimpleModel-specific
    embed_dim: Optional[int] = None
    seq_length: Optional[int] = None
    
    # Tokenizer
    tokenizer_type: str = "bpe"  # bpe, simple
    tokenizer_version: str = "0.1.0"
    
    # Training info (for context)
    training_epochs: Optional[int] = None
    training_batch_size: Optional[int] = None
    
    @classmethod
    def for_next_token_model(
        cls,
        vocab_size: int = 32000,
        embed_dim: int = 32,
        seq_length: int = 8,
    ) -> "ModelConfig":
        """Create config for NextTokenModel (v0.3)."""
        return cls(
            model_type=ModelType.NEXT_TOKEN_MODEL.value,
            model_version="0.3.0",
            vocab_size=vocab_size,
            d_model=embed_dim,
            max_seq_len=seq_length,
            embed_dim=embed_dim,
            seq_length=seq_length,
            tokenizer_type="simple",
            tokenizer_version="0.1.0",
        )
    
    @classmethod
    def for_voxline_transformer(
        cls,
        vocab_size: int = 50000,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> "ModelConfig":
        """Create config for VoxlineTransformer."""
        return cls(
            model_type=ModelType.VOXLINE_TRANSFORMER.value,
            model_version="1.0.0",
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            tokenizer_type="bpe",
            tokenizer_version="0.1.0",
        )
    
    def config_id(self) -> str:
        """Unique config ID for compatibility checking."""
        # Minimal unique signature of architecture
        key_params = f"{self.model_type}_{self.vocab_size}_{self.d_model}_{self.max_seq_len}"
        if self.num_layers:
            key_params += f"_{self.num_layers}_{self.num_heads}_{self.d_ff}"
        if self.embed_dim:
            key_params += f"_{self.embed_dim}_{self.seq_length}"
        return key_params
    
    def is_compatible_with_checkpoint(self, checkpoint_config: "ModelConfig") -> bool:
        """Check if checkpoint config is compatible with this config."""
        if self.model_type != checkpoint_config.model_type:
            return False
        # For now, require exact match on key architecture params
        return self.config_id() == checkpoint_config.config_id()
    
    def to_dict(self) -> dict:
        """Convert to dictionary (JSON serializable)."""
        return asdict(self)
    
    def to_model_kwargs(self) -> dict:
        """Get only the model architecture parameters (for model initialization)."""
        kwargs = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
        }
        
        # Add transformer-specific params
        if self.num_layers is not None:
            kwargs["num_layers"] = self.num_layers
        if self.num_heads is not None:
            kwargs["num_heads"] = self.num_heads
        if self.d_ff is not None:
            kwargs["d_ff"] = self.d_ff
        
        # Add legacy-specific params
        if self.embed_dim is not None:
            kwargs["embed_dim"] = self.embed_dim
        if self.seq_length is not None:
            kwargs["seq_length"] = self.seq_length
        
        return kwargs
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Create from dictionary. Ignores unknown keys and fills defaults for missing required fields."""
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        # Sensible defaults for required fields if missing
        defaults = {
            "model_type": ModelType.VOXLINE_TRANSFORMER.value,
            "model_version": "0.4.0",
        }
        for key, default in defaults.items():
            if key not in filtered:
                filtered[key] = default
        return cls(**filtered)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Default configurations
DEFAULT_NEXT_TOKEN_MODEL_CONFIG = ModelConfig.for_next_token_model()
DEFAULT_VOXLINE_TRANSFORMER_CONFIG = ModelConfig.for_voxline_transformer()

"""
Safe checkpoint loading with version/config validation.

Prevents silent incompatibility by validating checkpoint config before loading.
"""

import torch
from pathlib import Path
from typing import Tuple, Optional
import json
import logging

from src.config.model_config import ModelConfig, ModelType


logger = logging.getLogger(__name__)


class CheckpointIncompatibilityError(Exception):
    """Raised when checkpoint architecture is incompatible with target model."""
    pass


class CheckpointLoader:
    """Safe checkpoint loading with validation."""
    
    CHECKPOINT_CONFIG_SUFFIX = ".config.json"
    
    @staticmethod
    def save_checkpoint(
        model_state_dict: dict,
        config: ModelConfig,
        checkpoint_path: str,
        additional_data: Optional[dict] = None,
    ) -> None:
        """
        Save checkpoint with config validation.
        
        Args:
            model_state_dict: Model state dict
            config: Model configuration
            checkpoint_path: Path to save checkpoint
            additional_data: Optional additional data (e.g., optimizer state)
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config alongside checkpoint
        config_path = checkpoint_path.parent / f"{checkpoint_path.stem}.config.json"
        config.save(str(config_path))
        
        # Save checkpoint
        checkpoint_data = {
            "model_state_dict": model_state_dict,
            "config": config.to_dict(),
        }
        if additional_data:
            checkpoint_data.update(additional_data)
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        logger.info(f"Saved config: {config_path}")
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        target_config: ModelConfig,
        device: str = "cpu",
        strict: bool = True,
    ) -> Tuple[dict, ModelConfig]:
        """
        Load checkpoint with compatibility validation.
        
        Args:
            checkpoint_path: Path to checkpoint
            target_config: Expected model configuration
            device: Device to load on
            strict: If True, raise error on incompatibility; if False, return warning
            
        Returns:
            (state_dict, checkpoint_config)
            
        Raises:
            CheckpointIncompatibilityError: If checkpoint is incompatible
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config if available
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            checkpoint_config_data = checkpoint["config"]
            checkpoint_config = ModelConfig.from_dict(checkpoint_config_data)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            logger.info(f"Found config in checkpoint: {checkpoint_config.config_id()}")
        else:
            # Legacy checkpoint without config
            # Try to infer from state dict structure
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint
            checkpoint_config = CheckpointLoader._infer_config_from_state_dict(state_dict)
            logger.warning(f"No config found in checkpoint. Inferred: {checkpoint_config.config_id()}")
        
        # Validate compatibility
        is_compatible = target_config.is_compatible_with_checkpoint(checkpoint_config)
        
        if not is_compatible:
            error_msg = (
                f"\n"
                f"CHECKPOINT INCOMPATIBILITY DETECTED\n"
                f"  Target model:      {target_config.config_id()}\n"
                f"  Checkpoint config: {checkpoint_config.config_id()}\n"
                f"  Target type:       {target_config.model_type}\n"
                f"  Checkpoint type:   {checkpoint_config.model_type}\n"
                f"  Target version:    {target_config.model_version}\n"
                f"  Checkpoint version: {checkpoint_config.model_version}\n"
            )
            if strict:
                raise CheckpointIncompatibilityError(error_msg)
            else:
                logger.error(error_msg)
        
        logger.info(f"Checkpoint compatibility check passed")
        return state_dict, checkpoint_config
    
    @staticmethod
    def _infer_config_from_state_dict(state_dict: dict) -> ModelConfig:
        """
        Infer model config from state dict structure.
        
        This is a best-effort attempt to determine which model architecture
        the checkpoint is for based on layer names.
        """
        keys = list(state_dict.keys())
        
        # Check for NextTokenModel indicators
        if any(k.startswith("net.") for k in keys):
            logger.warning("Detected NextTokenModel (v0.3) architecture")
            # Infer from state dict
            if "embedding.weight" in state_dict:
                embed_dim = state_dict["embedding.weight"].shape[-1]
                vocab_size = state_dict["embedding.weight"].shape[0]
                config = ModelConfig.for_next_token_model(
                    vocab_size=vocab_size,
                    embed_dim=embed_dim,
                )
                return config
        
        # Check for VoxlineTransformer indicators
        if any(k.startswith("transformer_blocks.") for k in keys):
            logger.warning("Detected VoxlineTransformer architecture")
            # Infer from state dict
            if "token_embedding.weight" in state_dict:
                vocab_size = state_dict["token_embedding.weight"].shape[0]
                d_model = state_dict["token_embedding.weight"].shape[1]
                
                # Count transformer blocks
                block_indices = set()
                for k in keys:
                    if k.startswith("transformer_blocks."):
                        idx = int(k.split(".")[1])
                        block_indices.add(idx)
                num_layers = len(block_indices)
                
                # Infer other params (with defaults)
                num_heads = 12  # Default
                d_ff = 3072  # Default
                
                config = ModelConfig.for_voxline_transformer(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    d_ff=d_ff,
                )
                return config
        
        # Unknown architecture
        raise ValueError(f"Could not infer model config from checkpoint. Keys: {keys[:5]}")

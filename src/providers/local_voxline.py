"""
Local Voxline AI Provider - wraps VoxlineTransformer model.

Does NOT include logic from VoxlineTransformer - only wraps it.
Model remains in src/model/transformer.py.
"""

import torch
import logging
from typing import AsyncIterator
import time

from src.providers.base import AIProvider, GenerationConfig, ProviderHealth, ProviderStatus
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
        """
        Initialize local Voxline provider.
        
        Args:
            model: VoxlineTransformer model instance
            tokenizer: Tokenizer instance
            model_config: Model configuration
            device: Device to run on (cpu/cuda)
        """
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
        """Provider identifier."""
        return "local_voxline"
    
    @property
    def model_id(self) -> str:
        """Model identifier."""
        return f"voxline_{self.model_config.model_version}"
    
    @property
    def supports_streaming(self) -> bool:
        """Whether streaming is supported."""
        return True
    
    async def health_check(self) -> ProviderHealth:
        """Check provider health by running inference on dummy input."""
        try:
            start = time.time()
            
            # Quick inference test
            test_prompt = "test"
            tokens = self.tokenizer.encode(test_prompt)
            
            if len(tokens) == 0:
                # Tokenizer failed
                return ProviderHealth(
                    status=ProviderStatus.DEGRADED,
                    message="Tokenizer failed on test input",
                )
            
            # Try model forward pass
            input_ids = torch.tensor([tokens[:min(10, len(tokens))]], device=self.device)
            with torch.no_grad():
                _ = self.model(input_ids)
            
            elapsed = (time.time() - start) * 1000  # ms
            
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
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """
        Generate text response synchronously.
        
        This is a wrapper around the streaming method for simple use cases.
        """
        full_response = ""
        async for token in self.stream(prompt, config):
            full_response += token
        return full_response
    
    async def stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """
        Stream generated text token by token.
        
        Note: This is synchronous generation exposed as async iterator.
        For true async, would need async runtime.
        """
        try:
            # Tokenize prompt
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                logger.warning(f"Prompt tokenized to empty: {prompt}")
                yield ""
                return
            
            # Convert to tensor
            input_ids = torch.tensor([tokens], device=self.device)
            
            # Generate with model
            with torch.no_grad():
                for _ in range(config.max_tokens):
                    # Get logits from model
                    logits = self.model(input_ids)  # [batch, seq_len, vocab]
                    next_token_logits = logits[:, -1, :]  # [batch, vocab]
                    
                    # Apply temperature
                    if config.temperature != 1.0:
                        next_token_logits = next_token_logits / config.temperature
                    
                    # Sampling
                    if config.do_sample:
                        if config.top_k:
                            # Top-k sampling
                            top_k_logits, top_k_indices = torch.topk(
                                next_token_logits, config.top_k
                            )
                            probs = torch.softmax(top_k_logits, dim=-1)
                            sample_idx = torch.multinomial(probs, 1)
                            next_token = top_k_indices.gather(-1, sample_idx)
                        elif config.top_p:
                            # Top-p (nucleus) sampling
                            sorted_logits, sorted_indices = torch.sort(
                                next_token_logits, descending=True
                            )
                            sorted_probs = torch.softmax(sorted_logits, dim=-1)
                            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                            
                            # Find cutoff
                            sorted_indices_to_remove = cumsum_probs > config.top_p
                            sorted_indices_to_remove[..., 0] = False
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                -1, sorted_indices, sorted_indices_to_remove
                            )
                            next_token_logits[indices_to_remove] = float('-inf')
                            
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            # Standard sampling
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                    else:
                        # Greedy
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Decode token
                    token_str = self.tokenizer.decode([next_token.item()])
                    yield token_str
                    
                    # Append to sequence
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    
                    # Stop if max seq length reached
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
            
        Raises:
            CheckpointIncompatibilityError: If checkpoint architecture doesn't match model
        """
        from src.config.model_config import ModelType
        
        # Determine expected config based on model_class name
        if "VoxlineTransformer" in model_class.__name__:
            expected_config = ModelConfig.for_voxline_transformer()
        elif "NextTokenModel" in model_class.__name__:
            expected_config = ModelConfig.for_next_token_model()
        else:
            raise ValueError(f"Unknown model class: {model_class.__name__}")
        
        # Load and validate checkpoint
        state_dict, checkpoint_config = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            expected_config,
            device=device,
            strict=True,
        )
        
        # Recreate model with config from checkpoint
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

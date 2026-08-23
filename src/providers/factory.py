"""
Provider factory for initializing AI providers.

Handles provider selection, configuration, and initialization.
"""

import logging
from typing import Optional

from src.config.settings import VoxlineConfig
from src.providers.base import AIProvider
from src.errors import ProviderNotFoundError


logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating AI providers."""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, provider_id: str, provider_class: type) -> None:
        """Register a provider class."""
        cls._providers[provider_id] = provider_class
        logger.info(f"Registered provider: {provider_id}")
    
    @classmethod
    def create(
        cls,
        config: VoxlineConfig,
        tokenizer=None,
        model=None,
    ) -> AIProvider:
        """
        Create provider based on configuration.
        
        Args:
            config: Voxline configuration
            tokenizer: Tokenizer (for local providers)
            model: Model (for local providers)
            
        Returns:
            Initialized provider
            
        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider_id = config.ai_provider
        logger.info(f"Initializing provider: {provider_id}")
        
        if provider_id not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ProviderNotFoundError(
                f"Provider '{provider_id}' not found. Available: {available}"
            )
        
        provider_class = cls._providers[provider_id]
        
        try:
            # Local provider
            if provider_id == "local":
                if model is None or tokenizer is None:
                    raise ValueError("Local provider requires model and tokenizer")
                
                from src.providers.local_voxline import LocalVoxlineProvider
                from src.config.model_config import ModelConfig
                
                model_config = ModelConfig.for_voxline_transformer()
                return LocalVoxlineProvider(
                    model=model,
                    tokenizer=tokenizer,
                    model_config=model_config,
                    device=config.ai_device,
                )
            
            # External providers would be initialized here
            # For now, only local is implemented
            raise NotImplementedError(f"Provider '{provider_id}' not yet implemented")
        
        except Exception as e:
            logger.error(f"Failed to initialize provider '{provider_id}': {e}")
            raise


# Register default providers
ProviderFactory.register_provider("local", None)  # Will use LocalVoxlineProvider
# Future:
# ProviderFactory.register_provider("openai", OpenAIProvider)
# ProviderFactory.register_provider("gemini", GeminiProvider)
# ProviderFactory.register_provider("anthropic", AnthropicProvider)

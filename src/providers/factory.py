"""
Provider factory for initializing AI providers.

Handles provider selection, configuration, and initialization.
Provider selection is driven by VoxlineConfig.
"""

import logging
from typing import Optional, Dict, Type

from src.config.settings import VoxlineConfig
from src.providers.base import AIProvider
from src.errors import ProviderNotFoundError


logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating AI providers.

    Providers are registered by ID and created from VoxlineConfig.
    The config's AI_PROVIDER setting selects which provider to use.

    Supported provider IDs:
    - "qwen": QwenProvider (requires model_path) — default
    - "native": NativeVoxlineProvider (requires model + tokenizer)
    - "openai": OpenAICompatProvider (hosted, OpenAI-compatible API) — serverless/Vercel
    """

    _providers: Dict[str, Type[AIProvider]] = {}

    @classmethod
    def register_provider(cls, provider_id: str, provider_class: Type[AIProvider]) -> None:
        """Register a provider class."""
        cls._providers[provider_id] = provider_class
        logger.info(f"Registered provider: {provider_id}")

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of registered provider IDs."""
        return list(cls._providers.keys())

    @classmethod
    def create(
        cls,
        config: VoxlineConfig,
        tokenizer=None,
        model=None,
        model_config=None,
    ) -> AIProvider:
        """
        Create provider based on configuration.

        Args:
            config: Voxline configuration
            tokenizer: Tokenizer (for native provider)
            model: Model (for native provider)
            model_config: ModelConfig (for native provider)

        Returns:
            Initialized provider

        Raises:
            ProviderNotFoundError: If provider not found or misconfigured
        """
        _ensure_builtin_providers()

        provider_id = config.ai_provider
        logger.info(f"Creating provider: {provider_id}")

        if provider_id not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ProviderNotFoundError(
                f"Provider '{provider_id}' not found. Available: {available}"
            )

        try:
            if provider_id == "native":
                return cls._create_native(config, model, tokenizer, model_config)
            elif provider_id == "qwen":
                return cls._create_qwen(config)
            else:
                provider_class = cls._providers[provider_id]
                return provider_class(config=config)
        except ProviderNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to create provider '{provider_id}': {e}")
            raise ProviderNotFoundError(
                f"Failed to initialize provider '{provider_id}': {e}"
            ) from e

    @classmethod
    def _create_native(
        cls,
        config: VoxlineConfig,
        model,
        tokenizer,
        model_config,
    ) -> AIProvider:
        """Create native Voxline provider from loaded model and tokenizer."""
        from src.providers.local_voxline import LocalVoxlineProvider

        if model is None or tokenizer is None:
            raise ProviderNotFoundError(
                "Native provider requires a loaded model and tokenizer. "
                "Pass model and tokenizer to ProviderFactory.create()"
            )

        if model_config is None:
            from src.config.model_config import ModelConfig
            model_config = ModelConfig.for_voxline_transformer()

        return LocalVoxlineProvider(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            device=config.ai_device,
        )

    @classmethod
    def _create_qwen(cls, config: VoxlineConfig) -> AIProvider:
        """Create Qwen provider from configuration."""
        from src.providers.qwen_provider import QwenProvider

        model_path = config.get("AI_MODEL_PATH")
        if not model_path:
            raise ProviderNotFoundError(
                "Qwen provider requires AI_MODEL_PATH in configuration. "
                "Set AI_MODEL_PATH to the path of a Qwen model directory."
            )

        return QwenProvider(
            model_path=model_path,
            device=config.ai_device,
        )


# Register built-in providers (lazy — classes resolved on first use)
# Actual registration happens in _ensure_builtin_providers()
_builtin_registered = False


def _ensure_builtin_providers():
    """Register built-in providers on first factory use."""
    global _builtin_registered
    if _builtin_registered:
        return
    _builtin_registered = True
    from src.providers.local_voxline import LocalVoxlineProvider
    from src.providers.qwen_provider import QwenProvider
    from src.providers.hosted import OpenAICompatProvider
    ProviderFactory.register_provider("native", LocalVoxlineProvider)
    ProviderFactory.register_provider("qwen", QwenProvider)
    ProviderFactory.register_provider("openai", OpenAICompatProvider)

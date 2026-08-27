"""Voxline AI Providers module."""

from .base import AIProvider, GenerationConfig, ProviderHealth, ProviderStatus, ModelInfo

try:
    from .local_voxline import LocalVoxlineProvider
    from .qwen_provider import QwenProvider
except ImportError:
    LocalVoxlineProvider = None
    QwenProvider = None

from .hosted import OpenAICompatProvider
from .factory import ProviderFactory

__all__ = [
    "AIProvider",
    "GenerationConfig",
    "ModelInfo",
    "ProviderHealth",
    "ProviderStatus",
    "LocalVoxlineProvider",
    "QwenProvider",
    "OpenAICompatProvider",
    "ProviderFactory",
]

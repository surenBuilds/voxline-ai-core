"""
AI Provider abstraction layer.

Defines interface for language models to support:
- Local models (VoxlineTransformer, Qwen)
- External APIs (OpenAI, Gemini, Anthropic, etc.)

Agents and the API server interact with AIProvider, not directly with models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator
from enum import Enum


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    do_sample: bool = True


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""
    model_id: str
    provider_id: str
    model_type: str  # "native", "huggingface", "api"
    parameters: Optional[int] = None
    vocab_size: Optional[int] = None
    max_context_length: Optional[int] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    supports_streaming: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    """Provider health status info."""
    status: ProviderStatus
    message: str
    response_time_ms: Optional[float] = None
    available_models: Optional[List[str]] = None


class AIProvider(ABC):
    """
    Abstract base class for AI providers.

    Every provider must implement:
    - generate(prompt, config) -> str
    - health_check() -> ProviderHealth
    - provider_id, model_id, supports_streaming properties

    Default implementations provided for:
    - chat(messages, config) -> str (builds prompt from messages, calls generate)
    - stream(prompt, config) -> AsyncIterator[str] (falls back to generate)
    - get_model_info() -> ModelInfo (returns basic info)
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """
        Generate text response.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Generated text
        """
        pass

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> str:
        """
        Generate a response from multi-turn messages.

        Default implementation formats messages into a prompt and calls generate().
        Providers with native chat support (e.g. Qwen, OpenAI) should override this.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            config: Generation configuration

        Returns:
            Generated response text
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        return await self.generate(prompt, config)

    async def stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """
        Stream generated text token by token.

        Default implementation yields the full generate() result at once.
        Providers with streaming support should override this.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Text tokens
        """
        result = await self.generate(prompt, config)
        yield result

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """
        Check provider health.

        Returns:
            Health status
        """
        pass

    def get_model_info(self) -> ModelInfo:
        """
        Get metadata about the loaded model.

        Returns:
            ModelInfo with provider/model metadata
        """
        return ModelInfo(
            model_id=self.model_id,
            provider_id=self.provider_id,
            model_type="unknown",
            supports_streaming=self.supports_streaming,
        )

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Currently active model ID."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether provider supports streaming."""
        pass

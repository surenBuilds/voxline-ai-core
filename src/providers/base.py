"""
AI Provider abstraction layer.

Defines interface for language models to support:
- Local models (VoxlineTransformer, NextTokenModel)
- External APIs (OpenAI, Gemini, Anthropic, etc.)

Agents interact with AIProvider, not directly with models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
class ProviderHealth:
    """Provider health status info."""
    status: ProviderStatus
    message: str
    response_time_ms: Optional[float] = None
    available_models: Optional[List[str]] = None
    

class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
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
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """
        Stream generated text token by token.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Yields:
            Text tokens
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """
        Check provider health.
        
        Returns:
            Health status
        """
        pass
    
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

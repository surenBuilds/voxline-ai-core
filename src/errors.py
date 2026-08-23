"""Voxline AI centralized error classes.

All domain-specific exceptions inherit from VoxlineError.
"""


class VoxlineError(Exception):
    """Base exception for all Voxline errors."""


# Model errors
class ModelError(VoxlineError):
    """Base model error."""


class ModelLoadError(ModelError):
    """Failed to load model from checkpoint."""


class ModelInferenceError(ModelError):
    """Error during model inference/generation."""


class ModelConfigError(ModelError):
    """Invalid or incompatible model configuration."""


# Tokenizer errors
class TokenizerError(VoxlineError):
    """Base tokenizer error."""


class TokenizerLoadError(TokenizerError):
    """Failed to load tokenizer."""


class TokenizerEncodeError(TokenizerError):
    """Failed to encode text."""


# Checkpoint errors
class CheckpointError(VoxlineError):
    """Base checkpoint error."""


class CheckpointIncompatibilityError(CheckpointError):
    """Checkpoint does not match target model configuration."""


class CheckpointLoadError(CheckpointError):
    """Failed to load checkpoint file."""


# Provider errors
class ProviderError(VoxlineError):
    """Base provider error."""


class ProviderNotFoundError(ProviderError):
    """Requested provider is not registered."""


class ProviderUnavailableError(ProviderError):
    """Provider is not available or health check failed."""


# Memory errors
class MemoryError(VoxlineError):
    """Base memory error."""


class MemoryStoreError(MemoryError):
    """Database operation failed."""


# Tool errors
class ToolError(VoxlineError):
    """Base tool error."""


class ToolNotFoundError(ToolError):
    """Requested tool is not registered."""


class ToolExecutionError(ToolError):
    """Tool execution failed."""


class ToolPermissionError(ToolError):
    """Tool execution denied by permission policy."""


# Configuration errors
class ConfigError(VoxlineError):
    """Base configuration error."""


class ConfigLoadError(ConfigError):
    """Failed to load configuration."""


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""


# Training errors
class TrainingError(VoxlineError):
    """Base training error."""


class TrainingDataError(TrainingError):
    """Invalid or insufficient training data."""


# Agent errors
class AgentError(VoxlineError):
    """Base agent error."""


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""


class AgentMaxIterationsError(AgentError):
    """Agent exceeded maximum iterations."""

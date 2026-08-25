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


# Session errors
class SessionError(VoxlineError):
    """Base session error."""


class SessionNotFoundError(SessionError):
    """Requested session does not exist."""


class SessionExpiredError(SessionError):
    """Session has expired."""


# Workspace errors
class WorkspaceError(VoxlineError):
    """Base workspace error."""


class WorkspaceBoundaryError(WorkspaceError):
    """Operation attempted outside workspace boundary."""


# Coding agent errors
class CodingAgentError(VoxlineError):
    """Base coding agent error."""


class AgentPlanError(CodingAgentError):
    """Agent produced an invalid or unparseable plan."""


class CommandDeniedError(ToolError):
    """Command not in the allowed command list."""


# Integration errors
class IntegrationError(VoxlineError):
    """Base integration error."""


class GitHubError(IntegrationError):
    """Base GitHub integration error."""


class GitHubAuthenticationError(GitHubError):
    """GitHub authentication failed."""


class GitHubRepositoryNotFoundError(GitHubError):
    """GitHub repository not found."""


class GitHubBranchConflictError(GitHubError):
    """GitHub branch already exists or conflicts."""


class GitHubOperationDeniedError(GitHubError):
    """GitHub operation denied by permission policy."""


class VercelError(IntegrationError):
    """Base Vercel integration error."""


class VercelAuthenticationError(VercelError):
    """Vercel authentication failed."""


class VercelDeploymentError(VercelError):
    """Vercel deployment failed."""


class VercelProductionApprovalRequiredError(VercelError):
    """Production deployment requires explicit approval."""

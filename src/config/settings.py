"""
Configuration management with environment variable support.

Supports .env files and environment variables.
Secrets are never logged or cached in code.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class VoxlineConfig:
    """
    Central configuration for Voxline AI Core.
    
    Load order:
    1. .env file (if exists)
    2. Environment variables
    3. Defaults
    """
    
    # Default values (public, no secrets)
    DEFAULTS = {
        # AI Provider
        "AI_PROVIDER": "qwen",  # qwen|native — default deployment provider
        "AI_MODEL_PATH": "models/Qwen2.5-0.5B-Instruct",
        "AI_TOKENIZER_PATH": "checkpoints/voxline_tokenizer_v0_3.json",
        "AI_DEVICE": "auto",  # auto|cpu|cuda|mps
        "AI_DTYPE": "float32",  # float32|float16|bfloat16
        
        # Workspace
        "WORKSPACE_ROOT": ".",
        "PROJECTS_ROOT": "./projects",
        "MAX_PROJECT_SIZE_MB": "1000",
        
        # Database
        "DATABASE_URL": "sqlite:///./voxline.db",
        "DATABASE_ECHO": "false",
        
        # Logging
        "LOG_LEVEL": "INFO",  # DEBUG|INFO|WARNING|ERROR|CRITICAL
        "LOG_FORMAT": "text",  # text|json
        
        # API
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
        "API_DEBUG": "false",
        
        # Assistant
        "ASSISTANT_NAME": "Voxline",
        "ASSISTANT_DEFAULT_MODE": "chat",  # chat|business|coding
        "ASSISTANT_MAX_HISTORY": "20",
        
        # Agent (coding agent)
        "AGENT_MAX_ITERATIONS": "15",
        "AGENT_STEP_TIMEOUT": "60",
        "AGENT_EXECUTION_POLICY": "safe",  # safe|autonomous
        
        # Workspace (coding agent boundaries)
        "CODING_WORKSPACE_ROOT": ".",
        "CODING_ALLOWED_COMMANDS": "python,pytest,pip,git",
        "CODING_MAX_FILE_SIZE_MB": "10",
        "CODING_MAX_OUTPUT_BYTES": "1048576",

        # Coding Agent workflow
        "CODING_AGENT_MAX_PLAN_STEPS": "10",
        "CODING_AGENT_MAX_CONTEXT_CHARS": "8000",
        "CODING_AGENT_MAX_FIX_ITERATIONS": "3",
        "CODING_AGENT_REQUIRE_APPROVAL_FOR_WRITES": "true",
        
        # Environment
        "ENVIRONMENT": "development",  # development|staging|production
    }
    
    # These keys should never be logged
    SECRET_KEYS = {
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "WEB_SEARCH_API_KEY",
        "INSTAGRAM_ACCESS_TOKEN",
        "INSTAGRAM_ACCOUNT_ID",
        "DATABASE_PASSWORD",
    }
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file (defaults to .env in current dir)
        """
        self._config: Dict[str, str] = {}
        
        # Load .env file if exists
        if env_file is None:
            env_file = Path.cwd() / ".env"
        else:
            env_file = Path(env_file)
        
        if env_file.exists():
            logger.info(f"Loading environment from {env_file}")
            load_dotenv(env_file)
        
        # Load all config
        for key, default in self.DEFAULTS.items():
            value = os.environ.get(key, default)
            self._config[key] = value
        
        # Log non-secret config
        logger.info("Configuration loaded")
        for key, value in self._config.items():
            if key not in self.SECRET_KEYS:
                logger.debug(f"  {key}={value}")
            else:
                logger.debug(f"  {key}=***")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get config value (never logs secrets)."""
        value = self._config.get(key, default)
        if value is None and key in self.DEFAULTS:
            return self.DEFAULTS[key]
        return value
    
    def get_required(self, key: str) -> str:
        """Get config value, raise if missing."""
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required config missing: {key}")
        return value
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get config as integer."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer for {key}: {value}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get config as boolean."""
        value = self.get(key, "").lower()
        return value in ("true", "1", "yes", "on")
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Get config as float."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float for {key}: {value}")
            return default
    
    # Convenience properties
    @property
    def ai_provider(self) -> str:
        return self.get("AI_PROVIDER")
    
    @property
    def ai_model_path(self) -> str:
        return self.get_required("AI_MODEL_PATH")
    
    @property
    def ai_tokenizer_path(self) -> str:
        return self.get_required("AI_TOKENIZER_PATH")
    
    @property
    def ai_device(self) -> str:
        device = self.get("AI_DEVICE")
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    @property
    def workspace_root(self) -> Path:
        return Path(self.get_required("WORKSPACE_ROOT"))
    
    @property
    def projects_root(self) -> Path:
        return Path(self.get_required("PROJECTS_ROOT"))
    
    @property
    def log_level(self) -> str:
        return self.get("LOG_LEVEL")
    
    @property
    def environment(self) -> str:
        return self.get("ENVIRONMENT")
    
    @property
    def database_url(self) -> str:
        return self.get_required("DATABASE_URL")
    
    @property
    def assistant_name(self) -> str:
        return self.get("ASSISTANT_NAME")
    
    @property
    def assistant_default_mode(self) -> str:
        return self.get("ASSISTANT_DEFAULT_MODE")
    
    @property
    def assistant_max_history(self) -> int:
        return self.get_int("ASSISTANT_MAX_HISTORY", 20)
    
    @property
    def agent_max_iterations(self) -> int:
        return self.get_int("AGENT_MAX_ITERATIONS", 15)
    
    @property
    def agent_step_timeout(self) -> int:
        return self.get_int("AGENT_STEP_TIMEOUT", 60)
    
    @property
    def agent_execution_policy(self) -> str:
        return self.get("AGENT_EXECUTION_POLICY")
    
    @property
    def coding_workspace_root(self) -> Path:
        return Path(self.get("CODING_WORKSPACE_ROOT"))
    
    @property
    def coding_allowed_commands(self) -> list:
        raw = self.get("CODING_ALLOWED_COMMANDS", "")
        return [c.strip() for c in raw.split(",") if c.strip()]
    
    @property
    def coding_max_file_size_mb(self) -> int:
        return self.get_int("CODING_MAX_FILE_SIZE_MB", 10)
    
    @property
    def coding_max_output_bytes(self) -> int:
        return self.get_int("CODING_MAX_OUTPUT_BYTES", 1048576)

    @property
    def coding_agent_max_plan_steps(self) -> int:
        return self.get_int("CODING_AGENT_MAX_PLAN_STEPS", 10)

    @property
    def coding_agent_max_context_chars(self) -> int:
        return self.get_int("CODING_AGENT_MAX_CONTEXT_CHARS", 8000)

    @property
    def coding_agent_max_fix_iterations(self) -> int:
        return self.get_int("CODING_AGENT_MAX_FIX_ITERATIONS", 3)

    @property
    def coding_agent_require_approval_for_writes(self) -> bool:
        return self.get_bool("CODING_AGENT_REQUIRE_APPROVAL_FOR_WRITES", True)

    @property
    def api_host(self) -> str:
        return self.get("API_HOST")
    
    @property
    def api_port(self) -> int:
        return self.get_int("API_PORT", 8000)
    
    def is_development(self) -> bool:
        return self.environment == "development"
    
    def is_production(self) -> bool:
        return self.environment == "production"


# Global singleton
_config: Optional[VoxlineConfig] = None


def get_config(env_file: Optional[str] = None) -> VoxlineConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = VoxlineConfig(env_file)
    return _config


def reset_config() -> None:
    """Reset config (for testing)."""
    global _config
    _config = None

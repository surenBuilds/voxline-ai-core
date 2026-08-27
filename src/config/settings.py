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
        "AI_PROVIDER": "qwen",  # qwen|native|openai — default deployment provider
        "AI_MODEL_PATH": "models/Qwen2.5-0.5B-Instruct",
        "AI_TOKENIZER_PATH": "checkpoints/voxline_tokenizer_v0_3.json",
        "AI_DEVICE": "auto",  # auto|cpu|cuda|mps
        "AI_DTYPE": "float32",  # float32|float16|bfloat16

        # Hosted (OpenAI-compatible) provider — used on Vercel/serverless.
        # AI_API_KEY is a secret: server-side only, never committed/browser/logged.
        "AI_API_KEY": "",
        "AI_BASE_URL": "https://api.openai.com/v1",
        "AI_MODEL": "gpt-3.5-turbo",

        
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

        # GitHub integration
        "GITHUB_ENABLED": "false",
        "GITHUB_TOKEN": "",
        "GITHUB_ALLOWED_REPOSITORIES": "",

        # Vercel integration
        "VERCEL_ENABLED": "false",
        "VERCEL_TOKEN": "",
        "VERCEL_ALLOWED_PROJECTS": "",

        # Integration workflow
        "AUTO_CREATE_BRANCH": "true",
        "AUTO_CREATE_PR": "false",
        "AUTO_PREVIEW_DEPLOY": "false",
        "REQUIRE_PRODUCTION_APPROVAL": "true",
        
        # Environment
        "ENVIRONMENT": "development",  # development|staging|production
    }
    
    # These keys should never be logged
    SECRET_KEYS = {
        "OPENAI_API_KEY",
        "AI_API_KEY",
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
    def ai_api_key(self) -> str:
        return self.get("AI_API_KEY", "")

    @property
    def ai_base_url(self) -> str:
        return self.get("AI_BASE_URL", "https://api.openai.com/v1")

    @property
    def ai_model(self) -> str:
        return self.get("AI_MODEL", "gpt-3.5-turbo")
    
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
    def github_enabled(self) -> bool:
        return self.get_bool("GITHUB_ENABLED", False)

    @property
    def github_token(self) -> str:
        return self.get("GITHUB_TOKEN", "")

    @property
    def github_allowed_repositories(self) -> list:
        raw = self.get("GITHUB_ALLOWED_REPOSITORIES", "")
        return [r.strip() for r in raw.split(",") if r.strip()]

    @property
    def vercel_enabled(self) -> bool:
        return self.get_bool("VERCEL_ENABLED", False)

    @property
    def vercel_token(self) -> str:
        return self.get("VERCEL_TOKEN", "")

    @property
    def vercel_allowed_projects(self) -> list:
        raw = self.get("VERCEL_ALLOWED_PROJECTS", "")
        return [p.strip() for p in raw.split(",") if p.strip()]

    @property
    def auto_create_branch(self) -> bool:
        return self.get_bool("AUTO_CREATE_BRANCH", True)

    @property
    def auto_create_pr(self) -> bool:
        return self.get_bool("AUTO_CREATE_PR", False)

    @property
    def auto_preview_deploy(self) -> bool:
        return self.get_bool("AUTO_PREVIEW_DEPLOY", False)

    @property
    def require_production_approval(self) -> bool:
        return self.get_bool("REQUIRE_PRODUCTION_APPROVAL", True)

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

    def validate(self) -> list:
        """Validate configuration and return list of warnings/errors.

        Returns empty list if configuration is valid.
        Never logs or returns secret values.
        """
        issues = []

        if self.github_enabled:
            token = self.github_token
            if not token:
                issues.append(
                    "GITHUB_ENABLED=true but GITHUB_TOKEN is not set. "
                    "GitHub tools will not be available."
                )
            allowed = self.github_allowed_repositories
            if allowed:
                for repo in allowed:
                    if "/" not in repo:
                        issues.append(
                            f"GitHub allowed repository '{repo}' does not "
                            f"match expected format 'owner/repo'."
                        )

        if self.vercel_enabled:
            token = self.vercel_token
            if not token:
                issues.append(
                    "VERCEL_ENABLED=true but VERCEL_TOKEN is not set. "
                    "Vercel tools will not be available."
                )

        timeout = self.agent_step_timeout
        if timeout < 10:
            issues.append(
                f"AGENT_STEP_TIMEOUT={timeout}s is very low. "
                f"Recommend >= 30s."
            )

        max_fix = self.coding_agent_max_fix_iterations
        if max_fix > 10:
            issues.append(
                f"CODING_AGENT_MAX_FIX_ITERATIONS={max_fix} is very high. "
                f"Recommend <= 5."
            )

        return issues


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

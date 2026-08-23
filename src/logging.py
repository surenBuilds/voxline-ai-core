"""
Structured logging for Voxline AI Core.

Provides unified logging with:
- JSON output option
- Secret filtering
- Component tracking
- No automatic secret logging
"""

import logging
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path


# Never log these keys
SECRET_PATTERNS = {
    "api_key",
    "token",
    "password",
    "secret",
    "credential",
    "bearer",
}


class SecretFilteringFormatter(logging.Formatter):
    """Formatter that redacts sensitive information."""
    
    @staticmethod
    def _redact_dict(obj: Any, depth: int = 0) -> Any:
        """Recursively redact secrets from dict/list."""
        if depth > 10:  # Prevent infinite recursion
            return obj
        
        if isinstance(obj, dict):
            return {
                k: "***" if any(pattern in k.lower() for pattern in SECRET_PATTERNS) 
                     else SecretFilteringFormatter._redact_dict(v, depth + 1)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return type(obj)(
                SecretFilteringFormatter._redact_dict(item, depth + 1) 
                for item in obj
            )
        return obj
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record, redacting secrets."""
        # Redact message if it might contain secrets
        if "password" in record.getMessage().lower() or "key" in record.getMessage().lower():
            record.msg = "***"
            record.args = ()
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add optional fields
        if record.funcName:
            log_data["function"] = record.funcName
        if record.lineno:
            log_data["line"] = record.lineno
        
        # Add exception if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, "component"):
            log_data["component"] = record.component
        if hasattr(record, "event"):
            log_data["event"] = record.event
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        return json.dumps(log_data)


class StructuredLogger:
    """Wrapper for structured logging."""
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.name = name
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal log method."""
        # Create log record with extra fields
        extra = {
            "component": kwargs.pop("component", None),
            "event": kwargs.pop("event", None),
            "duration_ms": kwargs.pop("duration_ms", None),
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        self.logger.log(level, message, extra=extra)


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging for Voxline AI Core.
    
    Args:
        level: Log level (DEBUG|INFO|WARNING|ERROR|CRITICAL)
        format_type: Format type (text|json)
        log_file: Optional log file path
    """
    # Parse level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Choose formatter
    if format_type.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        formatter = SecretFilteringFormatter(formatter._fmt, formatter.datefmt)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging configured: level={level}, format={format_type}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

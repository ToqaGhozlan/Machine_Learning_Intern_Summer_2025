"""
Logging configuration for the Taxi Fare Prediction API.

This module sets up comprehensive logging with proper formatting,
levels, and output destinations.
"""

import logging
import logging.config
import sys
from typing import Dict, Any
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """
    Setup application logging configuration.
    
    Configures logging with appropriate handlers, formatters, and levels
    based on the application environment.
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Define logging configuration
    logging_config = get_logging_config()
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {settings.LOG_LEVEL.upper()}")


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration dictionary.
    
    Returns:
        Dictionary with logging configuration for dictConfig
    """
    
    # Base formatter
    detailed_formatter = {
        "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S"
    }
    
    simple_formatter = {
        "format": "%(levelname)s: %(message)s"
    }
    
    # Console handler
    console_handler = {
        "class": "logging.StreamHandler",
        "level": settings.LOG_LEVEL.upper(),
        "formatter": "detailed" if settings.is_development else "simple",
        "stream": sys.stdout
    }
    
    # File handler (if log file specified)
    handlers = {
        "console": console_handler
    }
    
    if settings.LOG_FILE:
        file_handler = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": settings.LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
        handlers["file"] = file_handler
    
    # Error file handler for production
    if settings.is_production:
        error_handler = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/error.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "encoding": "utf8"
        }
        handlers["error_file"] = error_handler
    
    # Logger configuration
    loggers = {
        "": {  # Root logger
            "level": settings.LOG_LEVEL.upper(),
            "handlers": list(handlers.keys()),
            "propagate": False
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn.access": {
            "level": "INFO" if settings.is_development else "WARNING",
            "handlers": ["console"],
            "propagate": False
        },
        "fastapi": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    }
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": detailed_formatter,
            "simple": simple_formatter
        },
        "handlers": handlers,
        "loggers": loggers
    }


class RequestLogger:
    """Custom request logging middleware."""
    
    def __init__(self):
        self.logger = logging.getLogger("api.requests")
    
    def log_request(self, method: str, url: str, status_code: int, 
                   response_time: float, client_ip: str = None):
        """Log API request details."""
        self.logger.info(
            f"{method} {url} - {status_code} - {response_time:.3f}s"
            f"{f' - {client_ip}' if client_ip else ''}"
        )
    
    def log_prediction(self, prediction_time: float, confidence: float, 
                      fare: float, status: str):
        """Log prediction details."""
        self.logger.info(
            f"Prediction: ${fare:.2f} (confidence: {confidence:.1f}%) "
            f"- {prediction_time:.3f}s - {status}"
        )


class ModelLogger:
    """Specialized logger for model operations."""
    
    def __init__(self):
        self.logger = logging.getLogger("model")
    
    def log_model_load(self, model_name: str, model_size: str, load_time: float):
        """Log model loading."""
        self.logger.info(
            f"Model loaded: {model_name} ({model_size}) in {load_time:.2f}s"
        )
    
    def log_prediction_error(self, error: str, inputs: dict):
        """Log prediction errors."""
        self.logger.error(
            f"Prediction failed: {error} | Inputs: {inputs}"
        )
    
    def log_health_check(self, status: str, details: dict):
        """Log health check results."""
        if status == "healthy":
            self.logger.info(f"Health check passed: {details}")
        else:
            self.logger.warning(f"Health check failed: {details}")


# Create logger instances
request_logger = RequestLogger()
model_logger = ModelLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_startup_info():
    """Log application startup information."""
    logger = get_logger("startup")
    
    logger.info("=" * 60)
    logger.info("🚖 TAXI FARE PREDICTION API")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"Log Level: {settings.LOG_LEVEL.upper()}")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"Workers: {settings.WORKERS}")
    logger.info(f"Model Path: {settings.MODEL_PATH}")
    logger.info("=" * 60)

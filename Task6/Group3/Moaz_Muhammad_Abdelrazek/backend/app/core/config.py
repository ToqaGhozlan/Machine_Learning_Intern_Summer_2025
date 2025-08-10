"""
Configuration management for the Taxi Fare Prediction API.

This module handles all application configuration including environment variables,
default settings, and configuration validation.
"""

import os
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"
    DEBUG: bool = False
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Taxi Fare Prediction API"
    VERSION: str = "1.0.0"
    
    # CORS settings - Updated for separate Render deployments
    CORS_ORIGINS: str = "*"  # Changed to string to avoid JSON parsing issues
    
    def get_cors_origins_list(self) -> List[str]:
        """Convert CORS_ORIGINS string to list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    # Model settings
    MODEL_PATH: str = "./models/best_taxi_fare_model.pkl"
    PROCESSOR_PATH: str = "./models/feature_processor.pkl"
    METADATA_PATH: str = "./models/final_model_metadata.json"
    
    # Server settings - Optimized for Render free tier
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", 8000))  # Render provides PORT env var
    WORKERS: int = int(os.getenv("WORKERS", 1))  # Single worker for free tier
    
    # Performance settings
    MAX_PREDICTION_TIME: float = 5.0  # seconds
    CACHE_PREDICTIONS: bool = False
    CACHE_TTL: int = 300  # seconds
    
    # Monitoring settings
    ENABLE_METRICS: bool = True
    METRICS_PATH: str = "/metrics"
    HEALTH_CHECK_PATH: str = "/health"
    
    # Logging settings
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    
    # Security settings
    ALLOWED_HOSTS: List[str] = ["*"]
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level setting."""
        allowed = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.lower()
    
    @validator("MODEL_PATH")
    def validate_model_path(cls, v):
        """Validate model path exists."""
        if not os.path.exists(v):
            # In development, this might not exist yet
            if os.getenv("ENVIRONMENT", "development") == "development":
                return v
            raise ValueError(f"Model file not found: {v}")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "debug"
    WORKERS: int = 1


class ProductionSettings(Settings):
    """Production-specific settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "info"
    WORKERS: int = 4
    RATE_LIMIT_ENABLED: bool = True
    CACHE_PREDICTIONS: bool = True


class TestingSettings(Settings):
    """Testing-specific settings."""
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    LOG_LEVEL: str = "debug"
    MODEL_PATH: str = "./tests/fixtures/test_model.pkl"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings based on environment.
    
    This function is cached to avoid re-reading environment variables
    on every call.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export settings instance for convenience
settings = get_settings()


def get_database_url() -> str:
    """Get database URL (placeholder for future database integration)."""
    return os.getenv("DATABASE_URL", "sqlite:///./taxi_predictions.db")


def get_redis_url() -> str:
    """Get Redis URL for caching (placeholder for future caching)."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_model_config() -> dict:
    """Get model-specific configuration."""
    return {
        "model_path": settings.MODEL_PATH,
        "processor_path": settings.PROCESSOR_PATH,
        "metadata_path": settings.METADATA_PATH,
        "max_prediction_time": settings.MAX_PREDICTION_TIME,
        "cache_enabled": settings.CACHE_PREDICTIONS,
        "cache_ttl": settings.CACHE_TTL
    }


def get_api_config() -> dict:
    """Get API-specific configuration."""
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "debug": settings.DEBUG,
        "cors_origins": settings.CORS_ORIGINS,
        "api_prefix": settings.API_V1_STR
    }

"""
Health check and monitoring routes for the Taxi Fare Prediction API.

This module provides endpoints for monitoring application health,
performance metrics, and system status.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import psutil
import time
from datetime import datetime

from app.models.ml_model import ModelManager
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Track application start time
app_start_time = time.time()


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    environment: str


class DetailedHealthResponse(BaseModel):
    """Detailed health response with system metrics."""
    
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    environment: str
    model_status: Dict[str, Any]
    system_metrics: Dict[str, Any]
    api_metrics: Dict[str, Any]


def get_model_manager() -> ModelManager:
    """Dependency to get model manager (will be overridden by main app)."""
    pass


def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage_percent": round(cpu_percent, 2),
            "memory_usage_percent": round(memory.percent, 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_usage_percent": round(disk.percent, 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2)
        }
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
        return {"error": "System metrics unavailable"}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns basic application health status and uptime information.
    This endpoint is designed to be lightweight for load balancer checks.
    """
    try:
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=round(uptime, 2),
            version="1.0.0",
            environment=settings.ENVIRONMENT
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable"
        )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    model_manager: ModelManager = Depends(get_model_manager)
) -> DetailedHealthResponse:
    """
    Detailed health check with comprehensive system and model metrics.
    
    Returns detailed information about application health, model status,
    system performance, and API metrics.
    """
    try:
        uptime = time.time() - app_start_time
        
        # Get model health status
        model_health = model_manager.health_check()
        
        # Get system metrics
        system_metrics = get_system_metrics()
        
        # Get model statistics
        model_stats = model_manager.get_statistics()
        
        # Determine overall status
        overall_status = "healthy"
        if model_health.get("status") != "healthy":
            overall_status = "degraded"
        
        if system_metrics.get("cpu_usage_percent", 0) > 90:
            overall_status = "degraded"
        
        if system_metrics.get("memory_usage_percent", 0) > 90:
            overall_status = "degraded"
        
        return DetailedHealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=round(uptime, 2),
            version="1.0.0",
            environment=settings.ENVIRONMENT,
            model_status=model_health,
            system_metrics=system_metrics,
            api_metrics={
                "total_predictions": model_stats["prediction_count"],
                "successful_predictions": model_stats["success_count"],
                "failed_predictions": model_stats["error_count"],
                "success_rate_percent": model_stats["success_rate_percent"],
                "avg_prediction_time_ms": model_stats["avg_prediction_time_ms"]
            }
        )
    
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable"
        )


@router.get("/health/model")
async def model_health_check(
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Model-specific health check.
    
    Returns detailed information about the machine learning model
    status, performance, and capabilities.
    """
    try:
        return model_manager.health_check()
    
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Model health check failed"
        )


@router.get("/metrics")
async def get_metrics(
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive application metrics.
    
    Returns detailed metrics about API performance, model statistics,
    and system resource usage.
    """
    try:
        uptime = time.time() - app_start_time
        
        # Get model statistics
        model_stats = model_manager.get_statistics()
        
        # Get system metrics
        system_metrics = get_system_metrics()
        
        return {
            "application": {
                "uptime_seconds": round(uptime, 2),
                "uptime_formatted": format_uptime(uptime),
                "version": "1.0.0",
                "environment": settings.ENVIRONMENT,
                "start_time": datetime.fromtimestamp(app_start_time).isoformat()
            },
            "api_performance": {
                "total_requests": model_stats["prediction_count"],
                "successful_requests": model_stats["success_count"],
                "failed_requests": model_stats["error_count"],
                "success_rate_percent": model_stats["success_rate_percent"],
                "error_rate_percent": model_stats["error_rate_percent"],
                "avg_response_time_ms": model_stats["avg_prediction_time_ms"],
                "total_processing_time_s": model_stats["total_prediction_time_s"]
            },
            "model_metrics": {
                "model_name": "Random Forest (Default Parameters)",
                "model_size": model_stats["model_size"],
                "model_uptime": model_stats["uptime"],
                "predictions_served": model_stats["prediction_count"]
            },
            "system_resources": system_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve metrics"
        )


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """
    Get simple application status.
    
    Returns a lightweight status response suitable for monitoring
    and alerting systems.
    """
    try:
        uptime = time.time() - app_start_time
        
        return {
            "status": "operational",
            "service": "Taxi Fare Prediction API",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "service": "Taxi Fare Prediction API",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def format_uptime(uptime_seconds: float) -> str:
    """Format uptime in human-readable format."""
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


@router.get("/ping")
async def ping() -> Dict[str, str]:
    """
    Simple ping endpoint for basic connectivity testing.
    
    Returns a simple pong response to verify the API is reachable.
    """
    return {
        "message": "pong",
        "timestamp": datetime.now().isoformat(),
        "service": "Taxi Fare Prediction API"
    }

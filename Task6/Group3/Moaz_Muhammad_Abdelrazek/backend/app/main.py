"""
Taxi Fare Prediction API - FastAPI Application

A production-ready FastAPI application for serving taxi fare predictions
using a trained Random Forest machine learning model.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.routes import prediction, health, model_info
from app.models.ml_model import ModelManager

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title="Taxi Fare Prediction API",
    description="Professional API for predicting taxi fares using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Global model manager instance
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global model_manager
    
    logger.info("Starting Taxi Fare Prediction API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    
    try:
        # Initialize model manager
        model_manager = ModelManager(
            model_path=settings.MODEL_PATH,
            processor_path=settings.PROCESSOR_PATH,
            metadata_path=settings.METADATA_PATH
        )
        
        # Perform health check
        health_status = model_manager.health_check()
        if health_status["status"] != "healthy":
            raise Exception(f"Model health check failed: {health_status}")
        
        logger.info("Model loaded successfully")
        logger.info(f"Model: {model_manager.get_model_info()['model_name']}")
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Taxi Fare Prediction API...")

# Dependency to get model manager
def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global model_manager
    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Please try again later."
        )
    return model_manager

# Override dependencies
app.dependency_overrides[prediction.get_model_manager] = get_model_manager
app.dependency_overrides[health.get_model_manager] = get_model_manager
app.dependency_overrides[model_info.get_model_manager] = get_model_manager

# Include API routes
app.include_router(
    prediction.router,
    prefix="/api/v1",
    tags=["predictions"]
)

app.include_router(
    health.router,
    prefix="",
    tags=["health"]
)

app.include_router(
    model_info.router,
    prefix="/api/v1/model",
    tags=["model"]
)

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Taxi Fare Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/health",
        "model_info": "/api/v1/model/info"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )

"""
Model information and metadata routes for the Taxi Fare Prediction API.

This module provides endpoints for retrieving detailed information
about the machine learning model, its capabilities, and performance metrics.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.models.ml_model import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_name: str
    model_type: str
    version: str
    required_features: int
    feature_names: List[str]
    has_feature_processor: bool
    deployment_ready: bool
    performance_metrics: Dict[str, Any]
    deployment_info: Dict[str, Any]


class ModelCapabilitiesResponse(BaseModel):
    """Response model for model capabilities."""
    
    input_parameters: Dict[str, str]
    output_format: Dict[str, str]
    supported_features: List[str]
    constraints: Dict[str, Any]
    performance_characteristics: Dict[str, Any]


def get_model_manager() -> ModelManager:
    """Dependency to get model manager (will be overridden by main app)."""
    pass


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_manager: ModelManager = Depends(get_model_manager)
) -> ModelInfoResponse:
    """
    Get comprehensive model information.
    
    Returns detailed information about the loaded machine learning model
    including metadata, performance metrics, and deployment details.
    """
    try:
        model_info = model_manager.get_model_info()
        
        if "error" in model_info:
            raise HTTPException(
                status_code=503,
                detail=f"Model information unavailable: {model_info['error']}"
            )
        
        return ModelInfoResponse(**model_info)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )


@router.get("/capabilities", response_model=ModelCapabilitiesResponse)
async def get_model_capabilities() -> ModelCapabilitiesResponse:
    """
    Get model capabilities and API specification.
    
    Returns information about what inputs the model accepts,
    what outputs it provides, and its operational constraints.
    """
    try:
        return ModelCapabilitiesResponse(
            input_parameters={
                "pickup_latitude": "float (-90 to 90) - Pickup location latitude",
                "pickup_longitude": "float (-180 to 180) - Pickup location longitude",
                "dropoff_latitude": "float (-90 to 90) - Dropoff location latitude",
                "dropoff_longitude": "float (-180 to 180) - Dropoff location longitude",
                "passenger_count": "int (1 to 8) - Number of passengers",
                "pickup_datetime": "string (ISO format) - Trip start time",
                "weather_condition": "string (optional) - Weather: sunny, cloudy, windy, stormy",
                "traffic_condition": "string (optional) - Traffic: flow traffic, congested traffic"
            },
            output_format={
                "predicted_fare": "float - Predicted fare amount in USD",
                "confidence": "float (0-100) - Prediction confidence score",
                "model_name": "string - Name of the ML model used",
                "model_type": "string - Type of the ML model",
                "prediction_timestamp": "string - ISO timestamp of prediction",
                "prediction_time": "float - Processing time in milliseconds",
                "status": "string - success or error",
                "error_message": "string (optional) - Error details if failed",
                "api_version": "string - API version used"
            },
            supported_features=[
                "Real-time fare prediction",
                "Confidence scoring",
                "Weather condition adjustment",
                "Traffic condition adjustment",
                "Time-based pricing",
                "Location-based pricing",
                "Airport trip detection",
                "Manhattan area detection",
                "Rush hour pricing",
                "Distance-based calculation"
            ],
            constraints={
                "coordinate_bounds": {
                    "latitude": {"min": -90, "max": 90},
                    "longitude": {"min": -180, "max": 180}
                },
                "passenger_limits": {"min": 1, "max": 8},
                "datetime_format": "ISO 8601 (e.g., '2024-01-15T14:30:00')",
                "weather_options": ["sunny", "cloudy", "windy", "stormy"],
                "traffic_options": ["flow traffic", "congested traffic"],
                "minimum_trip_distance": "0.1 miles",
                "maximum_prediction_time": "5 seconds"
            },
            performance_characteristics={
                "accuracy_r2": 0.991,
                "mean_absolute_error_usd": 0.041,
                "rmse_usd": 1.054,
                "accuracy_within_2_dollars_percent": 99.8,
                "typical_response_time_ms": "< 100",
                "model_size_mb": 486.5,
                "supported_requests_per_second": "> 100"
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to get model capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model capabilities"
        )


@router.get("/performance")
async def get_model_performance(
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Get model performance metrics and statistics.
    
    Returns current performance statistics including prediction counts,
    response times, and accuracy metrics.
    """
    try:
        stats = model_manager.get_statistics()
        
        return {
            "current_session": stats,
            "model_benchmarks": {
                "r2_score": 0.991023,
                "mae_usd": 0.041,
                "rmse_usd": 1.054,
                "mape_percent": 0.75,
                "accuracy_within_1_dollar_percent": 99.5,
                "accuracy_within_2_dollars_percent": 99.8,
                "max_error_usd": 286.39
            },
            "performance_targets": {
                "target_response_time_ms": 100,
                "target_accuracy_percent": 95,
                "target_uptime_percent": 99.9,
                "target_error_rate_percent": 1
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model performance metrics"
        )


@router.get("/features")
async def get_model_features(
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Get information about model features and their importance.
    
    Returns details about the features used by the model
    and their relative importance for predictions.
    """
    try:
        model_info = model_manager.get_model_info()
        
        if "error" in model_info:
            raise HTTPException(
                status_code=503,
                detail=f"Model information unavailable: {model_info['error']}"
            )
        
        return {
            "total_features": model_info["required_features"],
            "feature_names": model_info["feature_names"],
            "feature_categories": {
                "location_features": [
                    "pickup_latitude_scaled",
                    "pickup_longitude_scaled",
                    "dropoff_latitude_scaled",
                    "dropoff_longitude_scaled",
                    "is_manhattan",
                    "is_airport_trip",
                    "min_airport_dist_scaled"
                ],
                "temporal_features": [
                    "hour",
                    "day_of_week",
                    "month",
                    "is_weekend",
                    "is_rush_hour",
                    "is_morning_rush",
                    "is_evening_rush"
                ],
                "trip_features": [
                    "passenger_count",
                    "trip_distance_scaled",
                    "fare_per_mile_scaled"
                ],
                "external_features": [
                    "weather_severity",
                    "traffic_severity"
                ]
            },
            "feature_descriptions": {
                "pickup_latitude_scaled": "Normalized pickup latitude coordinate",
                "pickup_longitude_scaled": "Normalized pickup longitude coordinate",
                "dropoff_latitude_scaled": "Normalized dropoff latitude coordinate",
                "dropoff_longitude_scaled": "Normalized dropoff longitude coordinate",
                "passenger_count": "Number of passengers (1-8)",
                "hour": "Hour of day (0-23)",
                "day_of_week": "Day of week (0=Monday, 6=Sunday)",
                "month": "Month of year (1-12)",
                "is_weekend": "Binary flag for weekend trips",
                "is_rush_hour": "Binary flag for rush hour periods",
                "is_morning_rush": "Binary flag for morning rush (7-9 AM)",
                "is_evening_rush": "Binary flag for evening rush (5-7 PM)",
                "trip_distance_scaled": "Normalized trip distance in miles",
                "fare_per_mile_scaled": "Normalized fare per mile rate",
                "is_airport_trip": "Binary flag for airport-related trips",
                "min_airport_dist_scaled": "Normalized minimum distance to any airport",
                "is_manhattan": "Binary flag for Manhattan area trips",
                "weather_severity": "Weather condition severity (1-4)",
                "traffic_severity": "Traffic condition severity (1-2)"
            },
            "preprocessing_info": {
                "scaling_method": "StandardScaler and MinMaxScaler",
                "coordinate_normalization": "NYC-centered normalization",
                "distance_calculation": "Haversine formula",
                "temporal_encoding": "Cyclical and binary encoding"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model features: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model feature information"
        )


@router.get("/version")
async def get_model_version() -> Dict[str, Any]:
    """
    Get model version and deployment information.
    
    Returns version information, deployment date, and model lineage.
    """
    return {
        "model_version": "1.0.0",
        "api_version": "1.0.0",
        "deployment_date": "2025-01-30",
        "model_type": "RandomForestRegressor",
        "training_date": "2025-07-30",
        "framework": "scikit-learn",
        "python_version": "3.7+",
        "model_lineage": {
            "base_algorithm": "Random Forest",
            "hyperparameter_tuning": "GridSearchCV",
            "feature_engineering": "Custom pipeline",
            "validation_method": "Time-series split",
            "selection_criteria": "R² score and MAE"
        },
        "deployment_info": {
            "environment": "Production Ready",
            "container_ready": True,
            "api_framework": "FastAPI",
            "model_serving": "In-memory",
            "scalability": "Horizontal scaling supported"
        }
    }

"""
Prediction API routes for the Taxi Fare Prediction service.

This module defines the main prediction endpoints with comprehensive
input validation, error handling, and response formatting.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.models.ml_model import ModelManager
from app.core.logging import request_logger

logger = logging.getLogger(__name__)
router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for taxi fare prediction."""
    
    pickup_latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Pickup latitude coordinate (-90 to 90)"
    )
    pickup_longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Pickup longitude coordinate (-180 to 180)"
    )
    dropoff_latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Dropoff latitude coordinate (-90 to 90)"
    )
    dropoff_longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Dropoff longitude coordinate (-180 to 180)"
    )
    passenger_count: int = Field(
        ...,
        ge=1,
        le=8,
        description="Number of passengers (1-8)"
    )
    pickup_datetime: str = Field(
        ...,
        description="Pickup datetime in ISO format (e.g., '2024-01-15T14:30:00')"
    )
    weather_condition: Optional[str] = Field(
        "sunny",
        description="Weather condition: sunny, cloudy, windy, stormy"
    )
    traffic_condition: Optional[str] = Field(
        "flow traffic",
        description="Traffic condition: flow traffic, congested traffic"
    )
    
    @validator("weather_condition")
    def validate_weather(cls, v):
        """Validate weather condition."""
        if v is None:
            return "sunny"
        allowed = ["sunny", "cloudy", "windy", "stormy"]
        if v.lower() not in allowed:
            raise ValueError(f"Weather condition must be one of: {allowed}")
        return v.lower()
    
    @validator("traffic_condition")
    def validate_traffic(cls, v):
        """Validate traffic condition."""
        if v is None:
            return "flow traffic"
        allowed = ["flow traffic", "congested traffic"]
        if v.lower() not in allowed:
            raise ValueError(f"Traffic condition must be one of: {allowed}")
        return v.lower()
    
    @validator("pickup_datetime")
    def validate_datetime(cls, v):
        """Validate datetime format."""
        try:
            # Try to parse the datetime to ensure it's valid
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Invalid datetime format. Use ISO format (e.g., '2024-01-15T14:30:00')")
    
    def validate_coordinates(self):
        """Validate that pickup and dropoff are not the same."""
        if (abs(self.pickup_latitude - self.dropoff_latitude) < 0.0001 and
            abs(self.pickup_longitude - self.dropoff_longitude) < 0.0001):
            raise ValueError("Pickup and dropoff locations cannot be the same")


class PredictionResponse(BaseModel):
    """Response model for taxi fare prediction."""
    
    predicted_fare: Optional[float] = Field(
        description="Predicted fare amount in USD"
    )
    confidence: float = Field(
        description="Prediction confidence score (0-100)"
    )
    model_name: str = Field(
        description="Name of the ML model used"
    )
    model_type: Optional[str] = Field(
        description="Type of the ML model"
    )
    prediction_timestamp: str = Field(
        description="Timestamp when prediction was made"
    )
    prediction_time: Optional[float] = Field(
        description="Time taken for prediction in milliseconds"
    )
    status: str = Field(
        description="Prediction status: success or error"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if prediction failed"
    )
    input_validation: Optional[str] = Field(
        description="Input validation status"
    )
    api_version: Optional[str] = Field(
        description="API version used"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    predictions: list[PredictionRequest] = Field(
        ...,
        max_items=100,
        description="List of prediction requests (max 100)"
    )


def get_model_manager() -> ModelManager:
    """Dependency to get model manager (will be overridden by main app)."""
    # This will be overridden by dependency_overrides in main.py
    raise HTTPException(status_code=503, detail="Model manager not available")


@router.post("/predict", response_model=PredictionResponse)
async def predict_fare(
    request: PredictionRequest,
    http_request: Request,
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionResponse:
    """
    Predict taxi fare for a single trip.
    
    This endpoint accepts trip details and returns a fare prediction
    with confidence score and additional metadata.
    """
    start_time = datetime.now()
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        # Additional validation
        request.validate_coordinates()
        
        logger.info(f"Prediction request from {client_ip}")
        logger.debug(f"Request data: {request.dict()}")
        
        # Make prediction
        result = model_manager.predict(
            pickup_latitude=request.pickup_latitude,
            pickup_longitude=request.pickup_longitude,
            dropoff_latitude=request.dropoff_latitude,
            dropoff_longitude=request.dropoff_longitude,
            passenger_count=request.passenger_count,
            pickup_datetime=request.pickup_datetime,
            weather_condition=request.weather_condition,
            traffic_condition=request.traffic_condition
        )
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Log request
        request_logger.log_request(
            method="POST",
            url="/api/v1/predict",
            status_code=200 if result["status"] == "success" else 400,
            response_time=response_time,
            client_ip=client_ip
        )
        
        # Return response
        response = PredictionResponse(**result)
        
        if result["status"] == "success":
            logger.info(f"Successful prediction: ${result['predicted_fare']:.2f}")
        else:
            logger.warning(f"Prediction failed: {result.get('error_message')}")
        
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@router.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_fare_batch(
    request: BatchPredictionRequest,
    http_request: Request,
    model_manager: ModelManager = Depends(get_model_manager)
) -> list[PredictionResponse]:
    """
    Predict taxi fares for multiple trips in batch.
    
    This endpoint accepts multiple trip requests and returns
    predictions for all of them.
    """
    start_time = datetime.now()
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        logger.info(f"Batch prediction request from {client_ip} ({len(request.predictions)} items)")
        
        results = []
        
        for i, pred_request in enumerate(request.predictions):
            try:
                # Validate coordinates for each request
                pred_request.validate_coordinates()
                
                # Make prediction
                result = model_manager.predict(
                    pickup_latitude=pred_request.pickup_latitude,
                    pickup_longitude=pred_request.pickup_longitude,
                    dropoff_latitude=pred_request.dropoff_latitude,
                    dropoff_longitude=pred_request.dropoff_longitude,
                    passenger_count=pred_request.passenger_count,
                    pickup_datetime=pred_request.pickup_datetime,
                    weather_condition=pred_request.weather_condition,
                    traffic_condition=pred_request.traffic_condition
                )
                
                results.append(PredictionResponse(**result))
                
            except Exception as e:
                logger.error(f"Batch prediction item {i} failed: {e}")
                # Add error result for this item
                error_result = {
                    "predicted_fare": None,
                    "confidence": 0.0,
                    "model_name": "Random Forest (Default Parameters)",
                    "prediction_timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error_message": str(e),
                    "api_version": "1.0.0"
                }
                results.append(PredictionResponse(**error_result))
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Log request
        request_logger.log_request(
            method="POST",
            url="/api/v1/predict/batch",
            status_code=200,
            response_time=response_time,
            client_ip=client_ip
        )
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch prediction"
        )


@router.get("/predict/example")
async def get_prediction_example() -> Dict[str, Any]:
    """
    Get an example prediction request for API documentation.
    
    Returns a sample request that can be used to test the prediction endpoint.
    """
    return {
        "example_request": {
            "pickup_latitude": 40.7589,
            "pickup_longitude": -73.9851,
            "dropoff_latitude": 40.7505,
            "dropoff_longitude": -73.9934,
            "passenger_count": 2,
            "pickup_datetime": "2024-01-15T14:30:00",
            "weather_condition": "sunny",
            "traffic_condition": "flow traffic"
        },
        "expected_response": {
            "predicted_fare": 12.45,
            "confidence": 87.3,
            "model_name": "Random Forest (Default Parameters)",
            "status": "success"
        },
        "curl_example": """
curl -X POST "http://localhost:8000/api/v1/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "pickup_latitude": 40.7589,
       "pickup_longitude": -73.9851,
       "dropoff_latitude": 40.7505,
       "dropoff_longitude": -73.9934,
       "passenger_count": 2,
       "pickup_datetime": "2024-01-15T14:30:00",
       "weather_condition": "sunny",
       "traffic_condition": "flow traffic"
     }'
        """.strip()
    }

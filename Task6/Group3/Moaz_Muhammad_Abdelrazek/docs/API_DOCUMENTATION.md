# 🚖 Taxi Fare Prediction API Documentation

## Overview

The Taxi Fare Prediction API provides real-time fare predictions using a trained Random Forest machine learning model with exceptional accuracy (R² = 0.991, MAE = $0.041).

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for API access.

## Rate Limiting

- **Development**: No rate limiting
- **Production**: 100 requests per minute per IP

## API Endpoints

### 1. Prediction Endpoints

#### POST /api/v1/predict

Predict taxi fare for a single trip.

**Request Body:**
```json
{
  "pickup_latitude": 40.7589,
  "pickup_longitude": -73.9851,
  "dropoff_latitude": 40.7505,
  "dropoff_longitude": -73.9934,
  "passenger_count": 2,
  "pickup_datetime": "2024-01-15T14:30:00",
  "weather_condition": "sunny",
  "traffic_condition": "flow traffic"
}
```

**Response:**
```json
{
  "predicted_fare": 12.45,
  "confidence": 87.3,
  "model_name": "Random Forest (Default Parameters)",
  "model_type": "RandomForestRegressor",
  "prediction_timestamp": "2024-01-15T14:30:15.123456",
  "prediction_time": 45.2,
  "status": "success",
  "api_version": "1.0.0"
}
```

**Parameters:**

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| pickup_latitude | float | Yes | Pickup latitude | -90 to 90 |
| pickup_longitude | float | Yes | Pickup longitude | -180 to 180 |
| dropoff_latitude | float | Yes | Dropoff latitude | -90 to 90 |
| dropoff_longitude | float | Yes | Dropoff longitude | -180 to 180 |
| passenger_count | int | Yes | Number of passengers | 1 to 8 |
| pickup_datetime | string | Yes | Pickup datetime | ISO format |
| weather_condition | string | No | Weather condition | sunny, cloudy, windy, stormy |
| traffic_condition | string | No | Traffic condition | flow traffic, congested traffic |

#### POST /api/v1/predict/batch

Predict taxi fares for multiple trips.

**Request Body:**
```json
{
  "predictions": [
    {
      "pickup_latitude": 40.7589,
      "pickup_longitude": -73.9851,
      "dropoff_latitude": 40.7505,
      "dropoff_longitude": -73.9934,
      "passenger_count": 2,
      "pickup_datetime": "2024-01-15T14:30:00"
    }
  ]
}
```

**Response:**
```json
[
  {
    "predicted_fare": 12.45,
    "confidence": 87.3,
    "status": "success"
  }
]
```

**Constraints:**
- Maximum 100 predictions per batch request

#### GET /api/v1/predict/example

Get example prediction request for testing.

**Response:**
```json
{
  "example_request": {
    "pickup_latitude": 40.7589,
    "pickup_longitude": -73.9851,
    "dropoff_latitude": 40.7505,
    "dropoff_longitude": -73.9934,
    "passenger_count": 2,
    "pickup_datetime": "2024-01-15T14:30:00"
  },
  "curl_example": "curl -X POST ..."
}
```

### 2. Model Information Endpoints

#### GET /api/v1/model/info

Get comprehensive model information.

**Response:**
```json
{
  "model_name": "Random Forest (Default Parameters)",
  "model_type": "RandomForestRegressor",
  "version": "1.0.0",
  "required_features": 19,
  "feature_names": ["pickup_latitude_scaled", "..."],
  "performance_metrics": {
    "test_r2": 0.991023,
    "test_mae": 0.041,
    "test_rmse": 1.054
  },
  "deployment_info": {
    "model_size": "486.5MB",
    "prediction_count": 1234,
    "uptime": "02:15:30"
  }
}
```

#### GET /api/v1/model/capabilities

Get model capabilities and constraints.

**Response:**
```json
{
  "input_parameters": {
    "pickup_latitude": "float (-90 to 90) - Pickup location latitude"
  },
  "output_format": {
    "predicted_fare": "float - Predicted fare amount in USD"
  },
  "supported_features": [
    "Real-time fare prediction",
    "Confidence scoring"
  ],
  "constraints": {
    "coordinate_bounds": {
      "latitude": {"min": -90, "max": 90}
    }
  }
}
```

#### GET /api/v1/model/performance

Get model performance metrics.

**Response:**
```json
{
  "current_session": {
    "prediction_count": 1234,
    "success_rate_percent": 99.2,
    "avg_prediction_time_ms": 45.2
  },
  "model_benchmarks": {
    "r2_score": 0.991023,
    "mae_usd": 0.041,
    "rmse_usd": 1.054
  }
}
```

### 3. Health and Monitoring Endpoints

#### GET /health

Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00.000Z",
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "environment": "development"
}
```

#### GET /health/detailed

Detailed health check with system metrics.

**Response:**
```json
{
  "status": "healthy",
  "model_status": {
    "status": "healthy",
    "model_loaded": true,
    "prediction_working": true
  },
  "system_metrics": {
    "cpu_usage_percent": 15.2,
    "memory_usage_percent": 45.8
  },
  "api_metrics": {
    "total_predictions": 1234,
    "success_rate_percent": 99.2
  }
}
```

#### GET /metrics

Comprehensive application metrics.

**Response:**
```json
{
  "application": {
    "uptime_seconds": 3600.5,
    "version": "1.0.0"
  },
  "api_performance": {
    "total_requests": 1234,
    "success_rate_percent": 99.2
  },
  "model_metrics": {
    "predictions_served": 1234
  },
  "system_resources": {
    "cpu_usage_percent": 15.2
  }
}
```

#### GET /ping

Simple connectivity test.

**Response:**
```json
{
  "message": "pong",
  "timestamp": "2024-01-15T14:30:00.000Z",
  "service": "Taxi Fare Prediction API"
}
```

## Error Handling

### Error Response Format

```json
{
  "error": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-15T14:30:00.000Z"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters |
| 404 | Not Found - Endpoint not found |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

### Common Error Scenarios

1. **Invalid Coordinates**
   ```json
   {
     "error": "Invalid pickup latitude: 91.0 (must be -90 to 90)",
     "status_code": 400
   }
   ```

2. **Model Not Available**
   ```json
   {
     "error": "Model not initialized. Please try again later.",
     "status_code": 503
   }
   ```

3. **Validation Error**
   ```json
   {
     "error": "Passenger count must be between 1 and 8",
     "status_code": 422
   }
   ```

## Code Examples

### Python

```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/api/v1/predict', json={
    "pickup_latitude": 40.7589,
    "pickup_longitude": -73.9851,
    "dropoff_latitude": 40.7505,
    "dropoff_longitude": -73.9934,
    "passenger_count": 2,
    "pickup_datetime": "2024-01-15T14:30:00"
})

result = response.json()
print(f"Predicted fare: ${result['predicted_fare']:.2f}")
```

### JavaScript

```javascript
// Single prediction
const response = await fetch('http://localhost:8000/api/v1/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        pickup_latitude: 40.7589,
        pickup_longitude: -73.9851,
        dropoff_latitude: 40.7505,
        dropoff_longitude: -73.9934,
        passenger_count: 2,
        pickup_datetime: "2024-01-15T14:30:00"
    })
});

const result = await response.json();
console.log(`Predicted fare: $${result.predicted_fare.toFixed(2)}`);
```

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "pickup_latitude": 40.7589,
       "pickup_longitude": -73.9851,
       "dropoff_latitude": 40.7505,
       "dropoff_longitude": -73.9934,
       "passenger_count": 2,
       "pickup_datetime": "2024-01-15T14:30:00"
     }'
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive Swagger/OpenAPI documentation where you can test all endpoints directly in your browser.

## Support

For issues and questions:
- Check the health endpoint: `/health`
- Review logs for error details
- Consult the interactive documentation: `/docs`

"""
Production Taxi Fare Prediction Interface

This module provides a clean, production-ready interface for making
taxi fare predictions in web applications.
"""

import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Union, Optional
from math import radians, cos, sin, asin, sqrt
import logging

logger = logging.getLogger(__name__)

class TaxiFarePredictionModel:
    """
    Production-ready taxi fare prediction model.
    
    This class provides a complete interface for making taxi fare predictions
    with proper error handling, input validation, and confidence scoring.
    """
    
    def __init__(self, model_path: str = "best_taxi_fare_model.pkl",
                 processor_path: str = "feature_processor.pkl",
                 metadata_path: str = "final_model_metadata.json"):
        """
        Initialize the prediction model.
        
        Args:
            model_path: Path to the trained model file
            processor_path: Path to the feature processor file
            metadata_path: Path to the model metadata file
        """
        try:
            # Load model
            self.model = joblib.load(model_path)
            
            # Load feature processor (if available)
            try:
                self.feature_processor = joblib.load(processor_path)
                self.has_processor = True
            except:
                self.has_processor = False
                logger.warning("Feature processor not available, using basic processing")
            
            # Load metadata
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.required_features = self.metadata["feature_requirements"]["feature_names"]
            except:
                # Fallback feature list
                self.required_features = [
                    'pickup_latitude_scaled', 'pickup_longitude_scaled',
                    'dropoff_latitude_scaled', 'dropoff_longitude_scaled',
                    'passenger_count', 'trip_distance_scaled', 'fare_per_mile_scaled',
                    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
                    'is_morning_rush', 'is_evening_rush', 'weather_severity',
                    'traffic_severity', 'is_airport_trip', 'min_airport_dist_scaled',
                    'is_manhattan'
                ]
            
            self.model_name = "Random Forest (Default Parameters)"
            self.model_type = type(self.model).__name__
            
            logger.info(f"Loaded {self.model_name} model ({self.model_type})")
            logger.info(f"Required features: {len(self.required_features)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def predict_fare(self, pickup_latitude: float, pickup_longitude: float,
                    dropoff_latitude: float, dropoff_longitude: float,
                    passenger_count: int, pickup_datetime: str,
                    weather_condition: str = "sunny",
                    traffic_condition: str = "flow traffic") -> Dict[str, Union[float, str]]:
        """
        Predict taxi fare for a trip.
        
        Args:
            pickup_latitude: Pickup latitude (-90 to 90)
            pickup_longitude: Pickup longitude (-180 to 180)
            dropoff_latitude: Dropoff latitude (-90 to 90)
            dropoff_longitude: Dropoff longitude (-180 to 180)
            passenger_count: Number of passengers (1-8)
            pickup_datetime: Pickup datetime (ISO format or readable string)
            weather_condition: Weather condition (sunny, cloudy, windy, stormy)
            traffic_condition: Traffic condition (flow traffic, congested traffic)
            
        Returns:
            Dictionary with prediction results including fare, confidence, and metadata
        """
        try:
            # Validate inputs
            self._validate_inputs(pickup_latitude, pickup_longitude,
                                dropoff_latitude, dropoff_longitude, passenger_count)
            
            # Process features
            features_df = self._process_features(
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude,
                passenger_count, pickup_datetime,
                weather_condition, traffic_condition
            )
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            # Ensure non-negative fare
            prediction = max(0, prediction)
            
            # Calculate confidence
            confidence = self._calculate_confidence(features_df, prediction)
            
            return {
                "predicted_fare": round(prediction, 2),
                "confidence": round(confidence, 1),
                "model_name": self.model_name,
                "model_type": self.model_type,
                "prediction_timestamp": datetime.now().isoformat(),
                "status": "success",
                "input_validation": "passed"
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "predicted_fare": None,
                "confidence": 0.0,
                "model_name": self.model_name,
                "prediction_timestamp": datetime.now().isoformat(),
                "status": "error",
                "error_message": str(e),
                "input_validation": "failed"
            }
    
    def _validate_inputs(self, pickup_lat: float, pickup_lon: float,
                        dropoff_lat: float, dropoff_lon: float,
                        passenger_count: int) -> None:
        """Validate input parameters."""
        # Coordinate validation
        if not (-90 <= pickup_lat <= 90):
            raise ValueError(f"Invalid pickup latitude: {pickup_lat} (must be -90 to 90)")
        if not (-180 <= pickup_lon <= 180):
            raise ValueError(f"Invalid pickup longitude: {pickup_lon} (must be -180 to 180)")
        if not (-90 <= dropoff_lat <= 90):
            raise ValueError(f"Invalid dropoff latitude: {dropoff_lat} (must be -90 to 90)")
        if not (-180 <= dropoff_lon <= 180):
            raise ValueError(f"Invalid dropoff longitude: {dropoff_lon} (must be -180 to 180)")
        
        # Passenger count validation
        if not (1 <= passenger_count <= 8):
            raise ValueError(f"Invalid passenger count: {passenger_count} (must be 1-8)")
        
        # Distance validation (basic check)
        if (abs(pickup_lat - dropoff_lat) < 0.0001 and 
            abs(pickup_lon - dropoff_lon) < 0.0001):
            logger.warning("Pickup and dropoff locations are very close")
    
    def _process_features(self, pickup_lat: float, pickup_lon: float,
                         dropoff_lat: float, dropoff_lon: float,
                         passenger_count: int, pickup_datetime: str,
                         weather_condition: str, traffic_condition: str) -> pd.DataFrame:
        """Process raw inputs into model features with improved accuracy."""

        # Parse datetime
        if isinstance(pickup_datetime, str):
            dt = pd.to_datetime(pickup_datetime)
        else:
            dt = pickup_datetime

        # Calculate distance (Haversine formula)
        trip_distance = self._haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

        # Improved fare calculation based on NYC taxi rates
        # NYC taxi: $3.00 initial + $0.70 per 1/5 mile + $0.70 per minute in slow traffic
        # Simplified: Base fare + distance-based fare + time/traffic adjustments
        base_fare = 3.00
        distance_rate = 3.50  # per mile (0.70 * 5)

        # Calculate estimated fare per mile (more realistic)
        if trip_distance > 0:
            estimated_fare_per_mile = (base_fare / trip_distance) + distance_rate
        else:
            estimated_fare_per_mile = distance_rate

        # Improved feature scaling based on NYC taxi data statistics
        # Using more realistic means and standard deviations
        features = {
            'pickup_latitude_scaled': (pickup_lat - 40.7589) / 0.0648,  # Better NYC scaling
            'pickup_longitude_scaled': (pickup_lon + 73.9851) / 0.0781,  # Better NYC scaling
            'dropoff_latitude_scaled': (dropoff_lat - 40.7589) / 0.0648,
            'dropoff_longitude_scaled': (dropoff_lon + 73.9851) / 0.0781,
            'passenger_count': passenger_count,
            'trip_distance_scaled': (trip_distance - 2.97) / 3.52,  # Real NYC taxi stats
            'fare_per_mile_scaled': (estimated_fare_per_mile - 4.5) / 2.0,  # Realistic scaling
            'hour': dt.hour,
            'day_of_week': dt.dayofweek,
            'month': dt.month,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'is_rush_hour': 1 if (7 <= dt.hour <= 9) or (17 <= dt.hour <= 19) else 0,
            'is_morning_rush': 1 if 7 <= dt.hour <= 9 else 0,
            'is_evening_rush': 1 if 17 <= dt.hour <= 19 else 0,
            'weather_severity': self._get_weather_severity(weather_condition),
            'traffic_severity': self._get_traffic_severity(traffic_condition),
            'is_airport_trip': self._is_airport_trip(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon),
            'min_airport_dist_scaled': self._get_min_airport_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon) / 15.0,
            'is_manhattan': self._is_manhattan(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        }

        features_df = pd.DataFrame([features])

        # Ensure all required features are present
        for feature in self.required_features:
            if feature not in features_df.columns:
                features_df[feature] = 0  # Default value for missing features

        # Select only required features in correct order
        features_df = features_df[self.required_features]

        return features_df
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return c * 3956  # Earth radius in miles
    
    def _get_weather_severity(self, weather: str) -> int:
        """Convert weather condition to severity score."""
        weather_map = {'sunny': 1, 'cloudy': 2, 'windy': 3, 'stormy': 4}
        return weather_map.get(weather.lower(), 2)
    
    def _get_traffic_severity(self, traffic: str) -> int:
        """Convert traffic condition to severity score."""
        traffic_map = {'flow traffic': 1, 'congested traffic': 2}
        return traffic_map.get(traffic.lower(), 1)
    
    def _is_airport_trip(self, pickup_lat: float, pickup_lon: float, 
                        dropoff_lat: float, dropoff_lon: float) -> int:
        """Check if trip involves an airport."""
        airports = [(40.6413, -73.7781), (40.7769, -73.8740), (40.6895, -74.1745)]  # JFK, LGA, EWR
        
        for airport_lat, airport_lon in airports:
            pickup_dist = self._haversine_distance(pickup_lat, pickup_lon, airport_lat, airport_lon)
            dropoff_dist = self._haversine_distance(dropoff_lat, dropoff_lon, airport_lat, airport_lon)
            if pickup_dist < 2 or dropoff_dist < 2:  # Within 2 miles of airport
                return 1
        return 0
    
    def _get_min_airport_distance(self, pickup_lat: float, pickup_lon: float,
                                 dropoff_lat: float, dropoff_lon: float) -> float:
        """Get minimum distance to any airport."""
        airports = [(40.6413, -73.7781), (40.7769, -73.8740), (40.6895, -74.1745)]
        
        min_dist = float('inf')
        for airport_lat, airport_lon in airports:
            pickup_dist = self._haversine_distance(pickup_lat, pickup_lon, airport_lat, airport_lon)
            dropoff_dist = self._haversine_distance(dropoff_lat, dropoff_lon, airport_lat, airport_lon)
            min_dist = min(min_dist, pickup_dist, dropoff_dist)
        
        return min_dist
    
    def _is_manhattan(self, pickup_lat: float, pickup_lon: float,
                     dropoff_lat: float, dropoff_lon: float) -> int:
        """Check if trip involves Manhattan."""
        # Manhattan bounds (approximate)
        manhattan_bounds = (40.7, 40.8, -74.0, -73.9)  # lat_min, lat_max, lon_min, lon_max
        
        def in_manhattan(lat, lon):
            return (manhattan_bounds[0] <= lat <= manhattan_bounds[1] and
                   manhattan_bounds[2] <= lon <= manhattan_bounds[3])
        
        return 1 if in_manhattan(pickup_lat, pickup_lon) or in_manhattan(dropoff_lat, dropoff_lon) else 0
    
    def _calculate_confidence(self, features_df: pd.DataFrame, prediction: float) -> float:
        """Calculate prediction confidence score based on input quality and prediction reasonableness."""
        confidence = 90.0  # Base confidence for good model

        # Get actual trip distance for confidence calculation
        trip_distance_scaled = features_df['trip_distance_scaled'].iloc[0] if 'trip_distance_scaled' in features_df.columns else 0
        trip_distance = (trip_distance_scaled * 3.52) + 2.97  # Denormalize

        # Adjust based on trip distance reasonableness
        if trip_distance > 25:  # Very long trips (>25 miles)
            confidence -= 25
        elif trip_distance > 15:  # Long trips (15-25 miles)
            confidence -= 15
        elif trip_distance < 0.3:  # Very short trips (<0.3 miles)
            confidence -= 20
        elif trip_distance < 0.8:  # Short trips (0.3-0.8 miles)
            confidence -= 10

        # Adjust based on fare reasonableness relative to distance
        if trip_distance > 0:
            fare_per_mile = prediction / trip_distance
            if fare_per_mile > 15:  # Unreasonably high fare per mile
                confidence -= 30
            elif fare_per_mile > 10:  # High fare per mile
                confidence -= 20
            elif fare_per_mile < 2:  # Unreasonably low fare per mile
                confidence -= 25
            elif fare_per_mile < 3:  # Low fare per mile
                confidence -= 15

        # Adjust based on absolute fare amount
        if prediction > 150:  # Very high fares
            confidence -= 25
        elif prediction > 80:  # High fares
            confidence -= 15
        elif prediction < 3.0:  # Very low fares (below minimum)
            confidence -= 20

        # Adjust based on passenger count
        if 'passenger_count' in features_df.columns:
            passengers = features_df['passenger_count'].iloc[0]
            if passengers > 6:  # Unusual passenger count
                confidence -= 15
            elif passengers == 0:  # Invalid passenger count
                confidence -= 30

        # Adjust based on time of day
        if 'hour' in features_df.columns:
            hour = features_df['hour'].iloc[0]
            if hour < 4 or hour > 23:  # Very late/early hours
                confidence -= 10

        # Adjust based on location reasonableness (NYC area check)
        pickup_lat = (features_df['pickup_latitude_scaled'].iloc[0] * 0.0648) + 40.7589
        pickup_lon = (features_df['pickup_longitude_scaled'].iloc[0] * 0.0781) - 73.9851

        # Check if coordinates are in reasonable NYC area
        if not (40.4 <= pickup_lat <= 41.0 and -74.5 <= pickup_lon <= -73.5):
            confidence -= 20

        return max(45.0, min(95.0, confidence))
    
    def get_model_info(self) -> Dict[str, Union[str, int, list]]:
        """Get comprehensive model information."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "required_features": len(self.required_features),
            "feature_names": self.required_features,
            "has_feature_processor": self.has_processor,
            "version": "1.0.0",
            "deployment_ready": True,
            "performance_metrics": self.metadata.get("performance_metrics", {}) if hasattr(self, 'metadata') else {}
        }
    
    def health_check(self) -> Dict[str, Union[str, bool]]:
        """Perform model health check."""
        try:
            # Test prediction with sample data
            test_result = self.predict_fare(
                pickup_latitude=40.7589,
                pickup_longitude=-73.9851,
                dropoff_latitude=40.7505,
                dropoff_longitude=-73.9934,
                passenger_count=2,
                pickup_datetime="2024-01-15 14:30:00"
            )
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "prediction_working": test_result["status"] == "success",
                "test_prediction": test_result.get("predicted_fare"),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "prediction_working": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }


# Convenience functions for quick usage
def predict_taxi_fare(pickup_lat: float, pickup_lon: float,
                     dropoff_lat: float, dropoff_lon: float,
                     passenger_count: int, pickup_datetime: str,
                     weather: str = "sunny", traffic: str = "flow traffic") -> float:
    """
    Quick function to predict taxi fare.
    
    Returns:
        Predicted fare amount (float)
    """
    model = TaxiFarePredictionModel()
    result = model.predict_fare(
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
        passenger_count, pickup_datetime, weather, traffic
    )
    
    if result["status"] == "success":
        return result["predicted_fare"]
    else:
        raise Exception(f"Prediction failed: {result.get('error_message', 'Unknown error')}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = TaxiFarePredictionModel()
    
    # Health check
    health = model.health_check()
    print(f"Model health: {health}")
    
    # Example prediction
    result = model.predict_fare(
        pickup_latitude=40.7589,
        pickup_longitude=-73.9851,
        dropoff_latitude=40.7505,
        dropoff_longitude=-73.9934,
        passenger_count=2,
        pickup_datetime="2024-01-15 14:30:00",
        weather_condition="sunny",
        traffic_condition="flow traffic"
    )
    
    print(f"\nExample prediction: {result}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel info: {info}")

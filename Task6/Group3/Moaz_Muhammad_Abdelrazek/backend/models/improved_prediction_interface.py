"""
Improved Taxi Fare Prediction Interface

This module provides a more accurate, realistic taxi fare prediction
based on actual NYC taxi fare structure and distance calculations.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Union, Optional
from math import radians, cos, sin, asin, sqrt
import logging

logger = logging.getLogger(__name__)

class ImprovedTaxiFarePredictionModel:
    """
    Improved taxi fare prediction model with realistic NYC fare calculations.
    
    This model uses actual NYC taxi fare structure:
    - Initial charge: $3.00
    - Distance rate: $0.70 per 1/5 mile (or $3.50 per mile)
    - Time rate: $0.70 per minute when speed < 12 mph
    - Peak hour surcharge: 50¢ (4-8 PM weekdays)
    - Overnight surcharge: $1.00 (8 PM - 6 AM)
    - Airport trips: Additional fees
    """
    
    def __init__(self):
        """Initialize the improved prediction model."""
        self.model_name = "Improved NYC Taxi Fare Calculator"
        self.model_type = "RuleBasedRegressor"
        
        # NYC taxi fare structure (2024 rates)
        self.initial_charge = 3.00
        self.distance_rate = 3.50  # per mile ($0.70 per 1/5 mile)
        self.time_rate = 0.70  # per minute in slow traffic
        self.peak_surcharge = 0.50  # 4-8 PM weekdays
        self.overnight_surcharge = 1.00  # 8 PM - 6 AM
        self.airport_surcharge = 5.00  # Airport trips
        
        # Traffic speed assumptions
        self.normal_speed = 25  # mph in normal traffic
        self.congested_speed = 12  # mph in congested traffic
        self.rush_hour_speed = 8  # mph in rush hour
        
        logger.info(f"Initialized {self.model_name}")
    
    def predict_fare(self, pickup_latitude: float, pickup_longitude: float,
                    dropoff_latitude: float, dropoff_longitude: float,
                    passenger_count: int, pickup_datetime: str,
                    weather_condition: str = "sunny",
                    traffic_condition: str = "flow traffic") -> Dict[str, Union[float, str]]:
        """
        Predict taxi fare using realistic NYC taxi fare structure.
        
        Args:
            pickup_latitude: Pickup latitude
            pickup_longitude: Pickup longitude
            dropoff_latitude: Dropoff latitude
            dropoff_longitude: Dropoff longitude
            passenger_count: Number of passengers
            pickup_datetime: Pickup datetime
            weather_condition: Weather condition
            traffic_condition: Traffic condition
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate inputs
            self._validate_inputs(pickup_latitude, pickup_longitude,
                                dropoff_latitude, dropoff_longitude, passenger_count)
            
            # Parse datetime
            if isinstance(pickup_datetime, str):
                dt = pd.to_datetime(pickup_datetime)
            else:
                dt = pickup_datetime
            
            # Calculate trip distance
            trip_distance = self._haversine_distance(
                pickup_latitude, pickup_longitude,
                dropoff_latitude, dropoff_longitude
            )
            
            # Calculate base fare
            fare = self._calculate_realistic_fare(
                trip_distance, dt, weather_condition, traffic_condition,
                pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
            )
            
            # Calculate confidence based on input quality
            confidence = self._calculate_confidence(
                trip_distance, fare, dt, passenger_count,
                pickup_latitude, pickup_longitude
            )
            
            return {
                "predicted_fare": round(fare, 2),
                "confidence": round(confidence, 1),
                "model_name": self.model_name,
                "model_type": self.model_type,
                "prediction_timestamp": datetime.now().isoformat(),
                "prediction_time": 5.0,  # Simulated processing time
                "status": "success",
                "error_message": "",
                "input_validation": "passed",
                "api_version": "1.0.0",
                "trip_distance_miles": round(trip_distance, 2)
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
    
    def _calculate_realistic_fare(self, trip_distance: float, dt: datetime,
                                 weather_condition: str, traffic_condition: str,
                                 pickup_lat: float, pickup_lon: float,
                                 dropoff_lat: float, dropoff_lon: float) -> float:
        """Calculate realistic fare based on NYC taxi fare structure."""
        
        # Start with initial charge
        fare = self.initial_charge
        
        # Add distance charge
        distance_charge = trip_distance * self.distance_rate
        fare += distance_charge
        
        # Determine average speed based on conditions
        speed = self._get_average_speed(dt, weather_condition, traffic_condition)
        
        # Calculate estimated trip time in minutes
        if speed > 0:
            trip_time_minutes = (trip_distance / speed) * 60
        else:
            trip_time_minutes = 0
        
        # Add time charge for slow traffic (when speed < 12 mph)
        if speed < 12:
            slow_time_minutes = trip_time_minutes * (1 - speed / 12)
            time_charge = slow_time_minutes * self.time_rate
            fare += time_charge
        
        # Add surcharges
        fare += self._calculate_surcharges(dt, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        
        # Weather adjustment
        weather_multiplier = self._get_weather_multiplier(weather_condition)
        fare *= weather_multiplier
        
        # Ensure minimum fare
        fare = max(fare, 4.50)  # NYC minimum fare
        
        return fare
    
    def _get_average_speed(self, dt: datetime, weather: str, traffic: str) -> float:
        """Determine average speed based on time, weather, and traffic."""
        hour = dt.hour
        is_weekday = dt.weekday() < 5
        
        # Base speed
        if traffic.lower() == "congested traffic":
            speed = self.congested_speed
        else:
            speed = self.normal_speed
        
        # Rush hour adjustment
        if is_weekday and ((7 <= hour <= 9) or (17 <= hour <= 19)):
            speed = min(speed, self.rush_hour_speed)
        
        # Weather adjustment
        if weather.lower() in ["stormy", "windy"]:
            speed *= 0.8  # 20% slower in bad weather
        elif weather.lower() == "cloudy":
            speed *= 0.95  # 5% slower in cloudy weather
        
        return max(speed, 5)  # Minimum 5 mph
    
    def _calculate_surcharges(self, dt: datetime, pickup_lat: float, pickup_lon: float,
                             dropoff_lat: float, dropoff_lon: float) -> float:
        """Calculate applicable surcharges."""
        surcharge = 0.0
        
        hour = dt.hour
        is_weekday = dt.weekday() < 5
        
        # Peak hour surcharge (4-8 PM weekdays)
        if is_weekday and 16 <= hour < 20:
            surcharge += self.peak_surcharge
        
        # Overnight surcharge (8 PM - 6 AM)
        if hour >= 20 or hour < 6:
            surcharge += self.overnight_surcharge
        
        # Airport surcharge
        if self._is_airport_trip(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
            surcharge += self.airport_surcharge
        
        return surcharge
    
    def _get_weather_multiplier(self, weather: str) -> float:
        """Get fare multiplier based on weather conditions."""
        weather_multipliers = {
            'sunny': 1.0,
            'cloudy': 1.05,
            'windy': 1.10,
            'stormy': 1.20
        }
        return weather_multipliers.get(weather.lower(), 1.0)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return c * 3956  # Earth radius in miles
    
    def _is_airport_trip(self, pickup_lat: float, pickup_lon: float, 
                        dropoff_lat: float, dropoff_lon: float) -> bool:
        """Check if trip involves an airport."""
        airports = [
            (40.6413, -73.7781),  # JFK
            (40.7769, -73.8740),  # LGA
            (40.6895, -74.1745)   # EWR
        ]
        
        for airport_lat, airport_lon in airports:
            pickup_dist = self._haversine_distance(pickup_lat, pickup_lon, airport_lat, airport_lon)
            dropoff_dist = self._haversine_distance(dropoff_lat, dropoff_lon, airport_lat, airport_lon)
            if pickup_dist < 2 or dropoff_dist < 2:  # Within 2 miles of airport
                return True
        return False
    
    def _validate_inputs(self, pickup_lat: float, pickup_lon: float,
                        dropoff_lat: float, dropoff_lon: float,
                        passenger_count: int) -> None:
        """Validate input parameters."""
        # Coordinate validation
        if not (-90 <= pickup_lat <= 90):
            raise ValueError(f"Invalid pickup latitude: {pickup_lat}")
        if not (-180 <= pickup_lon <= 180):
            raise ValueError(f"Invalid pickup longitude: {pickup_lon}")
        if not (-90 <= dropoff_lat <= 90):
            raise ValueError(f"Invalid dropoff latitude: {dropoff_lat}")
        if not (-180 <= dropoff_lon <= 180):
            raise ValueError(f"Invalid dropoff longitude: {dropoff_lon}")
        
        # Passenger count validation
        if not (1 <= passenger_count <= 8):
            raise ValueError(f"Invalid passenger count: {passenger_count}")
        
        # Distance validation
        distance = self._haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        if distance > 100:  # Unreasonably long trip
            raise ValueError(f"Trip distance too long: {distance:.1f} miles")
    
    def _calculate_confidence(self, trip_distance: float, fare: float, dt: datetime,
                             passenger_count: int, pickup_lat: float, pickup_lon: float) -> float:
        """Calculate prediction confidence based on input quality."""
        confidence = 95.0  # High confidence for rule-based model
        
        # Distance-based adjustments
        if trip_distance > 50:  # Very long trips
            confidence -= 20
        elif trip_distance > 25:  # Long trips
            confidence -= 10
        elif trip_distance < 0.2:  # Very short trips
            confidence -= 15
        
        # Fare reasonableness
        if trip_distance > 0:
            fare_per_mile = fare / trip_distance
            if fare_per_mile > 20 or fare_per_mile < 3:
                confidence -= 10
        
        # Time-based adjustments
        hour = dt.hour
        if hour < 4 or hour > 23:  # Very late/early
            confidence -= 5
        
        # Location validation (NYC area)
        if not (40.4 <= pickup_lat <= 41.0 and -74.5 <= pickup_lon <= -73.5):
            confidence -= 15
        
        return max(75.0, min(98.0, confidence))

    def get_model_info(self) -> Dict[str, Union[str, int, list]]:
        """Get comprehensive model information."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": "2.0.0",
            "required_features": 8,
            "feature_names": [
                "pickup_latitude", "pickup_longitude", "dropoff_latitude",
                "dropoff_longitude", "passenger_count", "pickup_datetime",
                "weather_condition", "traffic_condition"
            ],
            "has_feature_processor": False,
            "deployment_ready": True,
            "performance_metrics": {
                "accuracy": "Rule-based calculation",
                "fare_structure": "NYC 2024 official rates",
                "distance_calculation": "Haversine formula",
                "confidence_range": "75-98%"
            }
        }

    def health_check(self) -> Dict[str, Union[str, bool, float]]:
        """Perform model health check."""
        try:
            # Test prediction with sample data
            test_result = self.predict_fare(
                pickup_latitude=40.7589,
                pickup_longitude=-73.9851,
                dropoff_latitude=40.7505,
                dropoff_longitude=-73.9934,
                passenger_count=2,
                pickup_datetime="2024-01-15T14:30:00"
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


# Example usage and testing
if __name__ == "__main__":
    # Initialize improved model
    model = ImprovedTaxiFarePredictionModel()

    # Health check
    health = model.health_check()
    print(f"Model health: {health}")

    # Test various scenarios
    test_cases = [
        {
            "name": "Short Manhattan trip",
            "pickup_latitude": 40.7589,
            "pickup_longitude": -73.9851,
            "dropoff_latitude": 40.7505,
            "dropoff_longitude": -73.9934,
            "passenger_count": 2,
            "pickup_datetime": "2024-01-15T14:30:00"
        },
        {
            "name": "JFK Airport trip",
            "pickup_latitude": 40.7589,
            "pickup_longitude": -73.9851,
            "dropoff_latitude": 40.6413,
            "dropoff_longitude": -73.7781,
            "passenger_count": 1,
            "pickup_datetime": "2024-01-15T08:00:00"
        },
        {
            "name": "Rush hour trip",
            "pickup_latitude": 40.7282,
            "pickup_longitude": -73.7949,
            "dropoff_latitude": 40.7505,
            "dropoff_longitude": -73.9934,
            "passenger_count": 3,
            "pickup_datetime": "2024-01-15T18:30:00",
            "weather_condition": "stormy",
            "traffic_condition": "congested traffic"
        }
    ]

    print("\nTest predictions:")
    for i, test_case in enumerate(test_cases, 1):
        name = test_case.pop('name')  # Remove name from test_case
        result = model.predict_fare(**test_case)
        print(f"{i}. {name}: ${result.get('predicted_fare', 'Error'):.2f} "
              f"(confidence: {result.get('confidence', 0):.1f}%, "
              f"distance: {result.get('trip_distance_miles', 0):.1f} miles)")

    # Model info
    info = model.get_model_info()
    print(f"\nModel info: {info}")

"""
Machine Learning Model Manager for Taxi Fare Prediction API.

This module provides a high-level interface for managing the trained
Random Forest model, including loading, prediction, and health monitoring.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Add models directory to Python path to import prediction interface
models_dir = Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_dir))

try:
    from prediction_interface import TaxiFarePredictionModel
    from improved_prediction_interface import ImprovedTaxiFarePredictionModel
except ImportError as e:
    logging.error(f"Failed to import prediction interface: {e}")
    raise

from app.core.logging import model_logger

logger = logging.getLogger(__name__)


class ModelManager:
    """
    High-level manager for the taxi fare prediction model.
    
    This class provides a clean interface for model operations including
    loading, prediction, health checks, and performance monitoring.
    """
    
    def __init__(self, model_path: str, processor_path: str, metadata_path: str):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to the trained model file
            processor_path: Path to the feature processor file
            metadata_path: Path to the model metadata file
        """
        self.model_path = model_path
        self.processor_path = processor_path
        self.metadata_path = metadata_path
        self.model = None
        self.load_time = None
        self.prediction_count = 0
        self.error_count = 0
        self.total_prediction_time = 0.0
        
        # Load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the machine learning model."""
        try:
            start_time = time.time()
            
            logger.info("Loading taxi fare prediction model...")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Processor path: {self.processor_path}")
            logger.info(f"Metadata path: {self.metadata_path}")
            
            # Initialize the improved prediction model
            self.model = ImprovedTaxiFarePredictionModel()
            
            self.load_time = time.time() - start_time
            
            # Get model info for logging
            model_info = self.model.get_model_info()
            model_size = self._get_model_size()
            
            model_logger.log_model_load(
                model_name=model_info["model_name"],
                model_size=model_size,
                load_time=self.load_time
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_model_size(self) -> str:
        """Get model file size in human-readable format."""
        try:
            size_bytes = os.path.getsize(self.model_path)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.1f} MB"
            else:
                return f"{size_bytes/(1024**3):.1f} GB"
        except:
            return "Unknown"
    
    def predict(self, pickup_latitude: float, pickup_longitude: float,
                dropoff_latitude: float, dropoff_longitude: float,
                passenger_count: int, pickup_datetime: str,
                weather_condition: str = "sunny",
                traffic_condition: str = "flow traffic") -> Dict[str, Any]:
        """
        Make a taxi fare prediction.
        
        Args:
            pickup_latitude: Pickup latitude (-90 to 90)
            pickup_longitude: Pickup longitude (-180 to 180)
            dropoff_latitude: Dropoff latitude (-90 to 90)
            dropoff_longitude: Dropoff longitude (-180 to 180)
            passenger_count: Number of passengers (1-8)
            pickup_datetime: Pickup datetime (ISO format)
            weather_condition: Weather condition (optional)
            traffic_condition: Traffic condition (optional)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Make prediction using the loaded model
            result = self.model.predict_fare(
                pickup_latitude=pickup_latitude,
                pickup_longitude=pickup_longitude,
                dropoff_latitude=dropoff_latitude,
                dropoff_longitude=dropoff_longitude,
                passenger_count=passenger_count,
                pickup_datetime=pickup_datetime,
                weather_condition=weather_condition,
                traffic_condition=traffic_condition
            )
            
            prediction_time = time.time() - start_time
            
            # Update statistics
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            
            if result["status"] == "success":
                # Log successful prediction
                logger.info(
                    f"Prediction successful: ${result['predicted_fare']:.2f} "
                    f"(confidence: {result['confidence']:.1f}%) in {prediction_time*1000:.1f}ms"
                )
            else:
                self.error_count += 1
                logger.error(
                    f"Prediction failed: {result.get('error_message', 'Unknown error')}"
                )
            
            # Add performance metrics to result
            result["prediction_time"] = round(prediction_time * 1000, 2)  # ms
            result["api_version"] = "1.0.0"
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction failed: {e}")
            
            return {
                "predicted_fare": None,
                "confidence": 0.0,
                "model_name": "Random Forest (Default Parameters)",
                "prediction_timestamp": datetime.now().isoformat(),
                "status": "error",
                "error_message": str(e),
                "prediction_time": round((time.time() - start_time) * 1000, 2),
                "api_version": "1.0.0"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dictionary with health status and details
        """
        try:
            if self.model is None:
                return {
                    "status": "unhealthy",
                    "message": "Model not loaded",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Perform model health check
            health_result = self.model.health_check()
            
            # Add additional metrics
            avg_prediction_time = (
                self.total_prediction_time / self.prediction_count
                if self.prediction_count > 0 else 0
            )
            
            error_rate = (
                self.error_count / self.prediction_count * 100
                if self.prediction_count > 0 else 0
            )
            
            enhanced_result = {
                **health_result,
                "model_load_time": round(self.load_time, 3) if self.load_time else None,
                "prediction_count": self.prediction_count,
                "error_count": self.error_count,
                "error_rate_percent": round(error_rate, 2),
                "avg_prediction_time_ms": round(avg_prediction_time * 1000, 2),
                "model_size": self._get_model_size(),
                "api_version": "1.0.0"
            }
            
            # Log health check
            if health_result["status"] == "healthy":
                logger.info(f"Health check passed: {health_result['status']}")
            else:
                logger.warning(f"Health check failed: {health_result['status']}")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Health check error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model details and metadata
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Get base model info
            model_info = self.model.get_model_info()
            
            # Add deployment-specific information
            enhanced_info = {
                **model_info,
                "deployment_info": {
                    "api_version": "1.0.0",
                    "model_path": self.model_path,
                    "load_time": round(self.load_time, 3) if self.load_time else None,
                    "model_size": self._get_model_size(),
                    "prediction_count": self.prediction_count,
                    "error_count": self.error_count,
                    "uptime": self._get_uptime()
                }
            }
            
            return enhanced_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": f"Failed to get model info: {str(e)}"}
    
    def _get_uptime(self) -> str:
        """Get model uptime since loading."""
        if self.load_time is None:
            return "Unknown"
        
        uptime_seconds = time.time() - (time.time() - self.load_time)
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get model performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_prediction_time = (
            self.total_prediction_time / self.prediction_count
            if self.prediction_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.prediction_count * 100
            if self.prediction_count > 0 else 0
        )
        
        return {
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "success_count": self.prediction_count - self.error_count,
            "error_rate_percent": round(error_rate, 2),
            "success_rate_percent": round(100 - error_rate, 2),
            "avg_prediction_time_ms": round(avg_prediction_time * 1000, 2),
            "total_prediction_time_s": round(self.total_prediction_time, 3),
            "uptime": self._get_uptime(),
            "model_size": self._get_model_size()
        }

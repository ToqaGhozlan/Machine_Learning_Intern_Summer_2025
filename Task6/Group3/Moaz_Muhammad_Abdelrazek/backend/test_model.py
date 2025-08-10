#!/usr/bin/env python3
"""
Test script for the Taxi Fare Prediction model integration.

This script tests the model loading and prediction functionality
to ensure everything is working correctly before deployment.
"""

import sys
import os
import time
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

try:
    from app.models.ml_model import ModelManager
    from app.core.config import get_settings
    from app.core.logging import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the backend directory")
    sys.exit(1)

def test_model_loading():
    """Test model loading functionality."""
    print("🔄 Testing model loading...")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Initialize model manager
        model_manager = ModelManager(
            model_path=settings.MODEL_PATH,
            processor_path=settings.PROCESSOR_PATH,
            metadata_path=settings.METADATA_PATH
        )
        
        print("✅ Model loaded successfully!")
        return model_manager
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_health_check(model_manager):
    """Test model health check."""
    print("\n🔄 Testing health check...")
    
    try:
        health_status = model_manager.health_check()
        
        if health_status["status"] == "healthy":
            print("✅ Health check passed!")
            print(f"   - Model loaded: {health_status.get('model_loaded', 'Unknown')}")
            print(f"   - Prediction working: {health_status.get('prediction_working', 'Unknown')}")
            print(f"   - Test prediction: ${health_status.get('test_prediction', 'N/A')}")
        else:
            print(f"⚠️  Health check status: {health_status['status']}")
            print(f"   - Message: {health_status.get('message', 'No message')}")
        
        return health_status["status"] == "healthy"
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_prediction(model_manager):
    """Test prediction functionality."""
    print("\n🔄 Testing prediction functionality...")
    
    # Test cases
    test_cases = [
        {
            "name": "NYC Times Square to JFK Airport",
            "pickup_latitude": 40.7589,
            "pickup_longitude": -73.9851,
            "dropoff_latitude": 40.6413,
            "dropoff_longitude": -73.7781,
            "passenger_count": 2,
            "pickup_datetime": "2024-01-15T14:30:00",
            "weather_condition": "sunny",
            "traffic_condition": "flow traffic"
        },
        {
            "name": "Short Manhattan trip",
            "pickup_latitude": 40.7505,
            "pickup_longitude": -73.9934,
            "dropoff_latitude": 40.7614,
            "dropoff_longitude": -73.9776,
            "passenger_count": 1,
            "pickup_datetime": "2024-01-15T08:00:00",
            "weather_condition": "cloudy",
            "traffic_condition": "congested traffic"
        },
        {
            "name": "Evening rush hour trip",
            "pickup_latitude": 40.7282,
            "pickup_longitude": -73.7949,
            "dropoff_latitude": 40.7505,
            "dropoff_longitude": -73.9934,
            "passenger_count": 4,
            "pickup_datetime": "2024-01-15T18:30:00",
            "weather_condition": "stormy",
            "traffic_condition": "congested traffic"
        }
    ]
    
    successful_predictions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['name']}")
        
        try:
            start_time = time.time()
            
            result = model_manager.predict(
                pickup_latitude=test_case["pickup_latitude"],
                pickup_longitude=test_case["pickup_longitude"],
                dropoff_latitude=test_case["dropoff_latitude"],
                dropoff_longitude=test_case["dropoff_longitude"],
                passenger_count=test_case["passenger_count"],
                pickup_datetime=test_case["pickup_datetime"],
                weather_condition=test_case["weather_condition"],
                traffic_condition=test_case["traffic_condition"]
            )
            
            prediction_time = time.time() - start_time
            
            if result["status"] == "success":
                print(f"   ✅ Prediction: ${result['predicted_fare']:.2f}")
                print(f"   📊 Confidence: {result['confidence']:.1f}%")
                print(f"   ⚡ Time: {prediction_time*1000:.1f}ms")
                successful_predictions += 1
            else:
                print(f"   ❌ Prediction failed: {result.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Exception during prediction: {e}")
    
    print(f"\n📊 Prediction test results: {successful_predictions}/{len(test_cases)} successful")
    return successful_predictions == len(test_cases)

def test_model_info(model_manager):
    """Test model information retrieval."""
    print("\n🔄 Testing model information...")
    
    try:
        model_info = model_manager.get_model_info()
        
        if "error" not in model_info:
            print("✅ Model info retrieved successfully!")
            print(f"   - Model: {model_info.get('model_name', 'Unknown')}")
            print(f"   - Type: {model_info.get('model_type', 'Unknown')}")
            print(f"   - Features: {model_info.get('required_features', 'Unknown')}")
            print(f"   - Version: {model_info.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ Model info error: {model_info['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_statistics(model_manager):
    """Test statistics retrieval."""
    print("\n🔄 Testing statistics...")
    
    try:
        stats = model_manager.get_statistics()
        
        print("✅ Statistics retrieved successfully!")
        print(f"   - Predictions: {stats.get('prediction_count', 0)}")
        print(f"   - Success rate: {stats.get('success_rate_percent', 0):.1f}%")
        print(f"   - Avg time: {stats.get('avg_prediction_time_ms', 0):.1f}ms")
        print(f"   - Model size: {stats.get('model_size', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚖 TAXI FARE PREDICTION MODEL TEST")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Test model loading
    model_manager = test_model_loading()
    if not model_manager:
        print("\n❌ Model loading failed. Cannot continue with tests.")
        return False
    
    # Run all tests
    tests = [
        ("Health Check", lambda: test_health_check(model_manager)),
        ("Model Info", lambda: test_model_info(model_manager)),
        ("Predictions", lambda: test_prediction(model_manager)),
        ("Statistics", lambda: test_statistics(model_manager))
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Model is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

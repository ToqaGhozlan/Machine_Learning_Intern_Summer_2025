#!/usr/bin/env python3
"""
Comprehensive deployment test script for Taxi Fare Prediction application.

This script tests the complete deployment including backend API, model functionality,
and validates that everything is working correctly for production use.
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
TIMEOUT = 30

class DeploymentTester:
    """Comprehensive deployment testing class."""
    
    def __init__(self, backend_url: str = BACKEND_URL, frontend_url: str = FRONTEND_URL):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.session = requests.Session()
        self.session.timeout = TIMEOUT
        
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, message: str = "", details: Any = None):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_name}")
        if message:
            print(f"     {message}")
        if details and not passed:
            print(f"     Details: {details}")
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
    
    def test_backend_connectivity(self) -> bool:
        """Test basic backend connectivity."""
        try:
            response = self.session.get(f"{self.backend_url}/ping")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Backend Connectivity", True, f"Response: {data.get('message', 'N/A')}")
                return True
            else:
                self.log_test("Backend Connectivity", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Backend Connectivity", False, f"Connection failed: {str(e)}")
            return False
    
    def test_health_endpoints(self) -> bool:
        """Test health check endpoints."""
        endpoints = [
            ("/health", "Basic Health Check"),
            ("/health/detailed", "Detailed Health Check"),
            ("/metrics", "Metrics Endpoint")
        ]
        
        all_passed = True
        
        for endpoint, name in endpoints:
            try:
                response = self.session.get(f"{self.backend_url}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    self.log_test(name, True, f"Status: {status}")
                else:
                    self.log_test(name, False, f"HTTP {response.status_code}")
                    all_passed = False
            except Exception as e:
                self.log_test(name, False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_model_endpoints(self) -> bool:
        """Test model information endpoints."""
        endpoints = [
            ("/api/v1/model/info", "Model Info"),
            ("/api/v1/model/capabilities", "Model Capabilities"),
            ("/api/v1/model/performance", "Model Performance")
        ]
        
        all_passed = True
        
        for endpoint, name in endpoints:
            try:
                response = self.session.get(f"{self.backend_url}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    if name == "Model Info":
                        model_name = data.get('model_name', 'Unknown')
                        self.log_test(name, True, f"Model: {model_name}")
                    else:
                        self.log_test(name, True, "Data retrieved successfully")
                else:
                    self.log_test(name, False, f"HTTP {response.status_code}")
                    all_passed = False
            except Exception as e:
                self.log_test(name, False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_prediction_functionality(self) -> bool:
        """Test prediction endpoints with various scenarios."""
        test_cases = [
            {
                "name": "NYC Times Square to JFK",
                "data": {
                    "pickup_latitude": 40.7589,
                    "pickup_longitude": -73.9851,
                    "dropoff_latitude": 40.6413,
                    "dropoff_longitude": -73.7781,
                    "passenger_count": 2,
                    "pickup_datetime": "2024-01-15T14:30:00",
                    "weather_condition": "sunny",
                    "traffic_condition": "flow traffic"
                },
                "expected_range": (20, 80)  # Expected fare range
            },
            {
                "name": "Short Manhattan Trip",
                "data": {
                    "pickup_latitude": 40.7505,
                    "pickup_longitude": -73.9934,
                    "dropoff_latitude": 40.7614,
                    "dropoff_longitude": -73.9776,
                    "passenger_count": 1,
                    "pickup_datetime": "2024-01-15T08:00:00",
                    "weather_condition": "cloudy",
                    "traffic_condition": "congested traffic"
                },
                "expected_range": (5, 25)
            },
            {
                "name": "Evening Rush Hour",
                "data": {
                    "pickup_latitude": 40.7282,
                    "pickup_longitude": -73.7949,
                    "dropoff_latitude": 40.7505,
                    "dropoff_longitude": -73.9934,
                    "passenger_count": 4,
                    "pickup_datetime": "2024-01-15T18:30:00",
                    "weather_condition": "stormy",
                    "traffic_condition": "congested traffic"
                },
                "expected_range": (15, 50)
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.backend_url}/api/v1/predict",
                    json=test_case["data"]
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        fare = data.get('predicted_fare')
                        confidence = data.get('confidence')
                        
                        # Validate fare range
                        min_fare, max_fare = test_case["expected_range"]
                        fare_valid = min_fare <= fare <= max_fare
                        
                        # Validate confidence
                        confidence_valid = 50 <= confidence <= 100
                        
                        # Validate response time
                        time_valid = response_time < 5.0  # Should be under 5 seconds
                        
                        if fare_valid and confidence_valid and time_valid:
                            self.log_test(
                                f"Prediction: {test_case['name']}", 
                                True, 
                                f"Fare: ${fare:.2f}, Confidence: {confidence:.1f}%, Time: {response_time*1000:.1f}ms"
                            )
                        else:
                            issues = []
                            if not fare_valid:
                                issues.append(f"fare ${fare:.2f} outside expected range ${min_fare}-${max_fare}")
                            if not confidence_valid:
                                issues.append(f"confidence {confidence:.1f}% outside valid range")
                            if not time_valid:
                                issues.append(f"response time {response_time:.2f}s too slow")
                            
                            self.log_test(
                                f"Prediction: {test_case['name']}", 
                                False, 
                                f"Validation failed: {', '.join(issues)}"
                            )
                            all_passed = False
                    else:
                        self.log_test(
                            f"Prediction: {test_case['name']}", 
                            False, 
                            f"Prediction failed: {data.get('error_message', 'Unknown error')}"
                        )
                        all_passed = False
                else:
                    self.log_test(
                        f"Prediction: {test_case['name']}", 
                        False, 
                        f"HTTP {response.status_code}"
                    )
                    all_passed = False
                    
            except Exception as e:
                self.log_test(
                    f"Prediction: {test_case['name']}", 
                    False, 
                    f"Error: {str(e)}"
                )
                all_passed = False
        
        return all_passed
    
    def test_input_validation(self) -> bool:
        """Test input validation with invalid data."""
        invalid_cases = [
            {
                "name": "Invalid Coordinates",
                "data": {
                    "pickup_latitude": 91.0,  # Invalid latitude
                    "pickup_longitude": -73.9851,
                    "dropoff_latitude": 40.7505,
                    "dropoff_longitude": -73.9934,
                    "passenger_count": 2,
                    "pickup_datetime": "2024-01-15T14:30:00"
                },
                "expected_status": 400
            },
            {
                "name": "Invalid Passenger Count",
                "data": {
                    "pickup_latitude": 40.7589,
                    "pickup_longitude": -73.9851,
                    "dropoff_latitude": 40.7505,
                    "dropoff_longitude": -73.9934,
                    "passenger_count": 10,  # Invalid passenger count
                    "pickup_datetime": "2024-01-15T14:30:00"
                },
                "expected_status": 400
            },
            {
                "name": "Missing Required Field",
                "data": {
                    "pickup_latitude": 40.7589,
                    "pickup_longitude": -73.9851,
                    "dropoff_latitude": 40.7505,
                    # Missing dropoff_longitude
                    "passenger_count": 2,
                    "pickup_datetime": "2024-01-15T14:30:00"
                },
                "expected_status": 422
            }
        ]
        
        all_passed = True
        
        for test_case in invalid_cases:
            try:
                response = self.session.post(
                    f"{self.backend_url}/api/v1/predict",
                    json=test_case["data"]
                )
                
                if response.status_code == test_case["expected_status"]:
                    self.log_test(
                        f"Validation: {test_case['name']}", 
                        True, 
                        f"Correctly rejected with HTTP {response.status_code}"
                    )
                else:
                    self.log_test(
                        f"Validation: {test_case['name']}", 
                        False, 
                        f"Expected HTTP {test_case['expected_status']}, got {response.status_code}"
                    )
                    all_passed = False
                    
            except Exception as e:
                self.log_test(
                    f"Validation: {test_case['name']}", 
                    False, 
                    f"Error: {str(e)}"
                )
                all_passed = False
        
        return all_passed
    
    def test_frontend_accessibility(self) -> bool:
        """Test frontend accessibility."""
        try:
            response = self.session.get(self.frontend_url)
            if response.status_code == 200:
                # Check if it's HTML content
                if 'text/html' in response.headers.get('content-type', ''):
                    self.log_test("Frontend Accessibility", True, "Frontend is accessible")
                    return True
                else:
                    self.log_test("Frontend Accessibility", False, "Response is not HTML")
                    return False
            else:
                self.log_test("Frontend Accessibility", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Frontend Accessibility", False, f"Connection failed: {str(e)}")
            return False
    
    def test_api_documentation(self) -> bool:
        """Test API documentation accessibility."""
        try:
            response = self.session.get(f"{self.backend_url}/docs")
            if response.status_code == 200:
                self.log_test("API Documentation", True, "Swagger docs accessible")
                return True
            else:
                self.log_test("API Documentation", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Documentation", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all deployment tests."""
        print("🚖 TAXI FARE PREDICTION - DEPLOYMENT TEST")
        print("=" * 60)
        print(f"Backend URL: {self.backend_url}")
        print(f"Frontend URL: {self.frontend_url}")
        print(f"Timeout: {TIMEOUT}s")
        print("=" * 60)
        
        # Run test suites
        test_suites = [
            ("Backend Connectivity", self.test_backend_connectivity),
            ("Health Endpoints", self.test_health_endpoints),
            ("Model Endpoints", self.test_model_endpoints),
            ("Prediction Functionality", self.test_prediction_functionality),
            ("Input Validation", self.test_input_validation),
            ("Frontend Accessibility", self.test_frontend_accessibility),
            ("API Documentation", self.test_api_documentation)
        ]
        
        suite_results = []
        
        for suite_name, test_func in test_suites:
            print(f"\n🔍 Testing {suite_name}...")
            print("-" * 40)
            
            try:
                result = test_func()
                suite_results.append(result)
            except Exception as e:
                print(f"❌ Test suite failed with exception: {e}")
                suite_results.append(False)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        all_suites_passed = all(suite_results)
        
        if all_suites_passed and self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED! Deployment is ready for production.")
            return True
        else:
            print(f"\n⚠️  {len([r for r in suite_results if not r])} test suite(s) failed.")
            print("Please review the issues above before deploying to production.")
            return False

def main():
    """Main test function."""
    import argparse

    # Update global timeout
    global TIMEOUT

    parser = argparse.ArgumentParser(description="Test Taxi Fare Prediction deployment")
    parser.add_argument("--backend", default=BACKEND_URL, help="Backend URL")
    parser.add_argument("--frontend", default=FRONTEND_URL, help="Frontend URL")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="Request timeout")

    args = parser.parse_args()

    TIMEOUT = args.timeout
    
    # Run tests
    tester = DeploymentTester(args.backend, args.frontend)
    success = tester.run_all_tests()
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "backend_url": args.backend,
            "frontend_url": args.frontend,
            "summary": {
                "total_tests": tester.tests_passed + tester.tests_failed,
                "passed": tester.tests_passed,
                "failed": tester.tests_failed,
                "success_rate": (tester.tests_passed / (tester.tests_passed + tester.tests_failed) * 100) if (tester.tests_passed + tester.tests_failed) > 0 else 0
            },
            "results": tester.test_results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: test_results.json")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

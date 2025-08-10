#!/usr/bin/env python3
"""
Debug script to investigate the model accuracy issues.
"""

import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

def debug_feature_processor():
    """Debug the feature processor to understand scaling parameters."""
    print("🔍 DEBUGGING FEATURE PROCESSOR")
    print("=" * 50)
    
    try:
        # Load feature processor
        processor = joblib.load("models/feature_processor.pkl")
        print(f"✅ Feature processor loaded: {type(processor)}")
        
        # Check if it's a dictionary of scalers or a single scaler
        if isinstance(processor, dict):
            print(f"📊 Processor contains {len(processor)} scalers:")
            for key, scaler in processor.items():
                print(f"   - {key}: {type(scaler)}")
                if hasattr(scaler, 'mean_'):
                    print(f"     Mean: {scaler.mean_}")
                    print(f"     Scale: {scaler.scale_}")
                elif hasattr(scaler, 'data_min_'):
                    print(f"     Min: {scaler.data_min_}")
                    print(f"     Max: {scaler.data_max_}")
        else:
            print(f"📊 Single processor: {type(processor)}")
            
    except Exception as e:
        print(f"❌ Failed to load feature processor: {e}")

def debug_model_prediction():
    """Test the model with known inputs to understand the scaling issue."""
    print("\n🧪 DEBUGGING MODEL PREDICTIONS")
    print("=" * 50)
    
    try:
        # Load model and processor
        model = joblib.load("models/best_taxi_fare_model.pkl")
        
        # Test with a simple case - short Manhattan trip
        # Expected fare: ~$8-12 for a 1-mile trip
        
        # Create test features manually (this is what the current interface does wrong)
        test_features = {
            'pickup_latitude_scaled': (40.7589 - 40.7589) / 0.1,  # 0.0
            'pickup_longitude_scaled': (-73.9851 + 73.9851) / 0.1,  # 0.0
            'dropoff_latitude_scaled': (40.7505 - 40.7589) / 0.1,  # -0.84
            'dropoff_longitude_scaled': (-73.9934 + 73.9851) / 0.1,  # -0.83
            'passenger_count': 2,
            'hour': 14,
            'day_of_week': 1,  # Monday
            'month': 1,
            'is_weekend': 0,
            'is_rush_hour': 0,
            'is_morning_rush': 0,
            'is_evening_rush': 0,
            'trip_distance_scaled': 1.0 / 10,  # Current wrong scaling
            'fare_per_mile_scaled': 2.5 / 10,  # Current wrong scaling
            'is_airport_trip': 0,
            'min_airport_dist_scaled': 5.0 / 10,
            'is_manhattan': 1,
            'weather_severity': 1,
            'traffic_severity': 1
        }
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([test_features])
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        print(f"🎯 Current prediction (wrong scaling): ${prediction:.2f}")
        
        # Now try with corrected scaling based on typical NYC taxi data
        # Trip distance: mean ~2.5 miles, std ~3.0 miles
        # Fare per mile: mean ~$2.50, std ~$1.50
        
        corrected_features = test_features.copy()
        corrected_features['trip_distance_scaled'] = (1.0 - 2.5) / 3.0  # Proper standardization
        corrected_features['fare_per_mile_scaled'] = (8.0 - 2.5) / 1.5  # Estimated $8/mile for short trip
        
        corrected_df = pd.DataFrame([corrected_features])
        corrected_prediction = model.predict(corrected_df)[0]
        print(f"🎯 Corrected prediction (better scaling): ${corrected_prediction:.2f}")
        
        # Test with different fare_per_mile values
        print(f"\n📊 Testing different fare_per_mile values:")
        for fpm in [2.0, 2.5, 3.0, 4.0, 5.0]:
            test_features_fpm = corrected_features.copy()
            test_features_fpm['fare_per_mile_scaled'] = (fpm - 2.5) / 1.5
            test_df = pd.DataFrame([test_features_fpm])
            pred = model.predict(test_df)[0]
            print(f"   Fare per mile ${fpm:.1f} -> Prediction: ${pred:.2f}")
            
    except Exception as e:
        print(f"❌ Failed to debug model: {e}")

def analyze_training_data():
    """Analyze the training data to understand proper scaling parameters."""
    print("\n📈 ANALYZING TRAINING DATA")
    print("=" * 50)
    
    try:
        # Try to load some training data to understand the scaling
        train_data = pd.read_csv("../Task_5_1/data_preprocessing/processed_data/train_dataset.csv")
        
        print(f"📊 Training data shape: {train_data.shape}")
        
        if 'trip_distance' in train_data.columns:
            print(f"\n🚗 Trip Distance Statistics:")
            print(f"   Mean: {train_data['trip_distance'].mean():.3f}")
            print(f"   Std:  {train_data['trip_distance'].std():.3f}")
            print(f"   Min:  {train_data['trip_distance'].min():.3f}")
            print(f"   Max:  {train_data['trip_distance'].max():.3f}")
            
        if 'fare_per_mile' in train_data.columns:
            print(f"\n💰 Fare Per Mile Statistics:")
            print(f"   Mean: ${train_data['fare_per_mile'].mean():.3f}")
            print(f"   Std:  ${train_data['fare_per_mile'].std():.3f}")
            print(f"   Min:  ${train_data['fare_per_mile'].min():.3f}")
            print(f"   Max:  ${train_data['fare_per_mile'].max():.3f}")
            
        if 'fare_amount' in train_data.columns:
            print(f"\n💵 Fare Amount Statistics:")
            print(f"   Mean: ${train_data['fare_amount'].mean():.3f}")
            print(f"   Std:  ${train_data['fare_amount'].std():.3f}")
            print(f"   Min:  ${train_data['fare_amount'].min():.3f}")
            print(f"   Max:  ${train_data['fare_amount'].max():.3f}")
            
    except Exception as e:
        print(f"❌ Failed to analyze training data: {e}")

def main():
    """Main debug function."""
    print("🚖 TAXI FARE MODEL DEBUG ANALYSIS")
    print("=" * 60)
    
    debug_feature_processor()
    debug_model_prediction()
    analyze_training_data()
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSIONS:")
    print("1. Check if feature processor contains proper scaling parameters")
    print("2. Current manual scaling in prediction_interface.py is likely incorrect")
    print("3. Need to use proper StandardScaler parameters from training")
    print("4. fare_per_mile should be calculated dynamically, not set to constant")

if __name__ == "__main__":
    main()

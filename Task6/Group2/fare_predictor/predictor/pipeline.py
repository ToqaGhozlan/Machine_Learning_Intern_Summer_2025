
"""
Robust preprocessing & feature pipeline for fare prediction.
Designed to accept `cleaned_data` (a dict from Django Form.cleaned_data),
and produce model-ready features in the exact order required by the models.

Main functions:
 - preprocess_input(cleaned_data) -> DataFrame (renamed & numeric conversions)
 - feature_engineering(df) -> DataFrame (PCA of landmark distances + drop originals)
 - feature_encoding(df) -> DataFrame (one-hot + mapping)
 - prepare_features(df) -> numpy array (ordered to feature_order)
"""

import joblib
import os
import numpy as np
import pandas as pd

# Load model once (optimized for performance)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'predictor', 'ml_models')

# Load models
model_gb = joblib.load(os.path.join(MODELS_DIR, 'gb_model.pkl'))
model_gxb = joblib.load(os.path.join(MODELS_DIR, 'gxb_model.pkl'))
model_rf = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))

# Load preprocessing tools
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
pca = joblib.load(os.path.join(MODELS_DIR, 'pca.pkl'))
feature_order = joblib.load(os.path.join(MODELS_DIR, 'feature_order.pkl'))

def preprocess_input(cleaned_data : dict) -> pd.DataFrame:
    """
    Turn validated form data into model-ready format.
    """
   # If it's already a DataFrame, just copy it
    if isinstance(cleaned_data, pd.DataFrame):
        df = cleaned_data.copy()
    # If it's a dict, wrap in a list to make a DataFrame
    elif isinstance(cleaned_data, dict):
        df = pd.DataFrame([cleaned_data])
    else:
        raise ValueError(f"Unexpected type for cleaned_data: {type(cleaned_data)}")


    df = df.rename(columns={
            'car_condition': 'Car Condition',
            'user_name': 'User Name',
            'driver_name': 'Driver Name',
            'Traffic': 'Traffic Condition',
        })

    # Add distance_log
    df['distance_log'] = np.log1p(df['distance'])

    # Drop original distance
    df.drop(['distance'], axis=1, inplace=True)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering steps.
    """
    cols = ['jfk_airport_distance', 'ewr_airport_distance', 'lga_airport_distance',
        'nyc_center_distance', 'statue_of_liberty_distance']

    X_landmarks = df[cols]
    # Scale features
    X_scaled = scaler.transform(X_landmarks)
    
    # Apply PCA
    df['landmark_distance_pca'] = pca.transform(X_scaled)
    
    return df 

def feature_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features.
    """
    cat_cols = ['Weather', 'Traffic Condition']

    # Convert categorical columns to category dtype 
    df_encode = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    car_condition_map = {
        "Bad": 0,
        "Good": 1,
        "Very Good": 2,
        "Excellent": 3
    }
    df_encode['Car Condition'] = df_encode['Car Condition'].map(car_condition_map)
    
    return df_encode

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Aligns incoming form data with the exact training feature order.
    Missing features are filled with 0.
    """
    # Reorder columns
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0  # Fill missing with 0 (or np.nan)
    df = df[feature_order]
    return df

def predict_fare_gb(df):
    """
    Predict fare using Gradient Boosting model.
    """
    prediction = model_gb.predict(df)[0]
    return round(prediction, 2)

def predict_fare_gxb(df):
    """
    Predict fare using Gradient Boosting with XGBoost model.
    """
    prediction = model_gxb.predict(df)[0]
    return round(prediction, 2)

def predict_fare_rf(df):
    """
    Predict fare using Random Forest model.
    """
    prediction = model_rf.predict(df)[0]
    return round(prediction, 2)



#!/usr/bin/env python3
"""
Backend Deployment Validation Script for Render Free Tier
"""

import os
import sys
from pathlib import Path

def check_backend_ready():
    """Check if backend is ready for Render deployment."""
    print("🔍 Checking Backend Deployment Readiness...")
    
    issues = []
    
    # Check required files
    required_files = [
        "requirements.txt",
        "app/main.py",
        "app/core/config.py",
        "models/best_taxi_fare_model.pkl",
        "models/feature_processor.pkl",
        "models/final_model_metadata.json"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"❌ Missing file: {file}")
        else:
            print(f"✅ Found: {file}")
    
    # Check model file size
    model_path = "models/best_taxi_fare_model.pkl"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📊 Model size: {size_mb:.1f} MB")
        if size_mb > 500:
            issues.append(f"⚠️ WARNING: Model size ({size_mb:.1f} MB) might exceed free tier limits")
    
    # Check Python dependencies
    try:
        with open("requirements.txt", "r") as f:
            deps = f.read()
            if "fastapi" not in deps:
                issues.append("❌ FastAPI not found in requirements.txt")
            if "uvicorn" not in deps:
                issues.append("❌ Uvicorn not found in requirements.txt")
            if "gunicorn" not in deps:
                issues.append("❌ Gunicorn not found in requirements.txt")
    except Exception as e:
        issues.append(f"❌ Error reading requirements.txt: {e}")
    
    # Summary
    if issues:
        print("\n⚠️ Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\n🔧 Please fix these issues before deploying.")
        return False
    else:
        print("\n✅ Backend is ready for Render deployment!")
        print("\n📋 Deployment Commands:")
        print("Build Command: pip install --upgrade pip && pip install -r requirements.txt")
        print("Start Command: gunicorn app.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120")
        return True

if __name__ == "__main__":
    success = check_backend_ready()
    sys.exit(0 if success else 1)

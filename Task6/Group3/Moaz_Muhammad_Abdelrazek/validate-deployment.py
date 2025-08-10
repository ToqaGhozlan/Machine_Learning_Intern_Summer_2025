#!/usr/bin/env python3
"""
Taxi Fare Prediction ML - Deployment Validation Script
This script validates that the project is ready for Render deployment.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m", 
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{colors.get(status, '')}[{status}] {message}{colors['RESET']}")

def check_file_exists(filepath, required=True):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print_status(f"✓ {filepath}", "SUCCESS")
        return True
    else:
        status = "ERROR" if required else "WARNING"
        print_status(f"✗ {filepath} (missing)", status)
        return False

def validate_json_file(filepath):
    """Validate JSON file format."""
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print_status(f"✓ {filepath} (valid JSON)", "SUCCESS")
        return True
    except json.JSONDecodeError as e:
        print_status(f"✗ {filepath} (invalid JSON): {e}", "ERROR")
        return False
    except FileNotFoundError:
        print_status(f"✗ {filepath} (not found)", "ERROR")
        return False

def check_command_available(command):
    """Check if a command is available in PATH."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print_status("🚀 Validating Taxi Fare Prediction ML for Render Deployment", "INFO")
    print("=" * 60)
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    validation_passed = True
    
    # Check prerequisites
    print_status("Checking prerequisites...", "INFO")
    
    if check_command_available("git"):
        print_status("Git is available", "SUCCESS")
    else:
        print_status("Git is not available", "WARNING")
    
    if check_command_available("python") or check_command_available("python3"):
        print_status("Python is available", "SUCCESS")
    else:
        print_status("Python is not available", "WARNING")
    
    if check_command_available("node"):
        print_status("Node.js is available", "SUCCESS")
    else:
        print_status("Node.js is not available", "WARNING")
    
    # Check project structure
    print_status("Validating project structure...", "INFO")
    
    required_files = [
        "render.yaml",
        "backend/requirements.txt",
        "backend/app/main.py",
        "backend/app/core/config.py",
        "frontend/package.json",
        "frontend/src/App.tsx",
        "backend/models/best_taxi_fare_model.pkl",
        "backend/models/feature_processor.pkl",
        "backend/models/final_model_metadata.json"
    ]
    
    for file_path in required_files:
        if not check_file_exists(file_path):
            validation_passed = False
    
    # Validate JSON files
    print_status("Validating JSON configuration files...", "INFO")
    
    json_files = [
        "backend/models/final_model_metadata.json",
        "frontend/package.json"
    ]
    
    for json_file in json_files:
        if os.path.exists(json_file):
            if not validate_json_file(json_file):
                validation_passed = False
    
    # Check model file size
    print_status("Checking model file sizes...", "INFO")
    
    model_file = "backend/models/best_taxi_fare_model.pkl"
    if os.path.exists(model_file):
        size_bytes = os.path.getsize(model_file)
        size_mb = size_bytes / (1024 * 1024)
        print_status(f"Model file size: {size_mb:.2f} MB", "INFO")
        
        if size_bytes > 100 * 1024 * 1024:  # 100MB
            print_status(f"Model file is large ({size_mb:.2f} MB). Ensure Git LFS is configured.", "WARNING")
    
    # Check render.yaml configuration
    print_status("Validating render.yaml configuration...", "INFO")
    
    if os.path.exists("render.yaml"):
        try:
            with open("render.yaml", 'r') as f:
                render_config = f.read()
            
            required_services = ["taxi-fare-api", "taxi-fare-frontend"]
            for service in required_services:
                if service in render_config:
                    print_status(f"✓ Service '{service}' found in render.yaml", "SUCCESS")
                else:
                    print_status(f"✗ Service '{service}' missing in render.yaml", "ERROR")
                    validation_passed = False
        except Exception as e:
            print_status(f"Error reading render.yaml: {e}", "ERROR")
            validation_passed = False
    
    # Check environment files
    print_status("Checking environment configuration...", "INFO")
    
    env_files = [
        "backend/.env.example",
        "frontend/.env.example"
    ]
    
    for env_file in env_files:
        check_file_exists(env_file, required=False)
    
    # Final validation summary
    print("\n" + "=" * 60)
    if validation_passed:
        print_status("🎉 All validations passed! Project is ready for Render deployment.", "SUCCESS")
        print_status("Next steps:", "INFO")
        print("1. Commit and push changes to GitHub")
        print("2. Create Render account and connect repository")
        print("3. Deploy using Blueprint (render.yaml)")
        print("4. Follow RENDER_DEPLOYMENT_GUIDE.md for detailed instructions")
        return 0
    else:
        print_status("❌ Some validations failed. Please fix the issues above before deploying.", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main())

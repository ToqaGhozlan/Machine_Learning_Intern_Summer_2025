# Taxi Fare Prediction ML - Render Deployment Preparation Script (PowerShell)
# This script prepares the project for deployment to Render

param(
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Taxi Fare Prediction ML - Render Deployment Preparation Script

This script prepares your project for deployment to Render by:
- Checking prerequisites and project structure
- Validating configuration files
- Setting up Git LFS for large model files
- Creating deployment checklist

Usage:
    .\prepare-render-deployment.ps1

Requirements:
- Git installed and available in PATH
- Git LFS (recommended for large model files)
- PowerShell 5.0 or later

"@
    exit 0
}

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🚀 Preparing Taxi Fare Prediction ML for Render Deployment" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if we're in the correct directory
if (-not (Test-Path "render.yaml")) {
    Write-Error "render.yaml not found. Please run this script from the Task_5_2 directory."
    exit 1
}

Write-Status "Checking prerequisites..."

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Success "Git is available: $gitVersion"
} catch {
    Write-Error "Git is not installed. Please install Git first."
    exit 1
}

# Check if Git LFS is installed
try {
    $lfsVersion = git lfs version
    Write-Success "Git LFS is available: $lfsVersion"
} catch {
    Write-Warning "Git LFS is not installed. Installing Git LFS is recommended for handling large model files."
    Write-Host "Please install Git LFS from: https://git-lfs.github.io/"
}

# Check if Python is available (for local testing)
try {
    $pythonVersion = python --version 2>$null
    if (-not $pythonVersion) {
        $pythonVersion = python3 --version 2>$null
    }
    if ($pythonVersion) {
        Write-Success "Python is available: $pythonVersion"
    }
} catch {
    Write-Warning "Python not found. This is okay for deployment but needed for local testing."
}

# Check if Node.js is available (for local testing)
try {
    $nodeVersion = node --version
    Write-Success "Node.js is available: $nodeVersion"
} catch {
    Write-Warning "Node.js not found. This is okay for deployment but needed for local testing."
}

Write-Status "Validating project structure..."

# Check required files
$requiredFiles = @(
    "render.yaml",
    "backend/requirements.txt",
    "backend/app/main.py",
    "frontend/package.json",
    "frontend/src/App.tsx",
    "backend/models/best_taxi_fare_model.pkl",
    "backend/models/feature_processor.pkl",
    "backend/models/final_model_metadata.json"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Success "✓ $file"
    } else {
        Write-Error "✗ $file (missing)"
        exit 1
    }
}

Write-Status "Checking model file sizes..."

# Check model file sizes
$modelFile = "backend/models/best_taxi_fare_model.pkl"
if (Test-Path $modelFile) {
    $modelSize = (Get-Item $modelFile).Length
    $modelSizeMB = [math]::Round($modelSize / 1MB, 2)
    Write-Status "Model file size: $modelSizeMB MB"
    
    if ($modelSize -gt 100000000) {
        Write-Warning "Model file is large ($modelSizeMB MB). Ensure Git LFS is configured."
    }
}

Write-Status "Preparing Git repository..."

# Initialize Git LFS if not already done
if (Test-Path ".gitattributes") {
    $gitattributes = Get-Content ".gitattributes" -Raw
    if ($gitattributes -match "\*\.pkl") {
        Write-Success "Git LFS is already configured for .pkl files"
    } else {
        Write-Status "Adding .pkl files to Git LFS tracking"
        Add-Content ".gitattributes" "`n*.pkl filter=lfs diff=lfs merge=lfs -text"
    }
} else {
    Write-Status "Creating .gitattributes for Git LFS"
    Set-Content ".gitattributes" "*.pkl filter=lfs diff=lfs merge=lfs -text"
}

# Check if Git repository is initialized
if (-not (Test-Path ".git")) {
    Write-Status "Initializing Git repository..."
    git init
    try {
        git lfs install
    } catch {
        Write-Warning "Could not initialize Git LFS. Please install Git LFS if you haven't already."
    }
}

Write-Status "Validating render.yaml configuration..."

# Basic validation of render.yaml
$renderConfig = Get-Content "render.yaml" -Raw
if ($renderConfig -match "taxi-fare-api" -and $renderConfig -match "taxi-fare-frontend") {
    Write-Success "render.yaml contains required services"
} else {
    Write-Error "render.yaml is missing required services"
    exit 1
}

Write-Status "Creating deployment checklist..."

$checklistContent = @"
# 📋 Render Deployment Checklist

## Pre-Deployment
- [ ] Git repository is initialized and pushed to GitHub
- [ ] Git LFS is configured for model files
- [ ] All dependencies are updated to latest versions
- [ ] Environment variables are configured in render.yaml
- [ ] Model files are committed and accessible

## Deployment Steps
- [ ] Create Render account at render.com
- [ ] Connect GitHub repository to Render
- [ ] Deploy using Blueprint (render.yaml)
- [ ] Monitor build logs for both services
- [ ] Verify health checks pass

## Post-Deployment
- [ ] Test backend API endpoints
- [ ] Test frontend application
- [ ] Update CORS origins with actual frontend URL
- [ ] Update API URL in frontend environment
- [ ] Set up monitoring and alerts
- [ ] Configure custom domain (optional)

## URLs (Update after deployment)
- Frontend: https://taxi-fare-frontend.onrender.com
- Backend: https://taxi-fare-api.onrender.com
- API Docs: https://taxi-fare-api.onrender.com/docs

## Environment Variables to Set in Render

### Backend Service
```
ENVIRONMENT=production
LOG_LEVEL=info
WORKERS=2
CORS_ORIGINS=https://taxi-fare-frontend.onrender.com
MODEL_PATH=./models/best_taxi_fare_model.pkl
PROCESSOR_PATH=./models/feature_processor.pkl
METADATA_PATH=./models/final_model_metadata.json
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### Frontend Service
```
NODE_ENV=production
REACT_APP_API_URL=https://taxi-fare-api.onrender.com
REACT_APP_ENVIRONMENT=production
GENERATE_SOURCEMAP=false
CI=false
```
"@

Set-Content "DEPLOYMENT_CHECKLIST.md" $checklistContent
Write-Success "Created DEPLOYMENT_CHECKLIST.md"

Write-Status "Running final validation..."

# Check if all files are ready for commit
try {
    $gitStatus = git status --porcelain
    if ($gitStatus) {
        Write-Status "There are uncommitted changes. Ready to commit and push."
    } else {
        Write-Success "All files are committed."
    }
} catch {
    Write-Warning "Could not check Git status."
}

Write-Host ""
Write-Host "🎉 Deployment preparation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Review DEPLOYMENT_CHECKLIST.md"
Write-Host "2. Commit and push changes to GitHub:"
Write-Host "   git add ."
Write-Host "   git commit -m `"Prepare for Render deployment`""
Write-Host "   git push origin main"
Write-Host "3. Follow RENDER_DEPLOYMENT_GUIDE.md for deployment"
Write-Host ""
Write-Host "Happy deploying! 🚀" -ForegroundColor Green

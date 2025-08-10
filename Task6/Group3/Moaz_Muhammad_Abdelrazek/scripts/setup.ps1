# Taxi Fare Prediction - Setup Script (PowerShell)
# This script sets up the development environment for the application

Write-Host "🚖 Setting up Taxi Fare Prediction Application" -ForegroundColor Green
Write-Host "=============================================="

# Check if we're in the right directory
if (-not (Test-Path "README.md")) {
    Write-Host "❌ Please run this script from the Task_5_2 directory" -ForegroundColor Red
    exit 1
}

# Function to check if command exists
function Test-Command($command) {
    try {
        Get-Command $command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-Command "python")) {
    Write-Host "❌ Python is required but not installed" -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "node")) {
    Write-Host "❌ Node.js is required but not installed" -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "npm")) {
    Write-Host "❌ npm is required but not installed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Prerequisites check passed" -ForegroundColor Green

# Setup backend
Write-Host ""
Write-Host "🔧 Setting up backend..." -ForegroundColor Yellow
Set-Location backend

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "venv\Scripts\Activate.ps1"

# Install Python dependencies
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..."
    Copy-Item ".env.example" ".env"
    Write-Host "⚠️  Please review and update .env file with your configuration" -ForegroundColor Yellow
}

# Test model loading
Write-Host "Testing model integration..."
python test_model.py

Set-Location ..

# Setup frontend
Write-Host ""
Write-Host "🎨 Setting up frontend..." -ForegroundColor Yellow
Set-Location frontend

# Install Node.js dependencies
Write-Host "Installing Node.js dependencies..."
npm install

# Copy environment file
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..."
    Copy-Item ".env.example" ".env"
    Write-Host "⚠️  Please review and update .env file with your configuration" -ForegroundColor Yellow
}

Set-Location ..

# Create logs directory
if (-not (Test-Path "backend\logs")) {
    New-Item -ItemType Directory -Path "backend\logs" -Force
}

Write-Host ""
Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Review and update environment files:"
Write-Host "   - backend\.env"
Write-Host "   - frontend\.env"
Write-Host ""
Write-Host "2. Start the backend server:"
Write-Host "   cd backend"
Write-Host "   venv\Scripts\Activate.ps1"
Write-Host "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
Write-Host ""
Write-Host "3. Start the frontend (in a new terminal):"
Write-Host "   cd frontend"
Write-Host "   npm start"
Write-Host ""
Write-Host "4. Open your browser to http://localhost:3000"
Write-Host ""
Write-Host "🚀 Happy predicting!" -ForegroundColor Green

#!/bin/bash

# Taxi Fare Prediction ML - Render Deployment Preparation Script
# This script prepares the project for deployment to Render

set -e  # Exit on any error

echo "🚀 Preparing Taxi Fare Prediction ML for Render Deployment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "render.yaml" ]; then
    print_error "render.yaml not found. Please run this script from the Task_5_2 directory."
    exit 1
fi

print_status "Checking prerequisites..."

# Check if Git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    print_warning "Git LFS is not installed. Installing Git LFS is recommended for handling large model files."
    echo "Please install Git LFS from: https://git-lfs.github.io/"
else
    print_success "Git LFS is available"
fi

# Check if Python is available (for local testing)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION is available"
else
    print_warning "Python 3 not found. This is okay for deployment but needed for local testing."
fi

# Check if Node.js is available (for local testing)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION is available"
else
    print_warning "Node.js not found. This is okay for deployment but needed for local testing."
fi

print_status "Validating project structure..."

# Check required files
REQUIRED_FILES=(
    "render.yaml"
    "backend/requirements.txt"
    "backend/app/main.py"
    "frontend/package.json"
    "frontend/src/App.tsx"
    "backend/models/best_taxi_fare_model.pkl"
    "backend/models/feature_processor.pkl"
    "backend/models/final_model_metadata.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "✓ $file"
    else
        print_error "✗ $file (missing)"
        exit 1
    fi
done

print_status "Checking model file sizes..."

# Check model file sizes
MODEL_SIZE=$(du -h backend/models/best_taxi_fare_model.pkl | cut -f1)
print_status "Model file size: $MODEL_SIZE"

if [ -f "backend/models/best_taxi_fare_model.pkl" ]; then
    SIZE_BYTES=$(stat -f%z backend/models/best_taxi_fare_model.pkl 2>/dev/null || stat -c%s backend/models/best_taxi_fare_model.pkl 2>/dev/null)
    if [ "$SIZE_BYTES" -gt 100000000 ]; then  # 100MB
        print_warning "Model file is large ($MODEL_SIZE). Ensure Git LFS is configured."
    fi
fi

print_status "Preparing Git repository..."

# Initialize Git LFS if not already done
if [ -f ".gitattributes" ]; then
    if grep -q "*.pkl" .gitattributes; then
        print_success "Git LFS is already configured for .pkl files"
    else
        print_status "Adding .pkl files to Git LFS tracking"
        echo "*.pkl filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
    fi
else
    print_status "Creating .gitattributes for Git LFS"
    echo "*.pkl filter=lfs diff=lfs merge=lfs -text" > .gitattributes
fi

# Check if Git repository is initialized
if [ ! -d ".git" ]; then
    print_status "Initializing Git repository..."
    git init
    git lfs install
fi

print_status "Validating render.yaml configuration..."

# Basic validation of render.yaml
if grep -q "taxi-fare-api" render.yaml && grep -q "taxi-fare-frontend" render.yaml; then
    print_success "render.yaml contains required services"
else
    print_error "render.yaml is missing required services"
    exit 1
fi

print_status "Creating deployment checklist..."

cat << EOF > DEPLOYMENT_CHECKLIST.md
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
\`\`\`
ENVIRONMENT=production
LOG_LEVEL=info
WORKERS=2
CORS_ORIGINS=https://taxi-fare-frontend.onrender.com
MODEL_PATH=./models/best_taxi_fare_model.pkl
PROCESSOR_PATH=./models/feature_processor.pkl
METADATA_PATH=./models/final_model_metadata.json
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
\`\`\`

### Frontend Service
\`\`\`
NODE_ENV=production
REACT_APP_API_URL=https://taxi-fare-api.onrender.com
REACT_APP_ENVIRONMENT=production
GENERATE_SOURCEMAP=false
CI=false
\`\`\`
EOF

print_success "Created DEPLOYMENT_CHECKLIST.md"

print_status "Running final validation..."

# Check if all files are ready for commit
if git status --porcelain | grep -q .; then
    print_status "There are uncommitted changes. Ready to commit and push."
else
    print_success "All files are committed."
fi

echo ""
echo "🎉 Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Review DEPLOYMENT_CHECKLIST.md"
echo "2. Commit and push changes to GitHub:"
echo "   git add ."
echo "   git commit -m 'Prepare for Render deployment'"
echo "   git push origin main"
echo "3. Follow RENDER_DEPLOYMENT_GUIDE.md for deployment"
echo ""
echo "Happy deploying! 🚀"

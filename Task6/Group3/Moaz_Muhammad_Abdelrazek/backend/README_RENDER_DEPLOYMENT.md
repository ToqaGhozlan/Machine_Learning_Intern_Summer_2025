# 🚀 Backend Deployment Guide - Render Free Tier

## Quick Deployment Steps

### 1. Create New Web Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository: `moazmo/taxi-fare-prediction-ml`

### 2. Configure Service Settings
- **Name**: `taxi-fare-api` (or any name you prefer)
- **Region**: `Oregon (US West)` (free tier)
- **Branch**: `main`
- **Root Directory**: `backend`
- **Runtime**: `Python 3`

### 3. Build & Start Commands
```bash
# Build Command:
pip install --upgrade pip && pip install -r requirements.txt

# Start Command:
gunicorn app.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120
```

### 4. Environment Variables
Add these in Render dashboard:
```
ENVIRONMENT=production
LOG_LEVEL=info
HOST=0.0.0.0
WORKERS=1
MODEL_PATH=./models/best_taxi_fare_model.pkl
PROCESSOR_PATH=./models/feature_processor.pkl
METADATA_PATH=./models/final_model_metadata.json
CORS_ORIGINS=*
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### 5. Important Notes
- **Free Tier Limitations**: Service sleeps after 15 minutes of inactivity
- **Memory**: 512MB RAM limit (our model is ~487MB, should fit)
- **Build Time**: First deploy may take 10-15 minutes due to large model file
- **Health Check**: Available at `/health` endpoint

### 6. After Deployment
- Your API will be available at: `https://your-service-name.onrender.com`
- Test the API: `https://your-service-name.onrender.com/docs`
- Copy the URL for frontend configuration

## Troubleshooting
- If build fails due to memory, the model file might be too large for free tier
- Check build logs for dependency issues
- Ensure Git LFS is working for model files

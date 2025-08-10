# 🚀 Render Deployment Guide - Taxi Fare Prediction ML Application

This guide provides step-by-step instructions for deploying the complete Taxi Fare Prediction application to Render using the Blueprint approach.

## 📋 Prerequisites

### Required Accounts & Tools
- **Render Account**: Sign up at [render.com](https://render.com)
- **GitHub Account**: For repository hosting
- **Git**: Installed locally for version control

### System Requirements
- **Python**: 3.11+ (for local testing)
- **Node.js**: 18+ (for local testing)
- **Git LFS**: For handling large ML model files

## 🏗️ Pre-Deployment Setup

### 1. Repository Preparation

```bash
# Clone or navigate to your project
cd Task_5_2

# Initialize Git LFS for large model files (if not already done)
git lfs install
git lfs track "*.pkl"
git add .gitattributes

# Add all files and commit
git add .
git commit -m "Prepare for Render deployment with updated dependencies"

# Push to GitHub (create repository if needed)
git remote add origin https://github.com/yourusername/taxi-fare-prediction-ml.git
git branch -M main
git push -u origin main
```

### 2. Environment Configuration

**Backend Environment Variables** (will be set in Render):
- `ENVIRONMENT=production`
- `LOG_LEVEL=info`
- `WORKERS=2`
- `CORS_ORIGINS=https://taxi-fare-frontend.onrender.com`
- `MODEL_PATH=./models/best_taxi_fare_model.pkl`
- `PROCESSOR_PATH=./models/feature_processor.pkl`
- `METADATA_PATH=./models/final_model_metadata.json`

**Frontend Environment Variables** (will be set in Render):
- `REACT_APP_API_URL=https://taxi-fare-api.onrender.com`
- `REACT_APP_ENVIRONMENT=production`
- `GENERATE_SOURCEMAP=false`
- `CI=false`

## 🚢 Deployment Methods

### Method 1: Blueprint Deployment (Recommended)

This method deploys both services simultaneously using infrastructure-as-code.

#### Step 1: Prepare Blueprint File

The `render.yaml` file is already configured in your project root. Update the following:

```yaml
# In render.yaml, update these values:
notifications:
  - type: email
    emails:
      - your-email@example.com  # Replace with your actual email
```

#### Step 2: Deploy via Render Dashboard

1. **Login to Render**: Go to [dashboard.render.com](https://dashboard.render.com)

2. **Create New Blueprint**:
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository containing your project
   - Render will automatically detect the `render.yaml` file

3. **Configure Services**:
   - Review the services that will be created:
     - `taxi-fare-api` (Backend FastAPI service)
     - `taxi-fare-frontend` (Frontend React service)
     - `taxi-predictions-db` (PostgreSQL database - optional)

4. **Deploy**:
   - Click "Apply" to start deployment
   - Monitor the build logs for both services
   - Deployment typically takes 5-10 minutes

#### Step 3: Post-Deployment Configuration

1. **Update CORS Origins**:
   - Once frontend is deployed, note its URL (e.g., `https://taxi-fare-frontend.onrender.com`)
   - Update backend environment variable `CORS_ORIGINS` to include this URL

2. **Update API URL**:
   - Note the backend URL (e.g., `https://taxi-fare-api.onrender.com`)
   - Update frontend environment variable `REACT_APP_API_URL`

### Method 2: Manual Service Creation

If you prefer to create services individually:

#### Backend Service

1. **Create Web Service**:
   - New + → Web Service
   - Connect GitHub repository
   - Root Directory: `backend`
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT`

2. **Environment Variables**:
   ```
   ENVIRONMENT=production
   LOG_LEVEL=info
   WORKERS=2
   MODEL_PATH=./models/best_taxi_fare_model.pkl
   PROCESSOR_PATH=./models/feature_processor.pkl
   METADATA_PATH=./models/final_model_metadata.json
   PYTHONUNBUFFERED=1
   PYTHONDONTWRITEBYTECODE=1
   ```

#### Frontend Service

1. **Create Web Service**:
   - New + → Web Service
   - Connect GitHub repository
   - Root Directory: `frontend`
   - Runtime: Node
   - Build Command: `npm ci && npm run build:production`
   - Start Command: `npm run serve`

2. **Environment Variables**:
   ```
   NODE_ENV=production
   REACT_APP_API_URL=https://your-backend-url.onrender.com
   REACT_APP_ENVIRONMENT=production
   GENERATE_SOURCEMAP=false
   CI=false
   ```

## 🔧 Configuration Details

### Service Plans

**Starter Plan (Free Tier)**:
- 512 MB RAM
- 0.1 CPU
- Sleeps after 15 minutes of inactivity
- Perfect for development/testing

**Standard Plan ($7/month per service)**:
- 2 GB RAM
- 1 CPU
- No sleep
- Recommended for production

### Health Checks

Both services include health check endpoints:
- **Backend**: `GET /health`
- **Frontend**: `GET /` (root path)

### Auto-Deploy

Services are configured to auto-deploy on:
- Push to `main` branch
- Pull request merge

## 🧪 Testing Deployment

### 1. Backend API Testing

```bash
# Health check
curl https://taxi-fare-api.onrender.com/health

# Model info
curl https://taxi-fare-api.onrender.com/api/v1/model/info

# Prediction test
curl -X POST https://taxi-fare-api.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 40.7589,
    "pickup_longitude": -73.9851,
    "dropoff_latitude": 40.7505,
    "dropoff_longitude": -73.9934,
    "passenger_count": 2,
    "pickup_datetime": "2024-01-15T14:30:00"
  }'
```

### 2. Frontend Testing

1. Open `https://taxi-fare-frontend.onrender.com`
2. Test the interactive map interface
3. Submit a prediction request
4. Verify results display correctly

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check build logs in Render dashboard
   - Verify all dependencies are listed in requirements.txt/package.json
   - Ensure Python/Node versions are compatible

2. **Model Loading Issues**:
   - Verify model files are committed with Git LFS
   - Check file paths in environment variables
   - Monitor memory usage (models require ~500MB RAM)

3. **CORS Errors**:
   - Update CORS_ORIGINS environment variable
   - Include both HTTP and HTTPS URLs
   - Restart backend service after changes

4. **Service Communication**:
   - Verify API URL in frontend environment variables
   - Check network connectivity between services
   - Review service logs for connection errors

### Performance Optimization

1. **Backend**:
   - Increase worker count for higher traffic
   - Enable prediction caching
   - Use Standard plan to avoid cold starts

2. **Frontend**:
   - Enable build optimizations
   - Use CDN for static assets
   - Implement service worker for caching

## 📞 Support & Monitoring

### Monitoring

- **Service Metrics**: Available in Render dashboard
- **Logs**: Real-time logs for debugging
- **Alerts**: Email notifications for deployment events

### Getting Help

- **Render Documentation**: [render.com/docs](https://render.com/docs)
- **Community Support**: [community.render.com](https://community.render.com)
- **Status Page**: [status.render.com](https://status.render.com)

## 🎉 Success!

Once deployed, your application will be available at:
- **Frontend**: `https://taxi-fare-frontend.onrender.com`
- **Backend API**: `https://taxi-fare-api.onrender.com`
- **API Documentation**: `https://taxi-fare-api.onrender.com/docs`

The application is now ready for production use with automatic scaling, monitoring, and deployment capabilities!

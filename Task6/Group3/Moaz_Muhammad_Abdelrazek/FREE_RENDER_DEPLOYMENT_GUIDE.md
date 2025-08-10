# 🚀 Complete Free Render Deployment Guide

## 🌐 **LIVE DEPLOYMENT**
**✅ The application is now live and ready to use!**

### 🔗 **Live Links:**
- **🎯 Frontend App**: https://taxi-fare-frontend-wf09.onrender.com/
- **⚡ Backend API**: https://taxi-fare-api.onrender.com

### 📋 **How to Use:**
1. **First**: Open the backend link above to wake up the API (may take 1-2 minutes)
2. **Then**: Open the frontend app link 
3. **Wait**: Initial load may take 1-2 minutes as free services wake up
4. **Enjoy**: Click on the map to predict taxi fares with ML!

---

## Overview
This guide shows how to deploy the Taxi Fare Prediction ML application using Render's **free tier** by deploying backend and frontend as separate services.

## 📋 Prerequisites
- GitHub account with the repository: `moazmo/taxi-fare-prediction-ml`
- Render account (free): [render.com](https://render.com)

## Part 1: Deploy Backend API (Free Web Service)

### Step 1: Create Backend Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository: `moazmo/taxi-fare-prediction-ml`

### Step 2: Configure Backend Settings
```
Name: taxi-fare-api
Region: Oregon (US West)
Branch: main
Root Directory: backend
Runtime: Python 3
Instance Type: Free
```

### Step 3: Build & Start Commands
```bash
# Build Command:
pip install --upgrade pip && pip install -r requirements.txt

# Start Command:
gunicorn app.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120
```

### Step 4: Environment Variables
Add these in the "Environment" section:
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

### Step 5: Deploy Backend
1. Click **"Create Web Service"**
2. Wait for deployment (10-15 minutes for first deploy)
3. **Copy your backend URL** (e.g., `https://taxi-fare-api-abc123.onrender.com`)

---

## Part 2: Deploy Frontend (Free Static Site)

### Step 1: Create Frontend Service
1. Go back to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Static Site"**
3. Connect the same GitHub repository: `moazmo/taxi-fare-prediction-ml`

### Step 2: Configure Frontend Settings
```
Name: taxi-fare-frontend
Region: Oregon (US West)
Branch: main
Root Directory: frontend
```

### Step 3: Build Settings
```bash
# Build Command:
npm ci && npm run build

# Publish Directory:
build
```

### Step 4: Environment Variables
**⚠️ IMPORTANT**: Use your actual backend URL from Part 1!

```
NODE_ENV=production
REACT_APP_API_URL=https://taxi-fare-api.onrender.com
REACT_APP_ENVIRONMENT=production
REACT_APP_DEFAULT_LAT=40.7589
REACT_APP_DEFAULT_LNG=-73.9851
REACT_APP_DEFAULT_ZOOM=12
GENERATE_SOURCEMAP=false
CI=false
```

### Step 5: Deploy Frontend
1. Click **"Create Static Site"**
2. Wait for deployment (5-10 minutes)
3. Your app will be live at: `https://taxi-fare-frontend-xyz789.onrender.com`

---

## 🎉 Testing Your Deployment

### Backend Testing
1. Visit: `https://your-backend-url.onrender.com/health`
2. Should return: `{"status": "healthy", "timestamp": "..."}`
3. Visit: `https://your-backend-url.onrender.com/docs`
4. Should show interactive API documentation

### Frontend Testing
1. Visit your frontend URL
2. The map should load showing NYC
3. Try making a fare prediction
4. Should successfully connect to backend API

### Full Integration Test
1. Open frontend in browser
2. Set pickup location (click on map)
3. Set dropoff location (click on map)
4. Fill in passenger count and other details
5. Click "Predict Fare"
6. Should get prediction result

---

## 🔧 Troubleshooting

### Backend Issues
- **Build fails**: Check if model files are properly committed with Git LFS
- **Memory errors**: Model is ~487MB, close to 512MB free tier limit
- **Service sleeps**: Free services sleep after 15 minutes of inactivity

### Frontend Issues
- **API connection fails**: Check REACT_APP_API_URL is set correctly
- **CORS errors**: Backend CORS is set to allow all origins
- **Build fails**: Check Node.js dependencies in build logs

### Common Issues
- **First load slow**: Free services need to "wake up" (30-60 seconds)
- **Timeout errors**: Free tier has connection limits
- **Map not loading**: Check environment variables are set

---

## 💰 Free Tier Limitations

### Backend (Web Service)
- ✅ 512MB RAM
- ✅ Sleeps after 15 minutes of inactivity
- ✅ 750 hours/month (enough for personal use)
- ⚠️ Model file (~487MB) is close to memory limit

### Frontend (Static Site)
- ✅ 100GB bandwidth/month
- ✅ Global CDN
- ✅ Always available (no sleeping)
- ✅ Custom domains supported

## 🎯 Success Criteria
- ✅ Backend API responding at `/health` endpoint
- ✅ API documentation accessible at `/docs`
- ✅ Frontend loads with interactive map
- ✅ Frontend can make successful predictions
- ✅ No CORS errors in browser console

## 🔗 Useful Links
- Backend health check: `https://your-backend-url.onrender.com/health`
- Backend API docs: `https://your-backend-url.onrender.com/docs`
- Frontend app: `https://your-frontend-url.onrender.com`

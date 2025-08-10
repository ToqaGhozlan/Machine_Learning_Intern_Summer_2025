# 🚀 Frontend Deployment Guide - Render Free Tier

## Prerequisites
- Backend must be deployed first
- Get your backend URL from the previous deployment

## Quick Deployment Steps

### 1. Create New Static Site
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Static Site"
3. Connect your GitHub repository: `moazmo/taxi-fare-prediction-ml`

### 2. Configure Site Settings
- **Name**: `taxi-fare-frontend` (or any name you prefer)
- **Region**: `Oregon (US West)` (same as backend)
- **Branch**: `main`
- **Root Directory**: `frontend`

### 3. Build Settings
```bash
# Build Command:
npm ci && npm run build

# Publish Directory:
build
```

### 4. Environment Variables
Add these in Render dashboard:
```
NODE_ENV=production
REACT_APP_API_URL=https://YOUR-BACKEND-URL.onrender.com
REACT_APP_ENVIRONMENT=production
REACT_APP_DEFAULT_LAT=40.7589
REACT_APP_DEFAULT_LNG=-73.9851
REACT_APP_DEFAULT_ZOOM=12
GENERATE_SOURCEMAP=false
CI=false
```

**⚠️ IMPORTANT**: Replace `YOUR-BACKEND-URL` with your actual backend service URL from step 1.

### 5. Custom Headers (Optional)
Add these for better performance:
```
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
```

### 6. After Deployment
- Your app will be available at: `https://your-site-name.onrender.com`
- Test that it can connect to your backend API
- The map should load with NYC coordinates

## Configuration Examples

### If your backend URL is: `https://taxi-api-xyz.onrender.com`
Set environment variable:
```
REACT_APP_API_URL=https://taxi-api-xyz.onrender.com
```

### Testing the Connection
1. Open your frontend URL
2. Try making a prediction
3. Check browser console for API connection issues

## Troubleshooting
- **CORS Errors**: Make sure backend CORS is configured correctly
- **API Connection Failed**: Verify the REACT_APP_API_URL is correct
- **Build Failures**: Check for dependency conflicts in build logs
- **Map Not Loading**: Verify environment variables are set correctly

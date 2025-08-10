# 🚀 Deployment Guide - Taxi Fare Prediction

This guide covers different deployment options for the Taxi Fare Prediction application.

## 📋 Prerequisites

### System Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Software Requirements
- **Python**: 3.7 or higher
- **Node.js**: 14 or higher
- **npm**: 6 or higher
- **Docker**: 20.10+ (for containerized deployment)

## 🏠 Local Development Deployment

### Quick Start

1. **Clone and Setup**
   ```bash
   cd Task_5_2
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Start Services**
   ```bash
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

3. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Manual Setup

#### Backend Setup
```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env file as needed

# Test model
python test_model.py

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env file as needed

# Start development server
npm start
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and Start**
   ```bash
   docker-compose up -d
   ```

2. **View Logs**
   ```bash
   docker-compose logs -f
   ```

3. **Stop Services**
   ```bash
   docker-compose down
   ```

### Individual Container Deployment

#### Backend Container
```bash
cd backend
docker build -t taxi-prediction-backend .
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name taxi-backend \
  taxi-prediction-backend
```

#### Frontend Container
```bash
cd frontend
docker build -t taxi-prediction-frontend .
docker run -d -p 3000:3000 \
  --name taxi-frontend \
  taxi-prediction-frontend
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using AWS ECS

1. **Create ECR Repositories**
   ```bash
   aws ecr create-repository --repository-name taxi-prediction-backend
   aws ecr create-repository --repository-name taxi-prediction-frontend
   ```

2. **Build and Push Images**
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Build and push backend
   docker build -t taxi-prediction-backend backend/
   docker tag taxi-prediction-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/taxi-prediction-backend:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/taxi-prediction-backend:latest

   # Build and push frontend
   docker build -t taxi-prediction-frontend frontend/
   docker tag taxi-prediction-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/taxi-prediction-frontend:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/taxi-prediction-frontend:latest
   ```

3. **Create ECS Task Definition**
   ```json
   {
     "family": "taxi-prediction",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "containerDefinitions": [
       {
         "name": "backend",
         "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/taxi-prediction-backend:latest",
         "portMappings": [{"containerPort": 8000}],
         "environment": [
           {"name": "ENVIRONMENT", "value": "production"}
         ]
       }
     ]
   }
   ```

#### Using AWS Lambda (Backend Only)

1. **Install Serverless Framework**
   ```bash
   npm install -g serverless
   ```

2. **Create Serverless Configuration**
   ```yaml
   # serverless.yml
   service: taxi-prediction-api
   
   provider:
     name: aws
     runtime: python3.9
     region: us-east-1
   
   functions:
     predict:
       handler: lambda_handler.predict
       events:
         - http:
             path: api/v1/predict
             method: post
   ```

### Google Cloud Platform

#### Using Cloud Run

1. **Build and Deploy Backend**
   ```bash
   cd backend
   gcloud builds submit --tag gcr.io/PROJECT-ID/taxi-prediction-backend
   gcloud run deploy taxi-backend \
     --image gcr.io/PROJECT-ID/taxi-prediction-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

2. **Build and Deploy Frontend**
   ```bash
   cd frontend
   gcloud builds submit --tag gcr.io/PROJECT-ID/taxi-prediction-frontend
   gcloud run deploy taxi-frontend \
     --image gcr.io/PROJECT-ID/taxi-prediction-frontend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Heroku Deployment

#### Backend Deployment

1. **Create Heroku App**
   ```bash
   cd backend
   heroku create taxi-prediction-backend
   ```

2. **Configure Environment**
   ```bash
   heroku config:set ENVIRONMENT=production
   heroku config:set MODEL_PATH=./models/best_taxi_fare_model.pkl
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

#### Frontend Deployment

1. **Create Heroku App**
   ```bash
   cd frontend
   heroku create taxi-prediction-frontend
   ```

2. **Configure Build**
   ```bash
   heroku buildpacks:set heroku/nodejs
   heroku config:set REACT_APP_API_URL=https://taxi-prediction-backend.herokuapp.com
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

## 🔧 Production Configuration

### Environment Variables

#### Backend (.env)
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
HOST=0.0.0.0
PORT=8000
WORKERS=4
CORS_ORIGINS=https://yourdomain.com
MODEL_PATH=./models/best_taxi_fare_model.pkl
PROCESSOR_PATH=./models/feature_processor.pkl
METADATA_PATH=./models/final_model_metadata.json
```

#### Frontend (.env)
```bash
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_ENABLE_ANALYTICS=true
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Health checks
    location /health {
        proxy_pass http://localhost:8000/health;
    }
}
```

### SSL/HTTPS Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring and Logging

### Application Monitoring

1. **Health Checks**
   - Backend: `GET /health`
   - Detailed: `GET /health/detailed`
   - Metrics: `GET /metrics`

2. **Log Aggregation**
   ```bash
   # View logs
   docker-compose logs -f backend
   docker-compose logs -f frontend
   
   # Log rotation
   sudo logrotate -f /etc/logrotate.conf
   ```

3. **Performance Monitoring**
   ```bash
   # System metrics
   htop
   iostat -x 1
   
   # Application metrics
   curl http://localhost:8000/metrics
   ```

### Error Tracking

1. **Sentry Integration** (Optional)
   ```bash
   pip install sentry-sdk[fastapi]
   npm install @sentry/react
   ```

2. **Custom Logging**
   ```python
   # Backend logging
   import logging
   logger = logging.getLogger(__name__)
   logger.error("Custom error message")
   ```

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Enable HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Use environment variables for secrets
- [ ] Enable security headers
- [ ] Regular security updates
- [ ] Monitor for vulnerabilities
- [ ] Backup model files
- [ ] Set up firewall rules
- [ ] Use non-root containers

### Security Headers

```python
# FastAPI security headers
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])
app.add_middleware(HTTPSRedirectMiddleware)
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```bash
   # Check model files exist
   ls -la backend/models/
   
   # Test model loading
   cd backend && python test_model.py
   ```

2. **CORS Issues**
   ```bash
   # Check CORS configuration
   grep CORS_ORIGINS backend/.env
   ```

3. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   
   # Increase container memory
   docker run -m 4g ...
   ```

4. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Kill process
   sudo kill -9 <PID>
   ```

### Performance Optimization

1. **Backend Optimization**
   - Use multiple workers: `--workers 4`
   - Enable model caching
   - Optimize model loading
   - Use connection pooling

2. **Frontend Optimization**
   - Enable gzip compression
   - Use CDN for static assets
   - Implement code splitting
   - Optimize bundle size

## 📞 Support

For deployment issues:
1. Check application logs
2. Verify environment configuration
3. Test health endpoints
4. Review system resources
5. Consult troubleshooting section

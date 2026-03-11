# 🚀 Deployment Guide

## Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train_models.py
```

Expected output:
- Training will take 2-5 minutes
- Models saved to `models/` directory
- Confusion matrix and ROC curve plots generated
- Best model accuracy: 90%+

### 3. Run Application
```bash
python app.py
```

Visit: `http://localhost:5000`

---

## 🌐 Cloud Deployment

### Option 1: Render (Recommended - Free Tier)

1. **Create Account**: https://render.com

2. **Connect Repository**:
   - Push code to GitHub
   - Connect GitHub repo to Render
   - Render will auto-detect `render.yaml`

3. **Deploy**:
   - Click "Create Web Service"
   - Select your repository
   - Render will automatically:
     - Install dependencies
     - Train models
     - Start the application

4. **Access**:
   - Your app will be live at: `https://your-app.onrender.com`

**Pros**: Free tier, auto-deploy on git push, SSL included
**Cons**: Cold starts on free tier (app sleeps after 15 min inactivity)

---

### Option 2: Railway

1. **Install Railway CLI**:
```bash
npm install -g @railway/cli
```

2. **Login**:
```bash
railway login
```

3. **Initialize Project**:
```bash
railway init
```

4. **Deploy**:
```bash
railway up
```

5. **Get URL**:
```bash
railway domain
```

**Pros**: Fast deployment, good free tier, automatic HTTPS
**Cons**: Requires credit card for free tier

---

### Option 3: Docker (Any Platform)

1. **Build Image**:
```bash
docker build -t thyroid-prediction .
```

2. **Run Container**:
```bash
docker run -p 5000:5000 thyroid-prediction
```

3. **Deploy to Cloud**:
   - **AWS ECS**: Push to ECR, deploy to ECS
   - **Google Cloud Run**: Push to GCR, deploy to Cloud Run
   - **Azure Container Instances**: Push to ACR, deploy to ACI

**Pros**: Portable, consistent environment
**Cons**: Requires Docker knowledge

---

### Option 4: Heroku

1. **Install Heroku CLI**:
```bash
# macOS
brew tap heroku/brew && brew install heroku

# Windows
# Download from: https://devcenter.heroku.com/articles/heroku-cli
```

2. **Login**:
```bash
heroku login
```

3. **Create App**:
```bash
heroku create your-app-name
```

4. **Deploy**:
```bash
git push heroku main
```

5. **Open App**:
```bash
heroku open
```

**Pros**: Simple deployment, good documentation
**Cons**: No free tier anymore (requires paid plan)

---

### Option 5: AWS EC2

1. **Launch EC2 Instance**:
   - Ubuntu 22.04 LTS
   - t2.medium or larger (for model training)
   - Open port 5000 in security group

2. **SSH into Instance**:
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install Dependencies**:
```bash
sudo apt update
sudo apt install python3-pip python3-venv -y
```

4. **Clone Repository**:
```bash
git clone your-repo-url
cd your-repo
```

5. **Setup Virtual Environment**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

6. **Train Models**:
```bash
python train_models.py
```

7. **Run with Gunicorn**:
```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 4
```

8. **Setup Nginx (Optional)**:
```bash
sudo apt install nginx -y
# Configure nginx as reverse proxy
```

9. **Setup Systemd Service** (Keep app running):
```bash
sudo nano /etc/systemd/system/thyroid-prediction.service
```

Add:
```ini
[Unit]
Description=Thyroid Prediction Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/your-repo
Environment="PATH=/home/ubuntu/your-repo/venv/bin"
ExecStart=/home/ubuntu/your-repo/venv/bin/gunicorn app:app --bind 0.0.0.0:5000 --workers 4

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable thyroid-prediction
sudo systemctl start thyroid-prediction
```

**Pros**: Full control, scalable
**Cons**: More setup, requires AWS knowledge

---

## 🔧 Environment Variables

Set these in your deployment platform:

```bash
# Required
FLASK_ENV=production

# Optional
PORT=5000                    # Port to run on
MAX_CONTENT_LENGTH=16777216  # Max upload size (16MB)
WORKERS=4                    # Number of Gunicorn workers
```

---

## 📊 Post-Deployment Checklist

- [ ] Models trained successfully
- [ ] Application starts without errors
- [ ] Image upload works
- [ ] Numerical prediction works
- [ ] Results display correctly
- [ ] Error handling works
- [ ] HTTPS enabled (production)
- [ ] Monitoring setup (optional)

---

## 🔍 Troubleshooting

### Issue: Models not found
**Solution**: Ensure `train_models.py` ran successfully during build

### Issue: Out of memory during training
**Solution**: Use instance with at least 2GB RAM

### Issue: Slow predictions
**Solution**: 
- Increase number of workers
- Use instance with more CPU cores
- Consider GPU instance for image predictions

### Issue: Port already in use
**Solution**: Change PORT environment variable

### Issue: File upload fails
**Solution**: Check MAX_CONTENT_LENGTH setting

---

## 📈 Scaling Recommendations

### Small Scale (< 100 users/day)
- Render Free Tier or Railway
- 1 worker, 512MB RAM

### Medium Scale (100-1000 users/day)
- Render Standard or AWS t3.small
- 2-4 workers, 2GB RAM

### Large Scale (1000+ users/day)
- AWS t3.medium or larger
- 4-8 workers, 4GB+ RAM
- Load balancer
- Auto-scaling group
- Redis for caching

---

## 🔒 Security Recommendations

1. **HTTPS**: Always use HTTPS in production
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **Input Validation**: Already implemented
4. **File Size Limits**: Already set to 16MB
5. **CORS**: Configure if needed for API access
6. **API Keys**: Add authentication for production API

---

## 📊 Monitoring

### Basic Monitoring
```python
# Add to app.py
import logging
logging.basicConfig(level=logging.INFO)
```

### Advanced Monitoring
- **Sentry**: Error tracking
- **New Relic**: Performance monitoring
- **CloudWatch**: AWS monitoring
- **Prometheus + Grafana**: Custom metrics

---

## 🎯 Performance Optimization

1. **Model Loading**: Models loaded once at startup
2. **Caching**: Consider Redis for frequent predictions
3. **Async Processing**: Use Celery for batch predictions
4. **CDN**: Use CDN for static assets
5. **Database**: Add PostgreSQL for prediction history

---

## 📝 Maintenance

### Regular Tasks
- Monitor disk space (logs, uploads)
- Update dependencies monthly
- Retrain models with new data
- Backup models and data
- Review error logs

### Model Updates
```bash
# Retrain with new data
python train_models.py

# Restart application
# Render/Railway: Auto-restart on git push
# EC2: sudo systemctl restart thyroid-prediction
```

---

## 🆘 Support

For deployment issues:
1. Check application logs
2. Verify all dependencies installed
3. Ensure models directory exists
4. Check file permissions
5. Review platform-specific documentation

---

**Ready to deploy? Choose your platform and follow the steps above!** 🚀

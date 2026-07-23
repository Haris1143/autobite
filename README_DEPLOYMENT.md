# AutoBite Deployment Guide

## 🚀 Deploy to Vercel (Recommended)

### Prerequisites
1. GitHub account
2. Vercel account (free at vercel.com)
3. Git installed

### Step 1: Initialize Git Repository
```powershell
cd e:\autobite
git init
git add .
git commit -m "Initial commit: AutoBite AI Food Recommender"
```

### Step 2: Push to GitHub
```powershell
# Create a new repo on github.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/autobite.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy to Vercel
```powershell
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

Follow the prompts:
- Link to your GitHub account
- Select your `autobite` repository
- Framework: "Other" (FastAPI is not pre-configured)
- Output Directory: `api`

### Step 4: Configure Environment (if needed)
In Vercel dashboard:
1. Go to Settings → Environment Variables
2. Add any secrets your app needs

**Your app will be live at: `https://your-project.vercel.app`**

---

## ⚠️ Known Limitations on Vercel

- **Cold Starts**: First request takes 10-15 seconds (models load on startup)
- **Timeout**: Requests timeout after 10-15 seconds
- **Model Size**: Keep total under 50MB (your models are small, so you're OK ✅)
- **Memory**: Limited to 3008MB per request

### Solution if Too Slow
Use **Railway, Render, or Fly.io** instead (see below)

---

## 🐳 Deploy with Docker (Best Performance)

### Option A: Railway (Easiest)

1. **Install Railway CLI**
   ```powershell
   npm install -g @railway/cli
   ```

2. **Create Dockerfile** (already in project)
   ```powershell
   # Run this in e:\autobite
   railway init
   railway up
   ```

3. **Get your URL**
   ```
   Your app is live at: https://xxx.railway.app
   ```

### Option B: Render

1. Go to https://render.com
2. Create new "Web Service"
3. Connect your GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Deploy!

### Option C: Fly.io

```powershell
npm install -g flyctl
flyctl auth login
flyctl launch
flyctl deploy
```

---

## 📦 What's in This Repo

```
autobite/
├── app.py              # FastAPI backend (local development)
├── api/index.py        # Vercel serverless function
├── index.html          # Frontend UI
├── requirements.txt    # Python dependencies
├── vercel.json         # Vercel configuration
├── models/             # ML models
├── data/               # Training data & CSVs
└── README_DEPLOYMENT.md
```

---

## 🔧 Local Development

```powershell
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Visit http://localhost:8000
```

---

## 💡 Comparison Table

| Platform | Cost | Cold Start | Best For |
|----------|------|-----------|----------|
| **Vercel** | Free | 10-15s | Simple APIs |
| **Railway** | Free tier + pay-as-you-go | ~1s | Production apps |
| **Render** | Free tier | ~2-5s | Full-stack apps |
| **Fly.io** | Free tier | ~1s | Global distribution |

---

## 📱 Frontend Deployment

Frontend is automatically served from the same backend. Just access:
- **Vercel**: https://your-project.vercel.app
- **Railway**: https://xxx.railway.app
- **Render**: https://xxx.onrender.com

---

## 🐛 Troubleshooting

**Models too slow to load?**
- Use Railway/Render instead of Vercel
- Compress models with `joblib` compression

**404 on `/`?**
- Make sure `index.html` is in project root
- Check vercel.json routes

**CORS errors?**
- CORS middleware is enabled for all origins
- Update in `api/index.py` if needed

---

## 🎯 Next Steps

1. Choose a platform (Vercel for simple, Railway for performance)
2. Push code to GitHub
3. Deploy and get your public URL
4. Share with friends! 🚀


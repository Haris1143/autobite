# 🚀 AutoBite - Quick Deploy to Vercel

## ✅ 3 Steps to Production

### Step 1: Initialize Git
```powershell
cd e:\autobite
git init
git add .
git commit -m "AutoBite AI Food Recommender"
```

### Step 2: Push to GitHub
1. Create repo at https://github.com/new
2. Run:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/autobite.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy to Vercel
1. Go to https://vercel.com/new
2. Import from GitHub
3. Select your `autobite` repo
4. Click "Deploy"

**That's it! Your app will be live in ~2 minutes** 🎉

---

## 🌍 Your Live URL
```
https://autobite-xxxxx.vercel.app
```

---

## ⚡ Performance Tips

If requests are slow (10-15s first request):
- **Use Railway instead** (better for ML models)
  1. Push to GitHub (same as above)
  2. Go to https://railway.app
  3. Create new project → GitHub repo → Deploy

---

## 📝 Update & Redeploy

After making changes:
```powershell
git add .
git commit -m "Your changes"
git push origin main
```

**Vercel auto-deploys on push!** ✅

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| 404 errors | Clear browser cache + hard refresh |
| Slow responses | Use Railway instead of Vercel |
| Models not found | Check GitHub has `/models` folder |
| CORS errors | Already fixed in code |

---

## 📚 More Options

- **Railway** (Best for Python ML): https://railway.app
- **Render** (Free tier): https://render.com
- **Fly.io** (Global): https://fly.io
- **DigitalOcean** (Cheap): https://digitalocean.com

See `README_DEPLOYMENT.md` for detailed guides.

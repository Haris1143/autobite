"""
api/index.py — AutoBite Vercel Serverless Entry Point
══════════════════════════════════════════════════════
Same logic as app.py but paths resolve relative to repo root for Vercel.

MODELS USED
───────────
/predict_text      → intent_pipeline.joblib, collab_knn.pkl, user_profiles.pkl
/recommend_reorder → reorder_knn.pkl, order_profiles.pkl, collab_knn.pkl,
                     user_profiles.pkl, item_emb.joblib, popularity.joblib
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib, os
from typing import List, Optional
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

# ── Load models ───────────────────────────────────────────────────────────────
intent_model   = joblib.load(MODEL_DIR / "intent_pipeline.joblib")
reorder_knn    = joblib.load(MODEL_DIR / "reorder_knn.pkl")
order_profiles = joblib.load(MODEL_DIR / "order_profiles.pkl")
collab_knn     = joblib.load(MODEL_DIR / "collab_knn.pkl")
user_profiles  = joblib.load(MODEL_DIR / "user_profiles.pkl")
item_emb       = joblib.load(MODEL_DIR / "item_emb.joblib")
popularity_df  = joblib.load(MODEL_DIR / "popularity.joblib")

products_df   = pd.read_csv(DATA_DIR / "products.csv")
food_prefs_df = pd.read_csv(DATA_DIR / "food_preferences.csv")
orders_df     = pd.read_csv(DATA_DIR / "order_history.csv")
user_pref_df  = pd.read_csv(DATA_DIR / "user_preferences.csv")

app = FastAPI(title="AutoBite API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    text: str
    user_id: int
    dislikes: List[str] = []

class ReorderRequest(BaseModel):
    user_id: int
    dislikes: List[str] = []
    time_of_day: Optional[str] = None
    top_n: int = 5

# ── Helpers (identical to app.py) ─────────────────────────────────────────────
def _filter_and_rank(df, dislikes, top_n):
    if dislikes:
        df = df[~df["category"].isin(dislikes)]
    cols = [c for c in ["food_item", "category", "price", "score"] if c in df.columns]
    return df.sort_values("score", ascending=False).head(top_n)[cols].to_dict(orient="records")

def _collab_candidates(user_id):
    if user_id not in user_profiles.index:
        return pd.DataFrame()
    _, idxs = collab_knn.kneighbors(user_profiles.loc[[user_id]], n_neighbors=10)
    sim = user_profiles.iloc[idxs[0]].index.tolist()
    recs = food_prefs_df[food_prefs_df["user_id"].isin(sim)]
    recs = recs.groupby("product_id")["preference_score"].mean().reset_index()
    recs = recs.merge(products_df, on="product_id").rename(columns={"preference_score": "score"})
    return recs

def _order_history_candidates(user_id):
    if user_id not in order_profiles.index:
        return pd.DataFrame()
    _, idxs = reorder_knn.kneighbors(order_profiles.loc[[user_id]], n_neighbors=10)
    sim = order_profiles.iloc[idxs[0]].index.tolist()
    sim_orders = orders_df[orders_df["user_id"].isin(sim)]
    recs = sim_orders.groupby("product_id").agg(
        score=("rating", "mean"), order_count=("order_id", "count")).reset_index()
    recs["score"] = recs["score"] * np.log1p(recs["order_count"])
    return recs.merge(products_df, on="product_id")

def _user_rated_candidates(user_id, time_of_day=None):
    uid_str = f"u{user_id:03d}"
    sub = user_pref_df[user_pref_df["user_id"] == uid_str].copy()
    if time_of_day:
        sub = sub[sub["time_of_day"] == time_of_day]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.rename(columns={"rating": "score"})
    sub = sub.merge(products_df[["food_item","category","price"]].drop_duplicates("food_item"),
                    on="food_item", how="left")
    return sub[["food_item","category","price","score"]].dropna()

def _norm(df, col="score"):
    if df.empty or col not in df.columns:
        return df
    mn, mx = df[col].min(), df[col].max()
    df[col] = (df[col] - mn) / (mx - mn) if mx > mn else 1.0
    return df

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    p = BASE_DIR / "index.html"
    return FileResponse(str(p)) if p.exists() else {"message": "AutoBite API v2"}

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AutoBite API", "version": "2.0"}

@app.post("/predict_text")
def predict_text(req: PredictRequest):
    intent = intent_model.predict([req.text])[0]
    collab = _collab_candidates(req.user_id)
    if collab.empty:
        pop = popularity_df[["product_id","food_item","category","price","popularity_score"]].rename(
            columns={"popularity_score":"score"})
        return {"intent": intent, "source": "popularity_fallback",
                "recommendations": _filter_and_rank(pop, req.dislikes, 5)}
    return {"intent": intent, "source": "collaborative",
            "recommendations": _filter_and_rank(collab, req.dislikes, 5)}

@app.post("/recommend_reorder")
def recommend_reorder(req: ReorderRequest):
    o = _norm(_order_history_candidates(req.user_id).copy())
    c = _norm(_collab_candidates(req.user_id).copy())
    r = _norm(_user_rated_candidates(req.user_id, req.time_of_day).copy())

    if o.empty and c.empty and r.empty:
        pop = popularity_df[["product_id","food_item","category","price","popularity_score"]].rename(
            columns={"popularity_score":"score"})
        return {"user_id": req.user_id, "source": "popularity_fallback",
                "recommendations": _filter_and_rank(pop, req.dislikes, req.top_n)}

    def extract(df, w, lbl):
        if df.empty or "food_item" not in df.columns:
            return {}
        return {row["food_item"]: {"category": row.get("category",""),
                                    "price": row.get("price", 0.0),
                                    lbl: row["score"] * w}
                for _, row in df.iterrows()}

    o_m = extract(o, 0.45, "order_score")
    c_m = extract(c, 0.30, "collab_score")
    r_m = extract(r, 0.25, "rating_score")
    all_items = set(o_m) | set(c_m) | set(r_m)

    merged = []
    for item in all_items:
        oi, ci, ri = o_m.get(item,{}), c_m.get(item,{}), r_m.get(item,{})
        merged.append({
            "food_item": item,
            "category":  oi.get("category") or ci.get("category") or ri.get("category",""),
            "price":     oi.get("price")    or ci.get("price")    or ri.get("price", 0.0),
            "score":     oi.get("order_score",0)+ci.get("collab_score",0)+ri.get("rating_score",0),
        })

    mdf = pd.DataFrame(merged)
    if req.time_of_day and not r.empty:
        tod = set(user_pref_df[(user_pref_df["time_of_day"]==req.time_of_day) &
                               (user_pref_df["user_id"]==f"u{req.user_id:03d}")]["food_item"])
        if tod:
            mdf["score"] = mdf.apply(lambda row: row["score"]+0.2
                                     if row["food_item"] in tod else row["score"], axis=1)

    return {"user_id": req.user_id,
            "source": "hybrid (order_history + collaborative + user_ratings)",
            "time_of_day": req.time_of_day,
            "recommendations": _filter_and_rank(mdf, req.dislikes, req.top_n)}

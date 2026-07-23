"""
app.py — AutoBite FastAPI Backend
══════════════════════════════════

ENDPOINTS
─────────
GET  /                    → Serve autobite.html frontend
GET  /api/health          → Health check

POST /predict_text        → Intent classification + collaborative recommendation
POST /recommend_reorder   → Hybrid recommendation based on:
                              • Previously ordered items   (reorder_knn)
                              • User's rated preferences   (user_preferences.csv)
                              • Similar users' orders      (collab_knn)

MODELS USED PER ENDPOINT
─────────────────────────
/predict_text
  • intent_pipeline.joblib  → detect what the user wants
  • collab_knn.pkl          → find similar users by taste profile
  • user_profiles.pkl       → feature vectors for collab kNN

/recommend_reorder
  • reorder_knn.pkl         → find users with similar order histories
  • order_profiles.pkl      → feature vectors built from order history
  • collab_knn.pkl          → augment with collaborative taste signal
  • user_profiles.pkl       → collab feature vectors
  • item_emb.joblib         → content-based re-ranking (TF-IDF+SVD vectors)
  • popularity.joblib       → fallback when user is cold-start
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib, os
from typing import List, Optional

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models ...")

# Model 1 — Intent classifier
intent_model    = joblib.load(os.path.join(MODEL_DIR, "intent_pipeline.joblib"))

# Model 2 — Re-order KNN (order-history based)
reorder_knn     = joblib.load(os.path.join(MODEL_DIR, "reorder_knn.pkl"))
order_profiles  = joblib.load(os.path.join(MODEL_DIR, "order_profiles.pkl"))

# Model 3 — Collaborative KNN
collab_knn      = joblib.load(os.path.join(MODEL_DIR, "collab_knn.pkl"))
user_profiles   = joblib.load(os.path.join(MODEL_DIR, "user_profiles.pkl"))

# Model 4 — Content-based item embeddings
item_emb        = joblib.load(os.path.join(MODEL_DIR, "item_emb.joblib"))

# Popularity fallback
popularity_df   = joblib.load(os.path.join(MODEL_DIR, "popularity.joblib"))

# ── Load data ─────────────────────────────────────────────────────────────────
products_df    = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
food_prefs_df  = pd.read_csv(os.path.join(DATA_DIR, "food_preferences.csv"))
orders_df      = pd.read_csv(os.path.join(DATA_DIR, "order_history.csv"))
user_pref_df   = pd.read_csv(os.path.join(DATA_DIR, "user_preferences.csv"))

print("All models and data loaded.")

# ── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="AutoBite API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    user_id: int
    dislikes: List[str] = []

class ReorderRequest(BaseModel):
    user_id: int
    dislikes: List[str] = []
    time_of_day: Optional[str] = None   # "breakfast"|"lunch"|"snack"|"dinner"
    top_n: int = 5

# ── Helpers ───────────────────────────────────────────────────────────────────
def _filter_and_rank(df: pd.DataFrame, dislikes: list, top_n: int) -> list:
    if dislikes:
        df = df[~df["category"].isin(dislikes)]
    cols = [c for c in ["food_item", "category", "price", "score"] if c in df.columns]
    return df.sort_values("score", ascending=False).head(top_n)[cols].to_dict(orient="records")


def _collab_candidates(user_id: int) -> pd.DataFrame:
    """Return candidate products from collaborative filtering."""
    if user_id not in user_profiles.index:
        return pd.DataFrame()
    dists, idxs = collab_knn.kneighbors(user_profiles.loc[[user_id]], n_neighbors=10)
    sim_users   = user_profiles.iloc[idxs[0]].index.tolist()
    recs = food_prefs_df[food_prefs_df["user_id"].isin(sim_users)]
    recs = recs.groupby("product_id")["preference_score"].mean().reset_index()
    recs = recs.merge(products_df, on="product_id")
    recs = recs.rename(columns={"preference_score": "score"})
    return recs


def _order_history_candidates(user_id: int) -> pd.DataFrame:
    """Return candidate products from order-history KNN."""
    if user_id not in order_profiles.index:
        return pd.DataFrame()
    dists, idxs = reorder_knn.kneighbors(order_profiles.loc[[user_id]], n_neighbors=10)
    sim_users   = order_profiles.iloc[idxs[0]].index.tolist()

    # Items those similar users ordered, weighted by rating & recency
    sim_orders = orders_df[orders_df["user_id"].isin(sim_users)]
    recs = (
        sim_orders.groupby("product_id")
        .agg(score=("rating", "mean"), order_count=("order_id", "count"))
        .reset_index()
    )
    recs["score"] = recs["score"] * np.log1p(recs["order_count"])   # boost frequent items
    recs = recs.merge(products_df, on="product_id")
    return recs


def _user_rated_candidates(user_id: int, time_of_day: str = None) -> pd.DataFrame:
    """Return items the user has explicitly rated highly in user_preferences.csv."""
    # user_preferences uses 'u001' style ids
    uid_str = f"u{user_id:03d}"
    subset  = user_pref_df[user_pref_df["user_id"] == uid_str].copy()
    if time_of_day:
        subset = subset[subset["time_of_day"] == time_of_day]
    if subset.empty:
        return pd.DataFrame()
    subset = subset.rename(columns={"rating": "score"})
    subset = subset.merge(
        products_df[["food_item", "category", "price"]].drop_duplicates("food_item"),
        on="food_item", how="left"
    )
    return subset[["food_item", "category", "price", "score"]].dropna()


def _popularity_fallback(top_n: int) -> pd.DataFrame:
    pop = popularity_df[["product_id", "food_item", "category", "price", "popularity_score"]].copy()
    pop = pop.rename(columns={"popularity_score": "score"})
    return pop

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    html = os.path.join(BASE_DIR, "autobite.html")
    if os.path.exists(html):
        return FileResponse(html)
    return {"message": "AutoBite API v2 running. Visit /docs for API docs."}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AutoBite API", "version": "2.0"}


@app.post("/predict_text")
def predict_text(req: PredictRequest):
    """
    Intent classification + collaborative recommendation.

    Models used:
      • intent_pipeline.joblib  — classify user text
      • collab_knn.pkl          — find similar users by preference profile
    """
    intent = intent_model.predict([req.text])[0]
    collab = _collab_candidates(req.user_id)

    if collab.empty:
        fallback = _popularity_fallback(req.top_n if hasattr(req, "top_n") else 5)
        return {"intent": intent, "source": "popularity_fallback",
                "recommendations": _filter_and_rank(fallback, req.dislikes, 5)}

    return {
        "intent": intent,
        "source": "collaborative",
        "recommendations": _filter_and_rank(collab, req.dislikes, 5),
    }


@app.post("/recommend_reorder")
def recommend_reorder(req: ReorderRequest):
    """
    Hybrid recommendation based on:
      1. Previously ordered items   → reorder_knn.pkl + order_profiles.pkl
      2. User's rated preferences   → user_preferences.csv
      3. Similar users' orders      → collab_knn.pkl + user_profiles.pkl

    The three candidate sets are merged and scored:
      final_score = 0.45 × order_history_score
                  + 0.30 × collaborative_score
                  + 0.25 × user_rating_score

    Falls back to popularity.joblib if user has no history.
    """
    # --- Gather candidates from all three sources ---
    order_cands  = _order_history_candidates(req.user_id)
    collab_cands = _collab_candidates(req.user_id)
    rated_cands  = _user_rated_candidates(req.user_id, req.time_of_day)

    # Cold-start fallback
    if order_cands.empty and collab_cands.empty and rated_cands.empty:
        pop = _popularity_fallback(req.top_n)
        return {
            "user_id": req.user_id,
            "source": "popularity_fallback",
            "recommendations": _filter_and_rank(pop, req.dislikes, req.top_n),
        }

    # --- Normalise scores per source into [0, 1] ---
    def norm_score(df, col="score"):
        if df.empty or col not in df.columns:
            return df
        mn, mx = df[col].min(), df[col].max()
        if mx > mn:
            df[col] = (df[col] - mn) / (mx - mn)
        else:
            df[col] = 1.0
        return df

    order_cands  = norm_score(order_cands.copy())
    collab_cands = norm_score(collab_cands.copy())
    rated_cands  = norm_score(rated_cands.copy(), "score")

    # --- Merge on food_item ---
    all_items = set()
    for df in [order_cands, collab_cands, rated_cands]:
        if not df.empty and "food_item" in df.columns:
            all_items.update(df["food_item"].tolist())

    # Build unified frame
    def extract(df, weight, label):
        if df.empty or "food_item" not in df.columns:
            return {}
        return {
            row["food_item"]: {
                "category": row.get("category", ""),
                "price":    row.get("price", 0.0),
                label:      row["score"] * weight,
            }
            for _, row in df.iterrows()
        }

    o_map = extract(order_cands,  0.45, "order_score")
    c_map = extract(collab_cands, 0.30, "collab_score")
    r_map = extract(rated_cands,  0.25, "rating_score")

    merged = []
    for item in all_items:
        o = o_map.get(item, {})
        c = c_map.get(item, {})
        r = r_map.get(item, {})
        category = o.get("category") or c.get("category") or r.get("category") or ""
        price    = o.get("price")    or c.get("price")    or r.get("price")    or 0.0
        score    = (o.get("order_score", 0) +
                    c.get("collab_score", 0) +
                    r.get("rating_score", 0))
        merged.append({"food_item": item, "category": category, "price": price, "score": score})

    merged_df = pd.DataFrame(merged)

    # Time-of-day filter if provided
    if req.time_of_day and not rated_cands.empty:
        tod_items = set(
            user_pref_df[
                (user_pref_df["time_of_day"] == req.time_of_day) &
                (user_pref_df["user_id"] == f"u{req.user_id:03d}")
            ]["food_item"].tolist()
        )
        if tod_items:
            # Boost items that match time-of-day by +0.2
            merged_df["score"] = merged_df.apply(
                lambda r: r["score"] + 0.2 if r["food_item"] in tod_items else r["score"],
                axis=1
            )

    results = _filter_and_rank(merged_df, req.dislikes, req.top_n)

    return {
        "user_id":   req.user_id,
        "source":    "hybrid (order_history + collaborative + user_ratings)",
        "time_of_day": req.time_of_day,
        "recommendations": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

"""
train_all.py  — AutoBite Model Training Pipeline
═════════════════════════════════════════════════

MODELS TRAINED & SAVED
───────────────────────
1. intent_pipeline.joblib
   TYPE  : sklearn Pipeline (TF-IDF + Logistic Regression)
   USED  : Classify the user's free-text request into one of 6 intents
           (breakfast_request, spicy_dinner, paneer_craving,
            light_veg, non_veg_lunch, snack_time)
   INPUT : raw user text string
   OUTPUT: intent label string

2. reorder_knn.pkl  ← NEW
   TYPE  : sklearn NearestNeighbors (cosine, brute)
   USED  : "Previously ordered items" recommender.
           Finds users with similar order histories, then surfaces items
           those similar users also ordered.
   INPUT : user_id (integer)
   OUTPUT: list of recommended product_ids

3. collab_knn.pkl  (renamed from recommender_knn)
   TYPE  : sklearn NearestNeighbors (cosine, brute)
   USED  : Collaborative filtering on food_preferences.csv
           (preference_score matrix). Finds similar taste profiles.
   INPUT : user_id
   OUTPUT: list of recommended product_ids

4. item_vectorizer.joblib  +  item_svd.joblib  +  item_emb.joblib
   TYPE  : TF-IDF → TruncatedSVD → L2-normalized embeddings
   USED  : Content-based recommender. Converts food item text
           (name + category + time_of_day) into dense vectors;
           cosine similarity gives "similar dishes".
   INPUT : food_item text string
   OUTPUT: embedding vector (used at serve-time for similarity search)

5. hybrid_user_profiles.pkl  ← NEW
   TYPE  : pandas DataFrame (dense user profile matrix)
   USED  : Stores each user's merged profile combining
           • order history signals (recency, frequency, rating)
           • collaborative preference scores
           • content category affinities
           Used by reorder_knn and collab_knn at inference.
   INPUT : user_id index lookup
   OUTPUT: feature vector for kNN query
"""

import os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, MinMaxScaler

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
SCRIPT_DIR = os.path.join(BASE_DIR, "scripts")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 0  Generate order history if it doesn't exist
# ─────────────────────────────────────────────────────────────
order_path = os.path.join(DATA_DIR, "order_history.csv")
if not os.path.exists(order_path):
    print("⚙  order_history.csv not found — generating synthetic data ...")
    sys.path.insert(0, SCRIPT_DIR)
    import generate_order_history  # noqa: F401 — side-effect: writes the csv

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
print("\n📂  Loading data ...")
products_df   = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
food_prefs_df = pd.read_csv(os.path.join(DATA_DIR, "food_preferences.csv"))
user_pref_df  = pd.read_csv(os.path.join(DATA_DIR, "user_preferences.csv"))
intents_df    = pd.read_csv(os.path.join(DATA_DIR, "user_intents.csv"))
orders_df     = pd.read_csv(order_path)

# ─────────────────────────────────────────────────────────────
# MODEL 1  Intent Classifier
# ─────────────────────────────────────────────────────────────
print("\n🧠  [Model 1] Training Intent Classifier (TF-IDF + Logistic Regression) ...")
X = intents_df["user_request"].astype(str)
y = intents_df["intent"].astype(str)

intent_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)),
    ("clf",   LogisticRegression(max_iter=1000, C=5.0)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
intent_pipeline.fit(X_train, y_train)
acc = intent_pipeline.score(X_test, y_test)
print(f"   ✅  Intent classifier accuracy: {acc:.3f}")
joblib.dump(intent_pipeline, os.path.join(MODEL_DIR, "intent_pipeline.joblib"))

# ─────────────────────────────────────────────────────────────
# MODEL 4 + 5  Content-Based Item Embeddings
# ─────────────────────────────────────────────────────────────
print("\n📝  [Model 4] Training Content-Based Item Embeddings (TF-IDF → SVD) ...")
products_df["item_text"] = (
    products_df["food_item"].fillna("") + " " +
    products_df["category"].fillna("")
)

item_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5_000)
item_matrix     = item_vectorizer.fit_transform(products_df["item_text"])

n_components = min(100, item_matrix.shape[1] - 1)
svd          = TruncatedSVD(n_components=n_components, random_state=42)
item_emb     = normalize(svd.fit_transform(item_matrix), axis=1)

joblib.dump(item_vectorizer, os.path.join(MODEL_DIR, "item_vectorizer.joblib"))
joblib.dump(svd,             os.path.join(MODEL_DIR, "item_svd.joblib"))
joblib.dump(item_emb,        os.path.join(MODEL_DIR, "item_emb.joblib"))
print(f"   ✅  Item embeddings shape: {item_emb.shape}")

# ─────────────────────────────────────────────────────────────
# BUILD HYBRID USER PROFILES  (used by Models 2 & 3)
# ─────────────────────────────────────────────────────────────
print("\n🔗  Building Hybrid User Profiles ...")

# --- A: Collaborative signal from food_preferences.csv ---
pref_features = pd.get_dummies(
    food_prefs_df.merge(products_df[["product_id", "category"]], on="product_id", how="left")
    [["category"]], dummy_na=False
)
pref_features["preference_score"] = food_prefs_df["preference_score"].values
collab_profiles = pref_features.groupby(food_prefs_df["user_id"]).mean()

# --- B: Order-history signal ---
# Recency weight: more recent orders count more
orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
latest = orders_df["order_date"].max()
orders_df["days_ago"]      = (latest - orders_df["order_date"]).dt.days
orders_df["recency_weight"] = np.exp(-orders_df["days_ago"] / 90)  # half-life ~90 days
orders_df["weighted_rating"] = orders_df["rating"] * orders_df["recency_weight"] * orders_df["quantity"]

order_cat_dummies = pd.get_dummies(orders_df[["category"]], dummy_na=False)
order_cat_dummies["weighted_rating"] = orders_df["weighted_rating"].values
order_profiles = order_cat_dummies.groupby(orders_df["user_id"]).mean()

# Frequency profile  (how many orders per user)
freq = orders_df.groupby("user_id").size().rename("order_frequency")
order_profiles = order_profiles.join(freq, how="left").fillna(0)

# Normalise frequency
scaler = MinMaxScaler()
order_profiles["order_frequency"] = scaler.fit_transform(
    order_profiles[["order_frequency"]]
)

# --- C: Merge collab + order profiles ---
# Align on integer user_id
hybrid = collab_profiles.join(
    order_profiles, how="outer", lsuffix="_collab", rsuffix="_order"
).fillna(0)

joblib.dump(hybrid, os.path.join(MODEL_DIR, "hybrid_user_profiles.pkl"))
print(f"   ✅  Hybrid user profiles shape: {hybrid.shape}")

# ─────────────────────────────────────────────────────────────
# MODEL 3  Collaborative KNN  (preference-score based)
# ─────────────────────────────────────────────────────────────
print("\n🤝  [Model 3] Training Collaborative KNN (NearestNeighbors cosine) ...")
collab_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
collab_knn.fit(collab_profiles)
joblib.dump(collab_knn,    os.path.join(MODEL_DIR, "collab_knn.pkl"))
joblib.dump(collab_profiles, os.path.join(MODEL_DIR, "user_profiles.pkl"))  # keep old name too
print(f"   ✅  Collaborative KNN trained on {collab_profiles.shape[0]} users")

# ─────────────────────────────────────────────────────────────
# MODEL 2  Re-order KNN  (order-history based)  ← NEW
# ─────────────────────────────────────────────────────────────
print("\n🔁  [Model 2] Training Re-order KNN (NearestNeighbors cosine on order history) ...")
reorder_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
reorder_knn.fit(order_profiles)
joblib.dump(reorder_knn,   os.path.join(MODEL_DIR, "reorder_knn.pkl"))
joblib.dump(order_profiles, os.path.join(MODEL_DIR, "order_profiles.pkl"))
print(f"   ✅  Re-order KNN trained on {order_profiles.shape[0]} users")

# ─────────────────────────────────────────────────────────────
# POPULARITY FALLBACK
# ─────────────────────────────────────────────────────────────
print("\n📈  Computing popularity scores (fallback) ...")
pop = (
    orders_df.groupby("product_id")
    .agg(order_count=("order_id", "count"), avg_rating=("rating", "mean"))
    .reset_index()
    .merge(products_df, on="product_id")
    .sort_values("order_count", ascending=False)
)
pop["popularity_score"] = (
    MinMaxScaler().fit_transform(pop[["order_count"]]) * 0.6 +
    MinMaxScaler().fit_transform(pop[["avg_rating"]]) * 0.4
)
joblib.dump(pop, os.path.join(MODEL_DIR, "popularity.joblib"))
print(f"   ✅  Popularity scores for {len(pop)} products")

print("\n🎉  All models trained and saved to /models\n")
print("  Model file               │ Purpose")
print("  ─────────────────────────┼──────────────────────────────────────────────")
print("  intent_pipeline.joblib   │ Classify user text → intent label")
print("  reorder_knn.pkl          │ Recommend based on YOUR order history (new)")
print("  collab_knn.pkl           │ Collaborative filtering (similar users)")
print("  user_profiles.pkl        │ User feature vectors (collab)")
print("  order_profiles.pkl       │ User feature vectors (order history)")
print("  hybrid_user_profiles.pkl │ Combined collab + order profile")
print("  item_vectorizer.joblib   │ TF-IDF for food item text")
print("  item_svd.joblib          │ SVD dimensionality reduction for items")
print("  item_emb.joblib          │ Final item embeddings (content-based)")
print("  popularity.joblib        │ Popularity fallback scores")

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Use relative paths from current directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Intent Classifier Training (assumes you have 'user_intents.csv' in DATA_DIR)
print("Training intent classifier...")
intents_path = f"{DATA_DIR}/user_intents.csv"
if not os.path.exists(intents_path):
    raise FileNotFoundError(f"{intents_path} not found. Please add your intent data.")

df_intents = pd.read_csv(intents_path)
X = df_intents["user_request"].astype(str)
y = df_intents["intent"].astype(str)

intent_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
intent_pipeline.fit(X_train, y_train)
print("Intent classifier accuracy:", intent_pipeline.score(X_test, y_test))
joblib.dump(intent_pipeline, f"{MODEL_DIR}/intent_pipeline.joblib")

# 2. Collaborative Filtering Recommender
print("Training collaborative filtering recommender...")
prefs_df = pd.read_csv(f"{DATA_DIR}/food_preferences.csv")
products_df = pd.read_csv(f"{DATA_DIR}/products.csv")

prefs_merged = prefs_df.merge(products_df[['product_id', 'category']], on='product_id', how='left')

features = pd.get_dummies(prefs_merged[['category']], dummy_na=True)
features['preference_score'] = prefs_merged['preference_score']

user_profiles = features.groupby(prefs_merged['user_id']).mean()

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_profiles)

joblib.dump(knn_model, f"{MODEL_DIR}/recommender_knn.pkl")
joblib.dump(user_profiles, f"{MODEL_DIR}/user_profiles.pkl")

# 3. Content-Based Recommender
print("Training content-based recommender...")
products_df['item_text'] = products_df['food_item'].fillna('') + " " + products_df['category'].fillna('')

item_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
item_matrix = item_vectorizer.fit_transform(products_df['item_text'])

svd = TruncatedSVD(n_components=min(100, item_matrix.shape[1]-1), random_state=42)
item_emb = svd.fit_transform(item_matrix)
item_emb = normalize(item_emb, axis=1)

joblib.dump(item_vectorizer, f"{MODEL_DIR}/item_vectorizer.joblib")
joblib.dump(svd, f"{MODEL_DIR}/item_svd.joblib")
joblib.dump(item_emb, f"{MODEL_DIR}/item_emb.joblib")

print("✅ All models trained and saved.")

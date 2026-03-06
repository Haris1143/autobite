import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

# Load food preferences dataset
df = pd.read_csv("data/user_preferences.csv")

# Prepare features (rating as importance)
features = pd.get_dummies(df[['category', 'time_of_day']])
features['rating'] = df['rating']

# Group by user
user_profiles = features.groupby(df['user_id']).mean()

# Train kNN recommender
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_profiles)

# Save model & profiles
joblib.dump(model, "models/recommender_model.pkl")
joblib.dump(user_profiles, "models/user_profiles.pkl")

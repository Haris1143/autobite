"""
generate_order_history.py
─────────────────────────
Generates a synthetic order_history.csv that combines:
  • food_preferences.csv  (preference_score as proxy for likelihood to order)
  • user_preferences.csv  (ratings + time_of_day context)
  • products.csv          (item metadata)

Output columns:
  order_id, user_id, product_id, food_item, category,
  time_of_day, order_date, quantity, rating
"""

import os, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
random.seed(42)
np.random.seed(42)

food_prefs = pd.read_csv(f"{DATA_DIR}/food_preferences.csv")
user_prefs  = pd.read_csv(f"{DATA_DIR}/user_preferences.csv")
products    = pd.read_csv(f"{DATA_DIR}/products.csv")

# Map user_prefs user_id like 'u001' → int 1 for merge
user_prefs["user_id_int"] = user_prefs["user_id"].str.replace("u", "").astype(int)

TIME_SLOTS = ["breakfast", "lunch", "snack", "dinner"]

orders = []
order_id = 1
start_date = datetime(2024, 1, 1)

print("Generating order history ...")
# Use top-scored preferences per user to simulate realistic orders
sample = food_prefs[food_prefs["preference_score"] >= 0.5].copy()
sample = sample.merge(products[["product_id", "food_item", "category"]], on="product_id")

# Build a quick lookup: user_id_int → {category → avg_rating, time_of_day}
user_pref_lookup = (
    user_prefs.groupby(["user_id_int", "category"])
    .agg(avg_rating=("rating", "mean"), time_of_day=("time_of_day", lambda x: x.mode()[0]))
    .reset_index()
)
lookup_dict = {}
for _, row in user_pref_lookup.iterrows():
    lookup_dict.setdefault(int(row["user_id_int"]), {})[row["category"]] = {
        "rating": row["avg_rating"],
        "time_of_day": row["time_of_day"],
    }

for _, row in sample.iterrows():
    uid = int(row["user_id"])
    cat = row["category"]

    # How many times did this user order this item? (1-4 based on score)
    n_orders = max(1, int(row["preference_score"] * 4))

    for _ in range(n_orders):
        info = lookup_dict.get(uid, {}).get(cat, {})
        time_of_day = info.get("time_of_day", random.choice(TIME_SLOTS))
        rating      = round(info.get("rating", random.uniform(3, 5)), 1)
        days_ago    = random.randint(0, 365)
        order_date  = (start_date + timedelta(days=days_ago)).strftime("%Y-%m-%d")
        quantity    = random.randint(1, 3)

        orders.append({
            "order_id":   order_id,
            "user_id":    uid,
            "product_id": int(row["product_id"]),
            "food_item":  row["food_item"],
            "category":   cat,
            "time_of_day": time_of_day,
            "order_date": order_date,
            "quantity":   quantity,
            "rating":     min(5.0, rating),
        })
        order_id += 1

df_orders = pd.DataFrame(orders)
df_orders.to_csv(f"{DATA_DIR}/order_history.csv", index=False)
print(f"✅ order_history.csv  →  {len(df_orders):,} rows  |  {df_orders['user_id'].nunique()} users")

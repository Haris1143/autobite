"""
generate_dataset.py — AutoBite Synthetic Dataset Generator
════════════════════════════════════════════════════════════

Generates ALL CSV files needed for training from scratch:
  data/users.csv
  data/products.csv
  data/food_preferences.csv
  data/user_preferences.csv
  data/user_intents.csv
  data/order_history.csv   ← NEW
"""

import os, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    os.system("pip install faker -q")
    from faker import Faker
    fake = Faker()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

NUM_USERS    = 1000
NUM_PRODUCTS = 100
PREFS_PER    = 100

CATEGORIES   = ["spicy", "sweet", "mild", "savory", "vegan", "dessert", "fast food", "beverage"]
TIME_SLOTS   = ["breakfast", "lunch", "snack", "dinner"]
INTENT_MAP   = {
    "breakfast": "breakfast_request",
    "spicy":     "spicy_dinner",
    "paneer":    "paneer_craving",
    "veg":       "light_veg",
    "non_veg":   "non_veg_lunch",
    "snack":     "snack_time",
}

# ── Users ─────────────────────────────────────────────────────────────────────
users = [{"user_id": i, "name": fake.name(), "email": fake.email(),
           "age": random.randint(18, 65), "location": fake.city()}
         for i in range(1, NUM_USERS + 1)]
pd.DataFrame(users).to_csv(f"{DATA_DIR}/users.csv", index=False)

# ── Products ──────────────────────────────────────────────────────────────────
DISH_TYPES = ["Burger", "Pizza", "Salad", "Soup", "Rice", "Cake", "Juice", "Sandwich",
              "Thali", "Curry", "Dosa", "Biryani"]
products = [{"product_id": i,
              "food_item": fake.word().capitalize() + " " + random.choice(DISH_TYPES),
              "category": random.choice(CATEGORIES),
              "price": round(random.uniform(2.0, 25.0), 2)}
            for i in range(1, NUM_PRODUCTS + 1)]
pd.DataFrame(products).to_csv(f"{DATA_DIR}/products.csv", index=False)

# ── Food Preferences ──────────────────────────────────────────────────────────
prefs = [{"user_id": u, "product_id": p,
           "preference_score": round(random.uniform(0.1, 1.0), 2)}
         for u in range(1, NUM_USERS + 1)
         for p in random.sample(range(1, NUM_PRODUCTS + 1), PREFS_PER)]
pd.DataFrame(prefs).to_csv(f"{DATA_DIR}/food_preferences.csv", index=False)

# ── User Preferences (rated items) ────────────────────────────────────────────
prod_df = pd.read_csv(f"{DATA_DIR}/products.csv")
FOOD_ITEMS = prod_df["food_item"].tolist()
CATS       = prod_df["category"].tolist()
user_prefs = []
for uid in range(1, NUM_USERS + 1):
    uid_str = f"u{uid:03d}"
    n = random.randint(10, 30)
    idxs = random.sample(range(len(FOOD_ITEMS)), n)
    for idx in idxs:
        user_prefs.append({"user_id": uid_str, "food_item": FOOD_ITEMS[idx],
                            "category": CATS[idx],
                            "time_of_day": random.choice(TIME_SLOTS),
                            "rating": random.randint(3, 5)})
pd.DataFrame(user_prefs).to_csv(f"{DATA_DIR}/user_preferences.csv", index=False)

# ── User Intents ──────────────────────────────────────────────────────────────
INTENT_TEMPLATES = {
    "breakfast_request": ["Suggest something for breakfast", "What's good for breakfast?",
                           "I want a morning meal", "Give me breakfast ideas"],
    "spicy_dinner":      ["Any spicy dinner ideas?", "I want something spicy for dinner",
                           "Hot and spicy dinner please", "Recommend a spicy dish"],
    "paneer_craving":    ["I'm craving paneer", "Give me paneer dishes",
                           "Something with paneer", "Paneer recommendations"],
    "light_veg":         ["I want something light and veg", "Healthy vegetarian options",
                           "Light vegetarian meal please", "Any veg options?"],
    "non_veg_lunch":     ["Non-veg lunch options", "I want chicken for lunch",
                           "Meat dish for lunch", "Non vegetarian lunch ideas"],
    "snack_time":        ["I need a quick snack", "Snack suggestions", "Something light to munch",
                           "Quick bite options"],
}
intents = []
for uid in range(1, NUM_USERS + 1):
    uid_str = f"u{uid:03d}"
    for intent, templates in INTENT_TEMPLATES.items():
        for t in templates:
            intents.append({"user_id": uid_str, "user_request": t, "intent": intent})
            # slight variation
            intents.append({"user_id": uid_str, "user_request": t + " please", "intent": intent})
pd.DataFrame(intents).to_csv(f"{DATA_DIR}/user_intents.csv", index=False)

# ── Order History ─────────────────────────────────────────────────────────────
pref_df = pd.read_csv(f"{DATA_DIR}/food_preferences.csv")
pref_df = pref_df.merge(prod_df[["product_id","food_item","category"]], on="product_id")
high    = pref_df[pref_df["preference_score"] >= 0.5]

start = datetime(2024, 1, 1)
orders, oid = [], 1
for _, row in high.iterrows():
    n = max(1, int(row["preference_score"] * 4))
    for _ in range(n):
        orders.append({
            "order_id":    oid,
            "user_id":     int(row["user_id"]),
            "product_id":  int(row["product_id"]),
            "food_item":   row["food_item"],
            "category":    row["category"],
            "time_of_day": random.choice(TIME_SLOTS),
            "order_date":  (start + timedelta(days=random.randint(0,365))).strftime("%Y-%m-%d"),
            "quantity":    random.randint(1, 3),
            "rating":      round(random.uniform(3.0, 5.0), 1),
        })
        oid += 1

pd.DataFrame(orders).to_csv(f"{DATA_DIR}/order_history.csv", index=False)

print("✅  Dataset generation complete!")
print(f"   users.csv          : {NUM_USERS} rows")
print(f"   products.csv       : {NUM_PRODUCTS} rows")
print(f"   food_preferences   : {len(prefs)} rows")
print(f"   user_preferences   : {len(user_prefs)} rows")
print(f"   user_intents       : {len(intents)} rows")
print(f"   order_history      : {len(orders)} rows")

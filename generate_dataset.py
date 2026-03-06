import os
import pandas as pd
import random
from faker import Faker

fake = Faker()

DATA_DIR = r"E:\LLm\project\autobite-ai\data"
os.makedirs(DATA_DIR, exist_ok=True)

# Parameters
NUM_USERS = 1000
NUM_PRODUCTS = 100
PREFERENCES_PER_USER = 100

# Generate Users
users = []
for user_id in range(1, NUM_USERS + 1):
    users.append({
        "user_id": user_id,
        "name": fake.name(),
        "email": fake.email(),
        "age": random.randint(18, 65),
        "location": fake.city()
    })

df_users = pd.DataFrame(users)
df_users.to_csv(f"{DATA_DIR}/users.csv", index=False)

# Generate Products
categories = ["spicy", "sweet", "mild", "savory", "vegan", "dessert", "fast food", "beverage"]
products = []
for product_id in range(1, NUM_PRODUCTS + 1):
    products.append({
        "product_id": product_id,
        "food_item": fake.word().capitalize() + " " + random.choice(
            ["Burger", "Pizza", "Salad", "Soup", "Rice", "Cake", "Juice", "Sandwich"]),
        "category": random.choice(categories),
        "price": round(random.uniform(2.0, 25.0), 2)
    })

df_products = pd.DataFrame(products)
df_products.to_csv(f"{DATA_DIR}/products.csv", index=False)

# Generate Food Preferences
preferences = []
for user in df_users["user_id"]:
    product_choices = random.sample(range(1, NUM_PRODUCTS + 1), PREFERENCES_PER_USER)
    for product in product_choices:
        preferences.append({
            "user_id": user,
            "product_id": product,
            "preference_score": round(random.uniform(0.1, 1.0), 2)
        })

df_preferences = pd.DataFrame(preferences)
df_preferences.to_csv(f"{DATA_DIR}/food_preferences.csv", index=False)

print("✅ Dataset generation complete!")
print(f"Users: {len(df_users)}")
print(f"Products: {len(df_products)}")
print(f"Preferences: {len(df_preferences)}")

# 🍽️ AutoBite - AI Powered Food Ordering & Recommendation System

<p align="center">

AI-powered intelligent food ordering platform that combines **Machine Learning**, **Natural Language Processing**, and **Recommendation Systems** to deliver personalized food suggestions and a smarter ordering experience.

</p>

---

# 🚀 Overview

AutoBite is an AI-driven food ordering system designed to make online food ordering more intelligent and personalized.

Unlike traditional food delivery applications, AutoBite learns user preferences, analyzes previous orders, understands food intent, and recommends dishes using multiple Machine Learning models.

The system combines recommendation algorithms with intelligent food preference analysis to improve customer experience.

---

# 🎯 Problem Statement

Traditional food ordering platforms display the same menu for every user.

AutoBite solves this problem by providing

- Personalized Recommendations
- Intelligent Food Suggestions
- Preference Learning
- Smart Reordering
- AI-based Recommendation Engine

making food ordering faster and more personalized.

---

# ✨ Features

## 🍔 Smart Food Recommendation

- Personalized dish recommendations
- Hybrid recommendation system
- Collaborative filtering
- Popularity-based recommendation

---

## 🤖 AI Recommendation Engine

The recommendation engine learns from

- Previous Orders
- User Preferences
- Food Categories
- User Behaviour
- Popular Dishes

---

## 🎙 Voice Ordering

Supports intelligent voice-based ordering for a faster user experience.

Future support:

- English
- Tamil
- Hindi

---

## 📊 User Preference Analysis

The system continuously analyzes

- Favourite Foods
- Frequently Ordered Items
- Taste Preferences
- Cuisine Preferences

---

## 🔄 Order History Analysis

Previous orders are used to

- Predict future orders
- Recommend similar foods
- Improve recommendation accuracy

---

# 🧠 Machine Learning Models

AutoBite combines multiple ML models.

### Recommendation Models

- Collaborative Filtering
- KNN Recommendation
- Popularity Recommendation
- Hybrid Recommendation

### NLP Models

- Intent Detection
- Food Query Understanding
- User Intent Classification

---

# 🏗️ System Architecture

```text
                   User
                     │
                     ▼
          Web Interface / Voice Input
                     │
                     ▼
            Flask Application
                     │
     ┌───────────────┼────────────────┐
     │               │                │
     ▼               ▼                ▼
User Preference   NLP Intent      Food Database
 Learning         Detection
     │               │
     └──────┬────────┘
            ▼
   Recommendation Engine
            │
            ▼
 Personalized Food Suggestions
            │
            ▼
      Order Placement
```

---

# 🔄 Workflow

### Step 1

User visits AutoBite.

↓

### Step 2

User searches or speaks a food request.

↓

### Step 3

System identifies the user's intent.

↓

### Step 4

Machine Learning models analyze

- Previous Orders
- Preferences
- Similar Users
- Popular Foods

↓

### Step 5

Recommendation engine generates personalized food suggestions.

↓

### Step 6

User places the order.

↓

### Step 7

Order history is stored to improve future recommendations.

---

# 🛠️ Technologies Used

## Backend

- Python
- Flask

## Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

## Recommendation System

- KNN
- Collaborative Filtering
- Hybrid Recommendation
- Popularity Model

## Frontend

- HTML
- CSS
- JavaScript

## Deployment

- Docker
- Vercel

---

# 📂 Project Structure

```text
AutoBite

│
├── api/
│
├── data/
│   ├── Users
│   ├── Products
│   ├── Order History
│   ├── Food Preferences
│   └── User Intents
│
├── models/
│   ├── Collaborative Model
│   ├── KNN Model
│   ├── Hybrid Model
│   ├── Popularity Model
│   └── Vectorizers
│
├── scripts/
│
├── app.py
├── train_all.py
├── train_recommender.py
├── generate_dataset.py
├── Dockerfile
└── requirements.txt
```

---

# 📊 Dataset

The project uses multiple datasets for training.

### User Dataset

Stores

- User Details
- Food Preferences

### Product Dataset

Contains

- Food Items
- Categories
- Prices

### Order Dataset

Stores

- Previous Orders
- Purchase Frequency

### Food Preference Dataset

Used for personalized recommendation.

---

# 🧠 AI Pipeline

```text
User Input
      │
      ▼
Intent Detection
      │
      ▼
Food Understanding
      │
      ▼
Recommendation Engine
      │
      ▼
Machine Learning Models
      │
      ▼
Rank Food Items
      │
      ▼
Return Best Recommendations
```

---

# ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/Haris1143/autobite.git
```

Move into project

```bash
cd autobite
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
python app.py
```

---

# 🐳 Docker

Build

```bash
docker build -t autobite .
```

Run

```bash
docker run -p 5000:5000 autobite
```

---

# 📈 Future Enhancements

- 🍽 Multi-Restaurant Support
- 💳 Online Payments
- 📍 Live Order Tracking
- 🤖 LLM-powered Food Assistant
- 🌍 Multi-language Voice Ordering
- 📱 Mobile Application
- 🔊 Speech-to-Text Ordering
- 😊 Sentiment-based Food Recommendation
- 🥗 Health-aware Meal Suggestions

---

# 👨‍💻 Author

**Haris B**

B.Tech – Artificial Intelligence & Machine Learning

St. Joseph's College of Engineering

GitHub

https://github.com/Haris1143

---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.

It motivates further development and helps others discover the project.

---

# 📄 License

This project is developed for educational, research, and learning purposes.

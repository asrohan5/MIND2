import os
import pandas as pd
import joblib


MODELS_DIR = "D:/Projects/MIND2/artifacts/models"
MODEL_PATH = os.path.join(MODELS_DIR, "pipeline_logreg_tuned.pkl")

print(f"Loading pipeline model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


sample_data = pd.DataFrame([
    {"title": "Biden addresses economic recovery plan", "category": "news", "subcategory": "politics"},
    {"title": "Manchester United wins the Premier League", "category": "sports", "subcategory": "football"}
])


pred_labels = model.predict(sample_data)
pred_probs = model.predict_proba(sample_data)[:, 1]


for i, row in sample_data.iterrows():
    print(f"Title: {row['title']}")
    print(f"Predicted Label: {pred_labels[i]} | Probability: {pred_probs[i]:.4f}")


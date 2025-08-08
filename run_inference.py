
import pandas as pd
import os
import pickle
import joblib
from scipy import sparse
import numpy as np


ARTIFACTS_DIR = "D:/Projects/MIND2/artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "models", "logreg_tfidf_cat.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "models", "category_encoder.pkl")
TFIDF_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "processed", "title_vectorizer.pkl")


print("Loading model and preprocessors")
model = joblib.load(MODEL_PATH)
category_encoder = joblib.load(ENCODER_PATH)
with open(TFIDF_MATRIX_PATH, "rb") as f:
    title_vectorizer = pickle.load(f)

print("Artifacts loaded successfully.")


def preprocess_data(df):
    
    df["title"] = df["title"].fillna("unknown_title")
    df["category"] = df["category"].fillna("unknown_cat")
    df["subcategory"] = df["subcategory"].fillna("unknown_subcat")

    
    X_title = title_vectorizer.transform(df["title"])

    
    encoded_df = category_encoder.transform(df)
    cat_cols = [col for col in encoded_df.columns if col.startswith("category_")]
    subcat_cols = [col for col in encoded_df.columns if col.startswith("subcategory_")]
    X_cat = sparse.csr_matrix(encoded_df[cat_cols].values)
    X_subcat = sparse.csr_matrix(encoded_df[subcat_cols].values)

    
    X_full = sparse.hstack([X_title, X_cat, X_subcat])
    return X_full

'''
if __name__ == "__main__":
    
    NEW_DATA_PATH = "D:/Projects/MIND2/data/sample_new_news.csv"
    df_new = pd.read_csv(NEW_DATA_PATH)

    X_new = preprocess_data(df_new)
    preds = model.predict(X_new)
    probs = model.predict_proba(X_new)[:, 1]

    df_new["predicted_label"] = preds
    df_new["probability"] = probs

    output_path = "D:/Projects/MIND2/artifacts/predictions/predicted_news.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_new.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
'''
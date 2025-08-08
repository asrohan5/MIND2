import os
import pickle
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


cleaned_path = os.path.join("D:/Projects/MIND2/artifacts", "processed", "cleaned_sample.csv")
print(f"Loading cleaned data from {cleaned_path}")
df = pd.read_csv(cleaned_path)
print(f"Loaded shape: {df.shape}")


tfidf_matrix_path = os.path.join("D:/Projects/MIND2/artifacts", "processed", "title_tfidf.npz")
tfidf_names_path = os.path.join("D:/Projects/MIND2/artifacts", "processed", "tfidf_features.csv")

print("Loading TF-IDF features")
X_tfidf = sparse.load_npz(tfidf_matrix_path)
tfidf_feature_names = pd.read_csv(tfidf_names_path, header=None)[0].tolist()

print(f"TF-IDF matrix shape: {X_tfidf.shape}")

print("Encoding category and subcategory")
cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
X_cat = cat_encoder.fit_transform(df[["category", "subcategory"]])
print(f"Category/Subcategory matrix shape: {X_cat.shape}")


X = hstack([X_tfidf, X_cat])
y = df["label"]
print(f"Final feature matrix shape: {X.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")


#########################LOGISTIC REGRESSION
model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")


model_dir = os.path.join("D:/Projects/MIND2/artifacts", "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "logreg_tfidf_cat.pkl")
encoder_path = os.path.join(model_dir, "category_encoder.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(encoder_path, "wb") as f:
    pickle.dump(cat_encoder, f)

print(f"Model saved to {model_path}")
print(f"Category encoder saved to {encoder_path}")

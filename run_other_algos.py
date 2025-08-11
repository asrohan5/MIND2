import os
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


MODEL_PARAMS = {
    "logreg": (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["liblinear", "lbfgs"]
        }
    ),
    "rf": (
        RandomForestClassifier(random_state=42),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [10, 20, None]
        }
    ),
    "svm": (
        SVC(probability=True, random_state=42),
        {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf"]
        }
    ),
    "xgb": (
        XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.3]
        }
    ),
    "nb": (
        MultinomialNB(),
        {
            "clf__alpha": [0.1, 1, 10]
        }
    )
}



def main(model_name):
    if model_name not in MODEL_PARAMS:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from: {list(MODEL_PARAMS.keys())}")


    artifacts_dir = "D:/Projects/MIND2/artifacts"
    models_dir = os.path.join(artifacts_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    logs_path = os.path.join(artifacts_dir, "experiments_log.csv")


    print("Loading dataset")
    df = pd.read_csv("D:/Projects/MIND2/artifacts/processed/merged_click_data.csv")

    X = df[["title", "category", "subcategory"]]
    y = df["label"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    text_transformer = TfidfVectorizer(max_features=5000)
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("title_tfidf", text_transformer, "title"),
            ("cat_ohe", cat_transformer, ["category", "subcategory"])
        ]
    )


    model, param_grid = MODEL_PARAMS[model_name]


    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", model)
    ])


    print(f"Tuning {model_name}")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)


    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")


    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else np.zeros(len(y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else None

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}" if auc else "AUC: Not applicable")
    print(classification_report(y_test, y_pred))


    model_path = os.path.join(models_dir, f"pipeline_{model_name}_tuned.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {model_path}")


    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "best_params": str(grid_search.best_params_),
        "accuracy": accuracy,
        "auc": auc
    }
    if os.path.exists(logs_path):
        pd.DataFrame([log_data]).to_csv(logs_path, mode="a", header=False, index=False)
    else:
        pd.DataFrame([log_data]).to_csv(logs_path, index=False)
    print(f"Experiment logged to {logs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline with chosen model")
    parser.add_argument("--model", type=str, required=True, help="Model to train: logreg, rf, svm, xgb, nb")
    args = parser.parse_args()
    main(args.model)

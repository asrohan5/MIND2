import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


ARTIFACT_DIR = "D:/Projects/MIND2/artifacts"
PROCESSED_DIR = os.path.join(ARTIFACT_DIR, "processed")
MODELS_DIR = os.path.join(ARTIFACT_DIR, "models")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(PROCESSED_DIR, "merged_click_data.csv")
SAMPLE_SIZE = 10000


def log_experiment(model_name, params, acc, auc, notes=""):
    log_path = os.path.join(ARTIFACT_DIR, "experiments_log.csv")
    log_entry = pd.DataFrame([{
        "model": model_name,
        "hyperparameters": str(params),
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "date": pd.Timestamp.now(),
        "notes": notes
    }])
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, log_entry], ignore_index=True)
    else:
        log_df = log_entry
    log_df.to_csv(log_path, index=False)
    print(f"Experiment logged to {log_path}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    plt.close()

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(MODELS_DIR, "roc_curve.png"))
    plt.close()


if __name__ == "__main__":
   
    print(f"Loading top {SAMPLE_SIZE} rows from dataset")
    df = pd.read_csv(RAW_DATA_PATH, nrows=SAMPLE_SIZE)
    df['title'] = df['title'].fillna("unknown_title")
    df['category'] = df['category'].fillna("unknown_cat")
    df['subcategory'] = df['subcategory'].fillna("unknown_subcat")

    X = df[['title', 'category', 'subcategory']]
    y = df['label']

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    preprocessor = ColumnTransformer(
        transformers=[
            ('title_tfidf', TfidfVectorizer(max_features=5000), 'title'),
            ('cat_ohe', OneHotEncoder(handle_unknown='ignore'), ['category', 'subcategory'])
        ]
    )

    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight="balanced", max_iter=500))
    ])

    
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__solver': ['liblinear', 'lbfgs']
    }

    grid_search = GridSearchCV(
        pipe, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    
    model_path = os.path.join(MODELS_DIR, "pipeline_logreg_tuned.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model pipeline saved to {model_path}")

    
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

    
    log_experiment(
        model_name="Pipeline(LogisticRegression)",
        params=grid_search.best_params_,
        acc=acc,
        auc=auc,
        notes="Pipeline with TF-IDF + OneHot encoding + tuned Logistic Regression"
    )

    print("\nPipeline completed successfully!")

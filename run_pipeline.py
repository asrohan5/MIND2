import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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



def load_and_clean():
    print(f"Loading top {SAMPLE_SIZE} rows from dataset")
    df = pd.read_csv(RAW_DATA_PATH, nrows=SAMPLE_SIZE)
    print(f"Loaded shape: {df.shape}")

    print("Handling missing values")
    df['title'] = df['title'].fillna("unknown_title")
    df['category'] = df['category'].fillna("unknown_cat")
    df['subcategory'] = df['subcategory'].fillna("unknown_subcat")

    cleaned_path = os.path.join(PROCESSED_DIR, "cleaned_sample.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned data to {cleaned_path}")
    return df



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


def train_pipeline(df):
    X = df[['title', 'category', 'subcategory']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("title_tfidf", TfidfVectorizer(max_features=5000), "title"),
            ("category_ohe", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("subcategory_ohe", OneHotEncoder(handle_unknown="ignore"), ["subcategory"])
        ]
    )


    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=500))
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)


    model_path = os.path.join(MODELS_DIR, "news_click_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

    return pipeline


def run_inference(pipeline, sample_df):
    preds = pipeline.predict(sample_df)
    probs = pipeline.predict_proba(sample_df)[:, 1]
    out_df = sample_df.copy()
    out_df['predicted_label'] = preds
    out_df['predicted_probability'] = probs
    return out_df


if __name__ == "__main__":
    df_clean = load_and_clean()
    pipeline = train_pipeline(df_clean)

    sample_new = pd.DataFrame([
        {"title": "New advances in AI transform health diagnostics", "category": "technology", "subcategory": "ai"},
        {"title": "Local team wins championship after dramatic final", "category": "sports", "subcategory": "basketball"},
    ])
    predictions_df = run_inference(pipeline, sample_new)
    print("\nSample predictions:")
    print(predictions_df)

    predictions_path = os.path.join(ARTIFACT_DIR, "predictions")
    os.makedirs(predictions_path, exist_ok=True)
    predictions_df.to_csv(os.path.join(predictions_path, "sample_predictions.csv"), index=False)
    print(f"Saved demo predictions to {os.path.join(predictions_path, 'sample_predictions.csv')}")

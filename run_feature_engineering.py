import os
import pandas as pd
from scipy import sparse
from src.components.title_vectorizer import TitleVectorizer
#from src.components.feature_engineering import NewsFeatureEngineer

cleaned_df = pd.read_csv("D:/Projects/MIND2/artifacts/processed/cleaned_sample.csv")

def load_data(path, n_rows: int = 10000):
    print(f"Loading top {n_rows} rows from merged dataset")
    df = pd.read_csv(path, nrows=n_rows)
    print(f"Loaded shape: {df.shape}")
    return df

def clean_data(df):
    print("Handling missing values in title, category, subcategory")
    df['title'] = df['title'].fillna("unknown_title")
    df['category'] = df['category'].fillna("unknown_cat")
    df['subcategory'] = df['subcategory'].fillna("unknown_subcat")
    print("Missing values are handled")
    return df


title_vectorizer = TitleVectorizer(max_features=5000)
X_title = title_vectorizer.fit_transform(cleaned_df['title'])

sparse.save_npz("D:/Projects/MIND2/artifacts/processed/title_tfidf.npz", X_title)

pd.Series(title_vectorizer.get_feature_names()).to_csv("D:/Projects/MIND2/artifacts/processed/tfidf_features.csv", index=False)
print("Saved TF-IDF matrix and feature names")


def main():
    os.makedirs("D:/Projects/MIND2/artifacts/processed", exist_ok=True)

    df = load_data("D:/Projects/MIND2/artifacts/processed/merged_click_data.csv", n_rows=10000)

    cleaned_df = clean_data(df)

    cleaned_df.to_csv("D:/Projects/MIND2/artifacts/processed/cleaned_sample.csv", index=False)
    print("Saved cleaned subset")

if __name__ == "__main__":
    main()

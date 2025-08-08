import pandas as pd
import numpy as np
import os
import joblib
from scipy import sparse

from src.components.category_encoder import CategoryEncoder


cleaned_df = pd.read_csv("D:/Projects/MIND2/artifacts/processed/cleaned_sample.csv")


encoder = CategoryEncoder(save_dir="D:/Projects/MIND2/artifacts/processed")
encoded_df = encoder.fit_transform(cleaned_df)


tfidf_matrix = sparse.load_npz("D:/Projects/MIND2/artifacts/processed/title_tfidf.npz")


category_cols = [col for col in encoded_df.columns if col.startswith("category_")]
subcategory_cols = [col for col in encoded_df.columns if col.startswith("subcategory_")]

category_matrix = sparse.csr_matrix(encoded_df[category_cols].values)
subcategory_matrix = sparse.csr_matrix(encoded_df[subcategory_cols].values)

full_feature_matrix = sparse.hstack([tfidf_matrix, category_matrix, subcategory_matrix])


os.makedirs("D:/Projects/MIND2/artifacts/features", exist_ok=True)
sparse.save_npz("D:/Projects/MIND2/artifacts/features/full_features.npz", full_feature_matrix)


labels = encoded_df["label"].values
np.save("D:/Projects/MIND2/artifacts/features/labels.npy", labels)

print(f"Full feature matrix shape: {full_feature_matrix.shape}")
print(f"Labels shape: {labels.shape}")





'''
import pandas as pd
from src.components.category_encoder import CategoryEncoder

cleaned_df = pd.read_csv("D:/Projects/MIND2/artifacts/processed/cleaned_sample.csv")


encoder = CategoryEncoder(save_dir="D:/Projects/MIND2/artifacts/processed")
encoded_df = encoder.fit_transform(cleaned_df)

print(f"Encoded dataset shape: {encoded_df.shape}")
print(f"Saved encoded dataset ")

'''


'''
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
'''

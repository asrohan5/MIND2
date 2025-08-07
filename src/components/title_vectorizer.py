from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TitleVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )

    def fit_transform(self, titles):
        print("Fitting and transforming titles using TF-IDF")
        X = self.vectorizer.fit_transform(titles)
        print(f"TF-IDF matrix shape: {X.shape}")
        return X

    def transform(self, titles):
        return self.vectorizer.transform(titles)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

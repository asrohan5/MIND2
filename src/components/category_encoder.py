import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

class CategoryEncoder:
    def __init__(self, save_dir="D:/Projects/MIND2/artifacts/processed"):
        self.save_dir = save_dir
        self.category_encoder = None
        self.subcategory_encoder = None

    def fit_transform(self, df):
        self.category_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.subcategory_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        category_encoded = self.category_encoder.fit_transform(df[['category']])
        subcategory_encoded = self.subcategory_encoder.fit_transform(df[['subcategory']])


        category_df = pd.DataFrame(
            category_encoded, 
            columns=self.category_encoder.get_feature_names_out(['category'])
        )
        subcategory_df = pd.DataFrame(
            subcategory_encoded, 
            columns=self.subcategory_encoder.get_feature_names_out(['subcategory'])
        )

        df = df.reset_index(drop=True)
        encoded_df = pd.concat([df, category_df, subcategory_df], axis=1)


        os.makedirs(self.save_dir, exist_ok=True)
        joblib.dump(self.category_encoder, os.path.join(self.save_dir, "category_encoder.joblib"))
        joblib.dump(self.subcategory_encoder, os.path.join(self.save_dir, "subcategory_encoder.joblib"))

        encoded_df.to_csv(os.path.join(self.save_dir, "encoded_categories.csv"), index=False)

        return encoded_df

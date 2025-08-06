'''
from src.components.data_parser import MINDDataParser
import pandas as pd

parser = MINDDataParser(raw_data_path='D:/Projects/MIND2/artifacts/raw')
behaviors_df = parser.parse_behaviors()
labeled_df = parser.generate_labeled_click_data(behaviors_df)

print("\nSample rows:")
print(labeled_df.head())

print("\nShape of labeled dataset:", labeled_df.shape)


labeled_df.to_csv("D:/Projects/MIND2/artifacts/processed/labeled_click_data.csv", index=False)
'''

import os
import pandas as pd
from src.components.data_parser import MINDDataParser

def main():
    os.makedirs("D:/Projects/MIND2/artifacts/processed", exist_ok=True)

    parser = MINDDataParser(raw_data_path='D:/Projects/MIND2/artifacts/raw')
    behaviors_df = parser.parse_behaviors()
    print(f"Behaviors DataFrame shape: {behaviors_df.shape}")

    labeled_df = parser.generate_labeled_click_data(behaviors_df)
    print(f"Labeled click data shape: {labeled_df.shape}")
    print(labeled_df.head())


    news_df = parser.parse_news("D:/Projects/MIND2/artifacts/raw/news.tsv")
    print(f"News metadata shape: {news_df.shape}")
    print(news_df.head()[["news_id", "title", "category"]])


    merged_df = parser.merge_clicks_with_news(labeled_df, news_df)
    print(f"Merged dataset shape: {merged_df.shape}")
    print(merged_df[["candidate_news", "title", "category", "label"]].head())


    merged_df.to_csv("D:/Projects/MIND2/artifacts/processed/merged_click_data.csv", index=False)

if __name__ == "__main__":
    main()

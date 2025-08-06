import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException

class MINDDataParser:
    def __init__(self, raw_data_path="D:/Projects/MIND2/artifacts/raw"):
        self.behaviors_file = os.path.join(raw_data_path, "behaviors.tsv")
        self.news_file = os.path.join(raw_data_path, "news.tsv")

    def parse_news(self):
        try:
            logging.info(f"Reading news file from: {self.news_file}")
            columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'entities', 'entity_abstracts']
            news_df = pd.read_csv(self.news_file, sep='\t', header=None, names=columns, encoding='utf-8')
            logging.info(f"Parsed news file with shape: {news_df.shape}")
            return news_df
        except Exception as e:
            raise CustomException(e, sys)

    def parse_behaviors(self):
        try:
            logging.info(f"Reading behaviors file from: {self.behaviors_file}")
            columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
            behaviors_df = pd.read_csv(self.behaviors_file, sep='\t', header=None, names=columns, encoding='utf-8')
            logging.info(f"Parsed behaviors file with shape: {behaviors_df.shape}")
            return behaviors_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def generate_labeled_click_data(self,behaviors_df):

        records = []

        for _, row in behaviors_df.iterrows():
            impression_id = row["impression_id"]
            user_id = row["user_id"]
            time = row["time"]
            impressions = row["impressions"].strip().split()

            for impression in impressions:
                if '-' not in impression:
                    continue 
                news_id, label = impression.split('-')
                records.append({
                    "impression_id": impression_id,
                    "user_id": user_id,
                    "time": time,
                    "candidate_news": news_id,
                    "label": int(label)
                })

        labeled_df = pd.DataFrame(records)
        return labeled_df

    def parse_news(self,news_path):
        columns = [
        "news_id", "category", "subcategory", 
        "title", "abstract", 
        "url", "title_entities", "abstract_entities"
        ]

        news_df = pd.read_csv(news_path, sep="\t", header=None, names=columns, encoding='utf-8')
        return news_df
    

    
    def merge_clicks_with_news(self, labeled_df, news_df):
    
        merged_df = pd.merge(
            labeled_df,
            news_df,
            how="left",
            left_on="candidate_news",
            right_on="news_id"
        )
        return merged_df



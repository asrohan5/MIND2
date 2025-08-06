from src.components.data_parser import MINDDataParser
import pandas as pd

parser = MINDDataParser(raw_data_path='D:/Projects/MIND2/artifacts/raw')
behaviors_df = parser.parse_behaviors()
labeled_df = parser.generate_labeled_click_data(behaviors_df)

print("\nSample rows:")
print(labeled_df.head())

print("\nShape of labeled dataset:", labeled_df.shape)


labeled_df.to_csv("D:/Projects/MIND2/artifacts/processed/labeled_click_data.csv", index=False)


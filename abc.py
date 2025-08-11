import pandas as pd

for file in [
    "D:/Projects/MIND2/artifacts/processed/labeled_click_data.csv",
    "D:/Projects/MIND2/artifacts/processed/merged_click_data.csv"
]:
    print(f"\n--- {file} ---")
    df = pd.read_csv(file)
    print(df.columns)
    print(df.head(3))

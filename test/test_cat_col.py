import pandas as pd

df = pd.read_csv("D:/Projects/MIND2/artifacts/processed/cleaned_sample.csv")
print("Shape of cleaned data:", df.shape)


print("\nInfo")
df.info()

print("\nUnique Value Counts")
for col in df.columns:
    unique_vals = df[col].nunique()
    print(f"{col}: {unique_vals} unique values")


print("\nSample Values")
print(df.head(3))

import pandas as pd

df = pd.read_csv(".kaggle/netflix_titles.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
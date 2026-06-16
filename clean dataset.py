import pandas as pd


df = pd.read_csv("netflix_titles.csv")



print(df.isnull().sum())


df.drop_duplicates(inplace=True)


df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")


df.to_csv("netflix_cleaned.csv", index=False)

print("Dataset cleaned successfully!")
import pandas as pd

df = pd.read_csv(".kaggle/netflix_titles.csv")

# Missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Convert date column safely
df['date_added'] = pd.to_datetime(
    df['date_added'],
    errors='coerce'
)

print("Dataset loaded successfully!")
print(df.head())
import pandas as pd

df = pd.read_csv(r".kaggle\netflix_titles.csv")

# Missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# Convert date column
df['date_added'] = pd.to_datetime(
    df['date_added'],
    errors='coerce'
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

df.to_csv("cleaned_netflix_titles.csv", index=False)

print("Cleaned dataset saved successfully!")
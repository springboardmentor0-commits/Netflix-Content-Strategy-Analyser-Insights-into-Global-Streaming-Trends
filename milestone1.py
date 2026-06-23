# Netflix Dataset Preparation - Milestone 1

import pandas as pd

# ----------------------------------
# Step 1: Load Dataset
# ----------------------------------
df = pd.read_csv("netflix_titles.csv")

print("Dataset Loaded Successfully!")
print(f"Original Shape: {df.shape}")

# ----------------------------------
# Step 2: Check Missing Values
# ----------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------------
# Step 3: Handle Missing Values
# ----------------------------------

# Fill missing categorical values
df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Not Rated")

# Fill missing date_added with mode
if df["date_added"].isnull().sum() > 0:
    df["date_added"] = df["date_added"].fillna(
        df["date_added"].mode()[0]
    )

# ----------------------------------
# Step 4: Remove Duplicates
# ----------------------------------
duplicates = df.duplicated().sum()
print(f"\nDuplicate Records Found: {duplicates}")

df = df.drop_duplicates()

print(f"Shape After Removing Duplicates: {df.shape}")

# ----------------------------------
# Step 5: Normalize Categorical Features
# ----------------------------------

# Normalize Genre
df["listed_in"] = (
    df["listed_in"]
    .str.lower()
    .str.strip()
)

# Normalize Rating
df["rating"] = (
    df["rating"]
    .str.upper()
    .str.strip()
)

# Normalize Country
df["country"] = (
    df["country"]
    .str.title()
    .str.strip()
)

# Normalize Type
df["type"] = (
    df["type"]
    .str.title()
    .str.strip()
)

# ----------------------------------
# Step 6: Verify Cleaned Dataset
# ----------------------------------
print("\nCleaned Dataset Information:")
print(df.info())

print("\nSample Records:")
print(df.head())

# ----------------------------------
# Step 7: Save Cleaned Dataset
# ----------------------------------
df.to_csv("netflix_cleaned.csv", index=False)

print("\nCleaned dataset saved as 'netflix_cleaned.csv'")
print("Milestone 1 Completed Successfully!")
# ==========================================
# MILESTONE 1: REQUIREMENTS & DATASET PREPARATION
# ==========================================

import pandas as pd

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
file_path = "netflix_titles.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("\nDataset Shape:", df.shape)

# ------------------------------------------
# 2. Display Basic Information
# ------------------------------------------
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ------------------------------------------
# 3. Remove Duplicate Records
# ------------------------------------------
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
duplicates_after = df.duplicated().sum()

print("\nDuplicates Removed:", duplicates_before)
print("Remaining Duplicates:", duplicates_after)

# ------------------------------------------
# 4. Handle Missing Values
# ------------------------------------------

# Replace missing text values with 'Unknown'
text_columns = [
    'director',
    'cast',
    'country',
    'rating'
]

for col in text_columns:
    df[col] = df[col].fillna('Unknown')

# Replace missing date values
df['date_added'] = df['date_added'].fillna('Not Available')

# Replace missing duration values
df['duration'] = df['duration'].fillna('Unknown')

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ------------------------------------------
# 5. Normalize Categorical Features
# ------------------------------------------

# Normalize Genre (listed_in)
df['listed_in'] = (
    df['listed_in']
    .str.lower()
    .str.strip()
)

# Normalize Rating
df['rating'] = (
    df['rating']
    .str.upper()
    .str.strip()
)

# Normalize Country
df['country'] = (
    df['country']
    .str.title()
    .str.strip()
)

# ------------------------------------------
# 6. Verify Normalization
# ------------------------------------------
print("\nSample Normalized Genres:")
print(df['listed_in'].head())

print("\nSample Normalized Ratings:")
print(df['rating'].unique()[:10])

print("\nSample Normalized Countries:")
print(df['country'].unique()[:10])

# ------------------------------------------
# 7. Save Cleaned Dataset
# ------------------------------------------
output_file = "netflix_titles_cleaned.csv"
df.to_csv(output_file, index=False)

print("\nCleaned dataset saved as:", output_file)

# ------------------------------------------
# 8. Final Summary
# ------------------------------------------
print("\nFinal Dataset Shape:", df.shape)

print("\nCleaning Completed Successfully!")
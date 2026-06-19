# =====================================================
# NETFLIX DATASET PREPARATION (Milestone 1: Week 1 & 2)
# =====================================================

import pandas as pd

# 1. Load the Netflix Kaggle Dataset
df = pd.read_csv("netflix_titles.csv")

print("Original Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# -----------------------------------------------------
# 2. Check Missing Values
# -----------------------------------------------------
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# -----------------------------------------------------
# 3. Handle Missing Values
# -----------------------------------------------------
df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Not Rated")

# Remove rows with missing values in important columns
df = df.dropna(subset=["date_added", "duration"])

# -----------------------------------------------------
# 4. Remove Duplicate Records
# -----------------------------------------------------
duplicates_before = df.duplicated().sum()
print("\nNumber of Duplicate Rows:", duplicates_before)

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()
print("Duplicate Rows After Removal:", duplicates_after)

# -----------------------------------------------------
# 5. Normalize Categorical Features
# -----------------------------------------------------

# Normalize Genre (listed_in)
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

# -----------------------------------------------------
# 6. Verify Cleaning Results
# -----------------------------------------------------
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

print("\nSample Cleaned Data:")
print(df.head())

# -----------------------------------------------------
# 7. Save Cleaned Dataset
# -----------------------------------------------------
output_file = "netflix_cleaned.csv"
df.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved successfully as '{output_file}'")

# -----------------------------------------------------
# 8. Project Summary
# -----------------------------------------------------
print("\n========== PROJECT SUMMARY ==========")
print("Project Scope:")
print("- Analyze Netflix content data.")
print("- Prepare clean data for analytics and visualization.")
print("- Enable future recommendation and business insights.")

print("\nSuccess Metrics:")
print("- Missing values handled.")
print("- Duplicate records removed.")
print("- Genre, rating, and country normalized.")
print("- Dataset ready for EDA and visualization.")
print("=====================================")
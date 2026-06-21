import pandas as pd
import numpy as np

# If you uploaded the raw CSV file directly to your workspace:
file_path = "netflix_titles.csv" 

print("--- Step 1: Loading Dataset ---")
df = pd.read_csv(file_path)
print(f"Dataset Loaded. Shape: {df.shape}")

print("\n--- Step 2: Data Cleaning ---")
# Remove duplicates
df = df.drop_duplicates(subset=['show_id'])

# Handle missing values
categorical_with_nas = ['director', 'cast', 'country', 'rating']
for col in categorical_with_nas:
    df[col] = df[col].fillna('Unknown')

# Parse date strings to Datetime
df['date_added'] = df['date_added'].str.strip()
df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')
if df['date_added'].isnull().sum() > 0:
    df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])

# Extract numbers from duration
df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)

print("\n--- Step 3: Categorical Feature Normalization ---")
df['type'] = df['type'].str.strip().str.title()
df['rating'] = df['rating'].str.strip().str.upper()

# Normalize Multi-valued columns by taking the primary (first) entry
df['primary_country'] = df['country'].apply(lambda x: x.split(',')[0].strip())
df['primary_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0].strip())

print("Cleaning operations complete!")
print("\n--- Data Quick Check ---")
# FIXED: Changed cleaned_df to df
print(df[['show_id', 'type', 'title', 'primary_country', 'primary_genre']].head())

# Save the cleaned file
df.to_csv("cleaned_netflix_titles.csv", index=False)
print("\n'cleaned_netflix_titles.csv' has been saved and is ready!")
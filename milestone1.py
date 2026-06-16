import pandas as pd
import numpy as np

# 1. LOAD THE DATASET
# Reading the uploaded file
df = pd.read_csv("netflix_titles.csv")
print(f"Dataset loaded successfully. Shape: {df.shape}")

# ---

# 2. CLEAN THE DATASET
# Check for duplicates based on show_id or title
duplicates_count = df.duplicated(subset=['title']).sum()
df = df.drop_duplicates(subset=['title'], keep='first')
print(f"Removed {duplicates_count} duplicate titles.")

# Handling Missing Values (NaNs)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Strategy for nulls:
# - 'director', 'cast', 'country': Replace with 'Unknown'
# - 'date_added', 'rating', 'duration': Drop or fill with placeholders since they are few
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# Drop rows with missing crucial dates or ratings (minimal data loss)
df = df.dropna(subset=['date_added', 'rating', 'duration'])

# ---

# 3. NORMALIZE CATEGORICAL FEATURES
# Standardizing string formats (stripping whitespace and lowercasing for consistency)
df['rating'] = df['rating'].str.strip().str.upper()

# Clean date_added to a standardized datetime format
df['date_added'] = df['date_added'].str.strip()
df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')

# Crucial Normalization: Expanding multi-valued categories (Genres and Countries)
# Because columns like 'listed_in' contain "Comedies, Horror Movies", we split them.

def explode_categorical(dataframe, column_name):
    """Splits comma-separated strings into individual rows for clean analysis."""
    # Split by comma, strip whitespace, and explode into separate rows
    exploded_df = dataframe.copy()
    exploded_df[column_name] = exploded_df[column_name].str.split(', ')
    return exploded_df.explode(column_name)

# Create normalized, separate DataFrames for deep-dive analysis
df_genres_normalized = explode_categorical(df, 'listed_in')
df_countries_normalized = explode_categorical(df, 'country')

# ---

# 4. VERIFY RESULTS
print("\n--- Milestone 1 Summary ---")
print(f"Final cleaned dataset shape: {df.shape}")
print(f"Unique Genres found: {df_genres_normalized['listed_in'].nunique()}")
print(f"Unique Countries found: {df_countries_normalized['country'].nunique()}")
print("Missing values remaining:", df.isnull().sum().sum())
df.to_csv('cleaned_netflix_data.csv', index=False)
import pandas as pd
import numpy as np

def run_milestone_1_pipeline(file_path='netflix_titles.csv', output_path='netflix_titles_cleaned.csv'):
    print("=== Milestone 1: Starting Combined Pandas & NumPy Pipeline ===")
    
    # 1. Load the Netflix dataset using Pandas
    try:
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Dataset loaded. Initial Shape: {df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] The file '{file_path}' was not found. Please verify the path.")
        return None

    # 2. Clean Anomalies using NumPy (Fixing the Louis C.K. duration/rating swap)
    # Using a combined logical mask natively interpreted by NumPy arrays
    swap_condition = df['duration'].isnull() & df['rating'].str.contains('min', na=False)
    
    if swap_condition.any():
        print(f"[FIX] Using np.where to correct {swap_condition.sum()} swapped duration/rating values...")
        # np.where(condition, value_if_true, value_if_false)
        df['duration'] = np.where(swap_condition, df['rating'], df['duration'])
        df['rating'] = np.where(swap_condition, 'Unknown', df['rating'])

    # 3. Handle Remaining Missing Values using Pandas fillna
    print("[CLEAN] Mass-imputing remaining missing values with 'Unknown'...")
    df.fillna('Unknown', inplace=True)

    # 4. Remove Duplicates using Pandas
    duplicate_count = df.duplicated().sum()
    print(f"[CLEAN] Total duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)

    # 5. Normalize Categorical Features (Stripping trailing/leading whitespaces)
    print("[NORMALIZE] Standardizing categorical text fields...")
    for col in ['type', 'title', 'rating']:
        df[col] = df[col].str.strip()

    # 6. Extract Multi-valued Categorical Attributes via Pandas & NumPy
    # .explode() unflattens the comma-separated lists, and np.unique extracts array values
    all_genres = np.unique(df['listed_in'].str.split(',').explode().str.strip())
    all_countries = np.unique(df['country'].str.split(',').explode().str.strip())

    # 7. Print pipeline execution telemetry
    print("\n=== Data Normalization Summary ===")
    print(f"• Final Cleaned Dataset Shape: {df.shape}")
    print(f"• Unique Content Types: {list(df['type'].unique())}")
    print(f"• Total Unique Normalized Genres Found (NumPy unique): {len(all_genres)}")
    print(f"• Total Unique Countries Represented (NumPy unique): {len(all_countries)}")
    print(f"• Remaining Missing (Null) Values:\n{df.isnull().sum().to_string()}")

    # 8. Save the cleaned dataset to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Pipeline executed successfully. Data saved to: '{output_path}'")
    print("=== Milestone 1 Pipeline Complete ===")
    
    return df

# Execute the integrated script
if __name__ == "__main__":
    cleaned_df = run_milestone_1_pipeline()
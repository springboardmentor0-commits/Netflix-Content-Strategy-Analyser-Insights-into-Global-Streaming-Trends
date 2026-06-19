import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------
# 0. Load Dataset
# -------------------------------------------------------------
df = pd.read_csv('netflix_titles_cleaned.csv')

# -------------------------------------------------------------
# Part 1: Exploratory Data Analysis (EDA) Visualizations
# -------------------------------------------------------------

# Step 1: Analyze Netflix Content Growth Over Time
growth = df['release_year'].value_counts().sort_index()
plt.clf()
plt.plot(growth.index, growth.values, marker='o', color='#E50914', linewidth=2)
plt.title('Netflix Content Growth Over Time (Release Year)', fontsize=14, pad=15)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.xlim(1980, 2022)  # Focus on the era of rapid digital growth
plt.tight_layout()
plt.savefig('content_growth.png')

# Step 2: Analyze Movies vs TV Shows
type_counts = df['type'].value_counts()
plt.clf()
sns.barplot(x=type_counts.index, y=type_counts.values, palette='Reds_r')
plt.title('Netflix Content Type Comparison: Movies vs TV Shows', fontsize=14, pad=15)
plt.xlabel('Content Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('movies_vs_tvshows.png')

genre_counts = df['listed_in'].value_counts().head(10)

genre_counts = df['listed_in'].value_counts().head(10)
plt.clf()
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='Oranges_r')
plt.title('Top 10 Genres on Netflix', fontsize=14, pad=15)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.tight_layout()
plt.savefig('top_genres.png')

# Step 4: Analyze Ratings (Target Audience Distribution)
rating_counts = df['rating'].value_counts().head(10)
plt.clf()
sns.barplot(x=rating_counts.values, y=rating_counts.index, palette='Purples_r')
plt.title('Top 10 Age Rating Distribution', fontsize=14, pad=15)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.tight_layout()
plt.savefig('rating_distribution.png')

# Step 5: Analyze Country Contributions
country_counts = df['country'].value_counts().head(10)
plt.clf()
sns.barplot(x=country_counts.values, y=country_counts.index, palette='Blues_r')
plt.title('Top 10 Country Contributions', fontsize=14, pad=15)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()
plt.savefig('country_contributions.png')

# Step 6: Analyze Movie Duration
movies_df = df[df['type'] == 'Movie'].copy()
movies_df['duration_min'] = movies_df['duration'].str.replace(' min', '', case=False, regex=True).fillna(0).astype(int)

plt.clf()
sns.histplot(movies_df['duration_min'], bins=30, kde=True, color='#831010')
plt.title('Movie Duration Distribution (Minutes)', fontsize=14, pad=15)
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('movie_duration_hist.png')


# -------------------------------------------------------------
# Part 2: Feature Engineering
# -------------------------------------------------------------

# Step 7: Create Content Length Category
def categorize_duration(row):
    if row['type'] == 'Movie':
        try:
            val = int(str(row['duration']).replace(' min', '').strip())
            if val < 60:
                return 'Short Movie'
            elif val <= 120:
                return 'Medium Movie'
            else:
                return 'Long Movie'
        except:
            return 'Unknown Movie Length'
    else:  # TV Show
        try:
            val = int(str(row['duration']).split()[0].strip())
            if val <= 2:
                return 'Short-running TV Show'
            else:
                return 'Long-running TV Show'
        except:
            return 'Unknown TV Show Length'

df['duration_category'] = df.apply(categorize_duration, axis=1)

# Step 8: Extract Date Features & Cast Count
df['date_added'] = df['date_added'].str.strip()
df['year_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year.fillna(0).astype(int)
df['month_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.month_name().fillna('Unknown')
df['cast_count'] = df['cast'].apply(lambda x: 0 if x == 'Unknown' else len(str(x).split(',')))

# Save the newly engineered dataset to a CSV file
df.to_csv('netflix_titles_featured.csv', index=False)
print("Milestone 2 Completed Successfully. 'netflix_titles_featured.csv' saved.")
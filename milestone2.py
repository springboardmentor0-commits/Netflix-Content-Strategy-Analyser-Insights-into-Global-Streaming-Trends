# ==============================
# NETFLIX DATA ANALYSIS PROJECT
# MILESTONE 2: EDA + FEATURE ENGINEERING
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("netflix_titles_cleaned.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ------------------------------
# 1. Content Growth Over Time
# ------------------------------
plt.figure(figsize=(12,6))
df['release_year'].value_counts().sort_index().plot(kind='line')
plt.title("Netflix Content Growth Over Time")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.grid(True)
plt.show()

# ------------------------------
# 2. Movies vs TV Shows
# ------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='type')
plt.title("Movies vs TV Shows")
plt.show()

# Pie Chart
df['type'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6)
)
plt.title("Content Distribution")
plt.ylabel("")
plt.show()

# ------------------------------
# 3. Top 10 Genres
# ------------------------------
genres = df['listed_in'].str.split(', ')

all_genres = []

for g in genres:
    all_genres.extend(g)

genre_counts = pd.Series(all_genres).value_counts()

plt.figure(figsize=(12,6))
genre_counts.head(10).plot(kind='bar')
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# ------------------------------
# 4. Rating Distribution
# ------------------------------
plt.figure(figsize=(10,5))
sns.countplot(
    data=df,
    x='rating',
    order=df['rating'].value_counts().index
)
plt.title("Rating Distribution")
plt.xticks(rotation=45)
plt.show()

# ------------------------------
# 5. Top Contributing Countries
# ------------------------------
countries = df['country'].dropna().str.split(', ')

all_countries = []

for c in countries:
    all_countries.extend(c)

country_counts = pd.Series(all_countries).value_counts()

plt.figure(figsize=(12,6))
country_counts.head(10).plot(kind='bar')
plt.title("Top 10 Contributing Countries")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.show()

# ------------------------------
# 6. Duration Analysis
# ------------------------------
movies = df[df['type'] == 'Movie'].copy()

movies['duration_num'] = movies['duration'].str.extract('(\d+)').astype(float)

# Histogram
plt.figure(figsize=(10,5))
sns.histplot(movies['duration_num'], bins=30)
plt.title("Movie Duration Distribution")
plt.xlabel("Duration (minutes)")
plt.show()

# Boxplot
plt.figure(figsize=(8,4))
sns.boxplot(x=movies['duration_num'])
plt.title("Movie Duration Boxplot")
plt.show()

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------

# Feature 1: Content Length Category
def classify_duration(x):
    if x < 60:
        return "Short"
    elif x <= 120:
        return "Medium"
    else:
        return "Long"

movies['length_category'] = movies['duration_num'].apply(classify_duration)

print("\nLength Category Counts:")
print(movies['length_category'].value_counts())

# Feature 2: Decade Feature
df['decade'] = (df['release_year'] // 10) * 10

print("\nDecade Distribution:")
print(df['decade'].value_counts())

# Feature 3: Content Age
current_year = 2026
df['content_age'] = current_year - df['release_year']

print("\nContent Age Statistics:")
print(df['content_age'].describe())

# ------------------------------
# Summary Statistics
# ------------------------------
print("\nMovies vs TV Shows")
print(df['type'].value_counts())

print("\nTop 10 Genres")
print(genre_counts.head(10))

print("\nTop 10 Countries")
print(country_counts.head(10))

print("\nRatings")
print(df['rating'].value_counts())

print("\nMovie Duration Statistics")
print(movies['duration_num'].describe())

print("\nAnalysis Complete!")
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_netflix_titles.csv")

print(df.head())
print(df.shape)


df['type'].value_counts().plot(kind='bar')

plt.title('Movies vs TV Shows')
plt.xlabel('Content Type')
plt.ylabel('Count')

plt.show()

top_countries = df['country'].value_counts().head(10)

top_countries.plot(kind='bar')

plt.title('Top 10 Countries')
plt.ylabel('Number of Titles')

plt.show()

genres = df['listed_in'].str.split(', ').explode()

genres.value_counts().head(10).plot(kind='bar')

plt.title('Top 10 Genres')
plt.ylabel('Count')

plt.show()

df['date_added'] = pd.to_datetime(df['date_added'])

growth = df['date_added'].dt.year.value_counts().sort_index()

growth.plot(kind='line', marker='o')

plt.title('Netflix Content Growth')
plt.xlabel('Year')
plt.ylabel('Titles Added')

plt.show()


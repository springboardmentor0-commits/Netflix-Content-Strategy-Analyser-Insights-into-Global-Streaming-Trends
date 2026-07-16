import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("netflix_titles_cleaned.csv")

# Keep required columns
# Keep required columns
df = df[['type', 'country', 'listed_in', 'duration']]

# Remove missing values
df = df.dropna()

# Convert duration to number
df['duration'] = df['duration'].str.extract(r'(\d+)')
df = df.dropna(subset=['duration'])
df['duration'] = df['duration'].astype(int)

# Encode categorical columns
country_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df['country'] = country_encoder.fit_transform(df['country'])
df['listed_in'] = genre_encoder.fit_transform(df['listed_in'])

X = df[['country', 'listed_in', 'duration']]
y = df['type']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(country_encoder, open("country_encoder.pkl", "wb"))
pickle.dump(genre_encoder, open("genre_encoder.pkl", "wb"))

print("Model saved successfully!")
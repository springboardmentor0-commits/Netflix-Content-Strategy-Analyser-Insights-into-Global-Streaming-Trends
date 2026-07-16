import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("netflix_titles_cleaned.csv")

# Keep required columns
df = pd.read_csv("netflix_titles_cleaned.csv")

# Only remove rows where these columns are missing
df = df.dropna(subset=["type", "country", "listed_in", "duration","release_year"])

# Convert duration to number
# Convert duration to numbers safely
df['duration'] = df['duration'].str.extract(r'(\d+)')
df = df.dropna(subset=['duration'])
df['duration'] = df['duration'].astype(int)

# Encode
country_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df["country_encoded"] = country_encoder.fit_transform(df["country"])
df["genre_encoded"] = genre_encoder.fit_transform(df["listed_in"])

X = df[["country_encoded", "genre_encoded", "duration", "release_year"]]

# ---------- Classification ----------
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X, df['type'])

# ---------- Clustering ----------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Save updated dataset
df.to_csv("netflix_feature_engineered.csv", index=False)
print(df.columns)
print(df.head())
# Save models
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(kmeans, open("cluster_model.pkl", "wb"))
pickle.dump(country_encoder, open("country_encoder.pkl", "wb"))
pickle.dump(genre_encoder, open("genre_encoder.pkl", "wb"))

print("Models saved successfully.")
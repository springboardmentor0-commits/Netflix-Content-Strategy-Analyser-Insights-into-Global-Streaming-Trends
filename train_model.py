import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("netflix_titles_cleaned.csv")

# Remove missing values
df = df.dropna(subset=["type", "country", "listed_in", "release_year"])

# Keep only the first country and genre
df["country"] = df["country"].apply(lambda x: x.split(",")[0].strip())
df["listed_in"] = df["listed_in"].apply(lambda x: x.split(",")[0].strip())

# Encode features
country_encoder = LabelEncoder()
genre_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["country"] = country_encoder.fit_transform(df["country"])
df["listed_in"] = genre_encoder.fit_transform(df["listed_in"])
df["type"] = target_encoder.fit_transform(df["type"])

# Features and target
df["duration"] = (
    df["duration"]
    .str.extract(r"(\d+)")
    .astype(int)
)
X = df[["duration", "country", "listed_in"]]
y = df["type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
classifier.fit(X_train, y_train)

# Save files
joblib.dump(classifier, "model.pkl")
joblib.dump(country_encoder, "country_encoder.pkl")
joblib.dump(genre_encoder, "genre_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Training completed.")
print("Saved:")
print("- model.pkl")
print("- country_encoder.pkl")
print("- genre_encoder.pkl")
print("- target_encoder.pkl")
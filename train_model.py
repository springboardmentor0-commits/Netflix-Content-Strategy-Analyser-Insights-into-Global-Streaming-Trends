# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("netflix_cleaned.csv")

# Drop missing values
df = df.dropna()

# Create duration_num
df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)

# Create country_count
df["country_count"] = df["country"].fillna("").apply(
    lambda x: len(x.split(","))
)

# Create genre_count
df["genre_count"] = df["listed_in"].fillna("").apply(
    lambda x: len(x.split(","))
)

# Target column
target = "type"      # Movie / TV Show

# Features
features = ["release_year", "duration_num", "country_count", "genre_count"]

X = df[features]
y = df[target]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train classifier
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(classifier, "classifier.pkl")

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

print("classifier.pkl saved successfully!")
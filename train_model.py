import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("netflix_cleaned.csv")

# Remove unwanted column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Fill missing values
df.fillna("Unknown", inplace=True)

# Convert duration to numeric
df["duration_num"] = (
    df["duration"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

# Keep required columns
data = df[[
    "country",
    "listed_in",
    "duration_num",
    "release_year",
    "type"
]].dropna()

# Encode categorical columns
country_encoder = LabelEncoder()
genre_encoder = LabelEncoder()
type_encoder = LabelEncoder()

data["country"] = country_encoder.fit_transform(data["country"])
data["listed_in"] = genre_encoder.fit_transform(data["listed_in"])
data["type"] = type_encoder.fit_transform(data["type"])

# Features and target
X = data[[
    "country",
    "listed_in",
    "duration_num",
    "release_year"
]]

y = data["type"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "netflix_model.pkl")
joblib.dump(country_encoder, "country_encoder.pkl")
joblib.dump(genre_encoder, "genre_encoder.pkl")
joblib.dump(type_encoder, "type_encoder.pkl")

print("Model trained successfully!")
# ==========================================
# MILESTONE 3: MODELING & ADVANCED ANALYSIS (UPDATED)
# ==========================================

# ------------------------------------------
# 1. Import Libraries
# ------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # Added for better visualizations
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------
# 2. Load Dataset
# ------------------------------------------
file_path = "netflix_titles_cleaned.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!")
    print(f"Dataset Shape: {df.shape}")
except FileNotFoundError:
    print(
        f"Error: The file '{file_path}' was not found. Please check the path."
    )
    exit()

# ------------------------------------------
# 3. Data Preprocessing & Cleaning
# ------------------------------------------
data = df.copy()

# Handle potential missing values to prevent ML model crashes
data["rating"] = data["rating"].fillna("Unknown")
data["duration"] = data["duration"].fillna("0 min")
data["country"] = data["country"].fillna("Unknown")
data["listed_in"] = data["listed_in"].fillna("Unknown")

# Encode categorical columns
encoder_rating = LabelEncoder()
encoder_type = LabelEncoder()

data["rating_encoded"] = encoder_rating.fit_transform(data["rating"])
data["type_encoded"] = encoder_type.fit_transform(data["type"])

# Extract numeric duration safely using regex
data["duration_num"] = (
    data["duration"].str.extract(r"(\d+)").astype(float).fillna(0)
)

# Clean release year for time-series plotting
data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")

print("\n--- Preprocessed Data Sample ---")
print(data[["type", "rating", "duration", "duration_num"]].head())

# ==========================================
# NEW PICTURE 1: PIE CHART OF CONTENT TYPES
# ==========================================
plt.figure(figsize=(6, 6))
data["type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["#ff9999", "#66b3ff"],
    explode=(0.05, 0),
    startangle=90,
)
plt.title("Distribution of Netflix Content: Movies vs TV Shows")
plt.ylabel("")  # Hide the default y-label
plt.tight_layout()
plt.show()

# ==========================================
# NEW PICTURE 2: LINE CHART OF RELEASES OVER TIME
# ==========================================
plt.figure(figsize=(10, 5))
release_trends = (
    data.groupby(["release_year", "type"]).size().unstack().fillna(0)
)
# Filtering from 2000 onwards for better visual clarity
release_trends.loc[2000:].plot(kind="line", marker="o", ax=plt.gca())
plt.title("Netflix Content Release Trends (Since 2000)")
plt.xlabel("Release Year")
plt.ylabel("Number of Accumulative Releases")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ==========================================
# NEW PICTURE 3: HISTOGRAM OF CONTENT DURATION
# ==========================================
plt.figure(figsize=(10, 5))
sns.histplot(
    data=data,
    x="duration_num",
    hue="type",
    kde=True,
    bins=30,
    element="step",
    stat="count",
)
plt.title("Distribution of Content Durations (Minutes for Movies / Seasons for TV Shows)")
plt.xlabel("Duration Value")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 4. Clustering Netflix Titles (KMeans)
# ------------------------------------------
X_cluster = data[["duration_num", "rating_encoded"]]

# Scale Data (Crucial for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data["cluster"] = kmeans.fit_predict(X_scaled)

print("\n--- Cluster Assignments Sample ---")
print(data[["title", "cluster"]].head(10))

# ------------------------------------------
# 5. Cluster Visualization (PICTURE 4)
# ------------------------------------------
plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    data["duration_num"],
    data["rating_encoded"],
    c=data["cluster"],
    cmap="viridis",
    alpha=0.6,
)
plt.colorbar(scatter, label="Cluster ID")
plt.xlabel("Duration (Minutes or Seasons)")
plt.ylabel("Rating (Encoded)")
plt.title("Netflix Content Clustering")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 6. Classification Model (Predict Movie vs TV Show)
# ------------------------------------------
features = ["duration_num", "rating_encoded"]
X = data[features]
y = data["type_encoded"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(f"\nClassification Accuracy: {accuracy:.4f}")

# ------------------------------------------
# 7. Classification Report
# ------------------------------------------
print("\n--- Classification Report ---")
print(
    classification_report(
        y_test, pred, target_names=encoder_type.classes_
    )
)

# ==========================================
# NEW PICTURE 5: CONFUSION MATRIX HEATMAP
# ==========================================
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder_type.classes_,
    yticklabels=encoder_type.classes_,
)
plt.title("Classification Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 8. Country Analysis (PICTURE 6)
# ------------------------------------------
country_count = data["country"].value_counts().head(10)

print("\n--- Top Countries Producing Content ---")
print(country_count)

plt.figure(figsize=(10, 5))
country_count.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Top Countries Producing Netflix Content")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------
# 9. Genre Analysis (PICTURE 7)
# ------------------------------------------
genre_count = data["listed_in"].value_counts().head(10)

print("\n--- Top Genres ---")
print(genre_count)

plt.figure(figsize=(10, 5))
genre_count.plot(kind="bar", color="salmon", edgecolor="black")
plt.title("Top Netflix Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------
# 10. Feature Importance (PICTURE 8)
# ------------------------------------------
importance = model.feature_importances_
feature_df = pd.DataFrame(
    {"Feature": features, "Importance": importance}
).sort_values(by="Importance", ascending=False)

print("\n--- Feature Importance ---")
print(feature_df)

# Plot Feature Importance
plt.figure(figsize=(6, 4))
plt.bar(
    feature_df["Feature"],
    feature_df["Importance"],
    color="purple",
    edgecolor="black",
    width=0.4,
)
plt.title("Feature Importance for Classification")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 11. Final Insights Summary
# ------------------------------------------
print("\n" + "=" * 40)
print("           FINAL INSIGHTS")
print("=" * 40)
print(f"1. Total Visualizations Generated: 8 pictures.")
print("2. Content successfully grouped into 4 distinct clusters.")
print("3. Random Forest used to classify 'Movie' vs 'TV Show'.")
print(f"4. Model Accuracy Score: {accuracy:.2%}")
print(f"5. Top content producing country: {country_count.index[0]}")
print(f"6. Most common genre category: {genre_count.index[0]}")
print(
    f"7. Primary driving feature: {feature_df.iloc[0]['Feature']} ({feature_df.iloc[0]['Importance']:.2%})"
)
print("=" * 40)
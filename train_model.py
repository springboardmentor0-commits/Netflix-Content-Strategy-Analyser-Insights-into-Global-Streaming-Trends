import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("cleaned_netflix_data.csv")

# ==============================
# ENCODE CATEGORICAL COLUMNS
# ==============================
country_encoder = LabelEncoder()
genre_encoder = LabelEncoder()
type_encoder = LabelEncoder()

df["country"] = df["country"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")

df["country_encoded"] = country_encoder.fit_transform(df["country"])
df["genre_encoded"] = genre_encoder.fit_transform(df["listed_in"])
df["type_encoded"] = type_encoder.fit_transform(df["type"])

# ==============================
# CREATE DURATION FEATURE
# ==============================
def extract_minutes(x):
    try:
        return int(str(x).split()[0])
    except:
        return 0

df["duration_minutes"] = df["duration"].apply(extract_minutes)

# ==============================
# FEATURES
# ==============================
features = [
    "release_year",
    "country_encoded",
    "genre_encoded",
    "duration_minutes"
]

X = df[features]

# ==============================
# SCALE FEATURES
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ==============================
# KMEANS CLUSTERING
# ==============================
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

joblib.dump(kmeans, "cluster_model.pkl")

# ==============================
# PCA VISUALIZATION
# ==============================
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    components[:,0],
    components[:,1],
    c=clusters,
    cmap="viridis"
)

plt.title("Netflix Clusters")
plt.savefig("clusters_pca.png")
plt.close()

# ==============================
# RANDOM FOREST CLASSIFIER
# ==============================
y = df["type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

classifier = RandomForestClassifier(
    random_state=42
)

classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

print(classification_report(
    y_test,
    prediction
))

cm = confusion_matrix(
    y_test,
    prediction
)

plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("confusion_matrix.png")
plt.close()

joblib.dump(classifier, "classification_model.pkl")

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = classifier.feature_importances_

plt.figure(figsize=(8,5))
plt.bar
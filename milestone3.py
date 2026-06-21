# ============================================================
# NETFLIX DATASET - MILESTONE 3
# Modeling & Advanced Analysis
# ============================================================

# ----------------------------
# Import Libraries
# ----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("cleaned_netflix_titles.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ============================================================
# PART 1: PREPROCESSING
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

# Create copy
data = df.copy()

# Label Encoding
label_encoders = {}

for col in data.select_dtypes(include='object').columns:
    
    data[col] = data[col].astype(str)

    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

    label_encoders[col] = le

print("\nData Encoded Successfully")

# ============================================================
# PART 2: CLUSTERING
# Group Netflix Titles Based on:
# Genre, Duration, Rating
# ============================================================

print("\n====================")
print("CLUSTERING")
print("====================")

# Select Features

cluster_features = []

for col in data.columns:
    if 'genre' in col.lower():
        cluster_features.append(col)

if 'duration' in data.columns:
    cluster_features.append('duration')

if 'rating' in data.columns:
    cluster_features.append('rating')

# Fallback

if len(cluster_features) < 3:
    
    cluster_features = data.columns[:5].tolist()

print("Features Used:", cluster_features)

X_cluster = data[cluster_features]

# Scale Data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow Method

inertia = []

for k in range(1,11):
    
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    km.fit(X_scaled)
    
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# KMeans

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters

print("\nCluster Counts:")
print(data['Cluster'].value_counts())

# Cluster Visualization

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=X_scaled[:,0],
    y=X_scaled[:,1],
    hue=clusters,
    palette='Set1'
)

plt.title("Netflix Title Clusters")
plt.show()

# ============================================================
# PART 3: MOVIE VS TV SHOW CLASSIFICATION
# ============================================================

print("\n====================")
print("CLASSIFICATION")
print("====================")

# Find Type Column

type_column = None

for col in df.columns:
    if col.lower() == "type":
        type_column = col
        break

if type_column is None:
    print("Type column not found!")
else:

    target = data[type_column]

    features = data.drop(columns=[type_column])

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# ============================================================
# PART 4: FEATURE IMPORTANCE
# ============================================================

print("\n====================")
print("FEATURE IMPORTANCE")
print("====================")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(15))

plt.figure(figsize=(12,6))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features")
plt.show()

# ============================================================
# PART 5: COUNTRY ANALYSIS
# Key Drivers of Content Availability
# ============================================================

print("\n====================")
print("COUNTRY ANALYSIS")
print("====================")

country_col = None

for col in df.columns:
    if "country" in col.lower():
        country_col = col
        break

if country_col:

    top_countries = (
        df[country_col]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(12,6))

    sns.barplot(
        x=top_countries.values,
        y=top_countries.index
    )

    plt.title("Top Countries by Netflix Content")
    plt.xlabel("Count")

    plt.show()

# ============================================================
# PART 6: GENRE ANALYSIS
# ============================================================

genre_col = None

for col in df.columns:
    if "genre" in col.lower():
        genre_col = col
        break

if genre_col:

    top_genres = (
        df[genre_col]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(12,6))

    sns.barplot(
        x=top_genres.values,
        y=top_genres.index
    )

    plt.title("Top Genres on Netflix")
    plt.xlabel("Count")

    plt.show()

# ============================================================
# PART 7: CLUSTER DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x='Cluster',
    data=data,
    palette='Set2'
)

plt.title("Cluster Distribution")

plt.show()

# ============================================================
# PART 8: CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(14,10))

corr = data.corr()

sns.heatmap(
    corr,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================================
# PART 9: SAVE OUTPUT
# ============================================================

data.to_csv(
    "Netflix_Milestone3_Output.csv",
    index=False
)

print("\nAnalysis Completed Successfully!")
print("Output Saved: Netflix_Milestone3_Output.csv")
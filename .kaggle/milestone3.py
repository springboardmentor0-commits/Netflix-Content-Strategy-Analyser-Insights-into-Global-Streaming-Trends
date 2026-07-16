import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
df = pd.read_csv("cleaned_netflix_titles.csv")

print(df.head())
print(df.info())
df["duration_num"] = (
    df["duration"]
      .astype(str)
      .str.extract(r'(\d+)')
      .astype(float)
)

df["duration_num"] = df["duration_num"].fillna(
    df["duration_num"].median()
)
encoder = LabelEncoder()


categorical = [
    "country",
    "rating",
    "listed_in",
    "content_length_category",
    "type"
]
for col in categorical:
    df[col] = encoder.fit_transform(
    df[col].astype(str)
    )
features = [
    "release_year",
    "duration_num",
    "country",
    "rating",
    "listed_in",
    "content_length_category",
    "type"
]
    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )
    features = [
    "release_year",
    "duration_num",
    "country",
    "rating",
    "listed_in",
    "duration_category"
]

X = df[features]
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(1,11),wcss,marker="o")

plt.title("Elbow Method")

plt.xlabel("Clusters")

plt.ylabel("WCSS")

plt.savefig("elbow_method.png")

plt.show()
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)
joblib.dump(
    kmeans,
    "clustering_model.pkl"
)
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["cluster"],
    cmap="viridis"
)

plt.title("Netflix Clusters")

plt.savefig("clusters_pca.png")

plt.show()
y = df["type"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)
joblib.dump(
    model,
    "classification_model.pkl"
)
pred = model.predict(X_test)
print("Accuracy")

print(
    accuracy_score(
        y_test,
        pred
    )
)
print(classification_report(
    y_test,
    pred
))
cm = confusion_matrix(
    y_test,
    pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png"
)

plt.show()
importance = pd.DataFrame({

    "Feature":features,

    "Importance":
    model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print(importance)
importance.to_csv(

    "feature_importance.csv",

    index=False

)
plt.figure(figsize=(8,5))

sns.barplot(

    data=importance,

    x="Importance",

    y="Feature"

)

plt.title("Feature Importance")

plt.savefig(

    "feature_importance.png"

)

plt.show()
df.to_csv(

    "netflix_milestone3_output.csv",

    index=False

)
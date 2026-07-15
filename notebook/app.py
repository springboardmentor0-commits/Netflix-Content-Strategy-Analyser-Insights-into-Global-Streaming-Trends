import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Netflix Content Strategy Dashboard",
    page_icon="🎬",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv("dataset/netflix_cleaned.csv")

df = load_data()

# =====================================
# LOAD MODELS
# =====================================
classifier = joblib.load("model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

# =====================================
# PREPARE DATA
# =====================================
df["duration_num"] = (
    df["duration"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("🎬 Netflix Filters")

country_list = sorted(
    set(
        c.strip()
        for row in df["country"].dropna().astype(str)
        for c in row.split(",")
    )
)

selected_countries = st.sidebar.multiselect(
    "Country",
    country_list,
    key="country_filter"
)

genre_list = sorted(
    set(
        g.strip()
        for row in df["listed_in"].dropna().astype(str)
        for g in row.split(",")
    )
)

selected_genres = st.sidebar.multiselect(
    "Genre",
    genre_list,
    key="genre_filter"
)

selected_types = st.sidebar.multiselect(
    "Content Type",
    sorted(df["type"].dropna().unique()),
    key="type_filter"
)

filtered_df = df.copy()

if selected_countries:
    filtered_df = filtered_df[
        filtered_df["country"].apply(
            lambda x: any(
                c.strip() in selected_countries
                for c in str(x).split(",")
            )
        )
    ]

if selected_genres:
    filtered_df = filtered_df[
        filtered_df["listed_in"].apply(
            lambda x: any(
                g.strip() in selected_genres
                for g in str(x).split(",")
            )
        )
    ]

if selected_types:
    filtered_df = filtered_df[
        filtered_df["type"].isin(selected_types)
    ]
    # =====================================
# TITLE
# =====================================

st.title("🎬 Netflix Content Strategy Dashboard")
st.markdown("### Milestone 4 - Interactive Dashboard")

st.caption(f"Filtered Titles: {len(filtered_df)}")

# =====================================
# KPI CARDS
# =====================================

total_titles = len(filtered_df)
movies = len(filtered_df[filtered_df["type"] == "Movie"])
tv_shows = len(filtered_df[filtered_df["type"] == "TV Show"])

country_count = (
    filtered_df["country"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .nunique()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("📚 Total Titles", total_titles)
col2.metric("🎬 Movies", movies)
col3.metric("📺 TV Shows", tv_shows)
col4.metric("🌍 Countries", country_count)

# =====================================
# TABS
# =====================================

tab1, tab2 = st.tabs(
    ["📊 Strategic Insights", "🤖 AI Analytics"]
)

# =====================================
# TAB 1
# =====================================

with tab1:

    st.subheader("Movies vs TV Shows")

    fig = px.histogram(
        filtered_df,
        x="type",
        color="type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top 10 Countries")

    top_country = (
        filtered_df["country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_country.columns = ["Country", "Titles"]

    fig = px.bar(
        top_country,
        x="Country",
        y="Titles"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top 10 Genres")

    top_genre = (
        filtered_df["listed_in"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_genre.columns = ["Genre", "Titles"]

    fig = px.bar(
        top_genre,
        x="Genre",
        y="Titles"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Netflix Content Strategy Dashboard",
    page_icon="🎬",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv("dataset/netflix_cleaned.csv")

df = load_data()

# =====================================
# LOAD MODELS
# =====================================
classifier = joblib.load("model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

# =====================================
# PREPARE DATA
# =====================================
df["duration_num"] = (
    df["duration"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

# =====================================
# TAB 2 - AI ANALYTICS
# =====================================
with tab2:
     import os

     base_path = os.path.dirname(os.path.dirname(__file__))

cluster_img = os.path.join(base_path, "clusters_pca.png.png")
feature_img = os.path.join(base_path, "feature_importance.png.png")
confusion_img = os.path.join(base_path, "confusion_matrix.png.png")

st.subheader("📊 Cluster Visualization")
st.image(cluster_img, use_container_width=True)

st.subheader("📈 Feature Importance")
st.image(feature_img, use_container_width=True)

st.subheader("📉 Confusion Matrix")
st.image(confusion_img, use_container_width=True)
st.subheader("🎯 Predict Content Type")
country = st.selectbox("Country", country_encoder.classes_)
genre = st.selectbox("Genre", genre_encoder.classes_)
duration = st.number_input("Duration (minutes)", min_value=1, value=90)

if st.button("Predict"):

    country_encoded = country_encoder.transform([country])[0]
    genre_encoded = genre_encoder.transform([genre])[0]

    features = np.array([[country_encoded, genre_encoded, duration]])

    prediction = classifier.predict(features)[0]

    if prediction == 1:
        st.success("Prediction: TV Show")
    else:
        st.success("Prediction: Movie")
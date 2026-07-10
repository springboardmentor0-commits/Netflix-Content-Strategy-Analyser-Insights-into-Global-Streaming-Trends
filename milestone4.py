import os

import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Netflix Content Strategy Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Strategy Dashboard")
st.markdown("### Milestone 4 - Interactive Dashboard")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_netflix_titles.csv")

df = load_data()
#Load the trained model and encoders
model = pickle.load(open("model.pkl", "rb"))
country_encoder = pickle.load(open("country_encoder.pkl", "rb"))
genre_encoder = pickle.load(open("genre_encoder.pkl", "rb"))
# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("Dashboard Filters")

# Country Filter
countries = sorted(df["country"].dropna().unique())
selected_country = st.sidebar.multiselect(
    "Select Country",
    countries,
    default=[]
)

# Genre Filter
genres = sorted(df["listed_in"].dropna().unique())
selected_genre = st.sidebar.multiselect(
    "Select Genre",
    genres,
    default=[]
)

# Content Type Filter
content_type = st.sidebar.multiselect(
    "Content Type",
    df["type"].unique(),
    default=df["type"].unique()
)

# Release Year Filter
year_range = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (
        int(df["release_year"].min()),
        int(df["release_year"].max())
    )
)

# ---------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------
filtered_df = df.copy()

if selected_country:
    filtered_df = filtered_df[
        filtered_df["country"].isin(selected_country)
    ]

if selected_genre:
    filtered_df = filtered_df[
        filtered_df["listed_in"].isin(selected_genre)
    ]

filtered_df = filtered_df[
    filtered_df["type"].isin(content_type)
]

filtered_df = filtered_df[
    (filtered_df["release_year"] >= year_range[0]) &
    (filtered_df["release_year"] <= year_range[1])
]

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2 = st.tabs([
    "📊 Strategic Catalog Insights",
    "🤖 AI Analytics"
])

# ==========================================
# DASHBOARD SUMMARY METRICS
# ==========================================

total_titles = len(filtered_df)
total_movies = len(filtered_df[filtered_df["type"] == "Movie"])
total_tvshows = len(filtered_df[filtered_df["type"] == "Tv Show"])
total_countries = filtered_df["country"].nunique()

m1, m2, m3, m4 = st.columns(4)

m1.metric("🎬 Total Titles", total_titles)
m2.metric("🎥 Movies", total_movies)
m3.metric("📺 TV Shows", total_tvshows)
m4.metric("🌍 Countries", total_countries)
with tab1:

    st.header("📈 Netflix Content Growth Over the Years")

    yearly = (
        filtered_df.groupby("release_year")
        .size()
        .reset_index(name="Count")
    )

    fig = px.line(
        yearly,
        x="release_year",
        y="Count",
        markers=True,
        title="Netflix Content Released by Year"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Movies vs TV Shows")

        type_count = filtered_df["type"].value_counts()

        fig = px.pie(
            values=type_count.values,
            names=type_count.index,
            title="Content Type Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Top 10 Countries")

        country_count = (
            filtered_df["country"]
            .value_counts()
            .head(10)
        )

        fig = px.bar(
            x=country_count.values,
            y=country_count.index,
            orientation="h",
            title="Top 10 Countries"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.subheader("🎭 Top 10 Genres")

genre_count = (
    filtered_df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

fig = px.bar(
    x=genre_count.values,
    y=genre_count.index,
    orientation="h",
    title="Top 10 Genres"
)

st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🤖 AI Analytics")

    st.subheader("Cluster Visualization")
    try:
        st.image("clusters_pca.png", use_container_width=True)
    except:
        st.warning("clusters_pca.png not found")

    st.subheader("Feature Importance")
    try:
        st.image("feature_importance.png", use_container_width=True)
    except:
        st.warning("feature_importance.png not found")

    st.subheader("Confusion Matrix")
    try:
        st.image("confusion_matrix.png", use_container_width=True)
    except:
        st.warning("confusion_matrix.png not found")

# ------------------------------------------------
# Interactive Prediction
# ------------------------------------------------

st.markdown("---")
st.header("🎯 Interactive Prediction")

country = st.selectbox(
    "Country",
    country_encoder.classes_
)

genre = st.selectbox(
    "Genre",
    genre_encoder.classes_
)

duration = st.number_input(
    "Duration (minutes)",
    min_value=1,
    max_value=300,
    value=120
)

if st.button("Predict"):

    country_value = country_encoder.transform([country])[0]
    genre_value = genre_encoder.transform([genre])[0]

    prediction = model.predict(
        [[country_value, genre_value, duration]]
    )[0]

    st.success(f"Predicted Content Type: {prediction}")
# ==========================================
# NETFLIX CONTENT STRATEGY ANALYSER
# MILESTONE 4 - DASHBOARD & DEPLOYMENT
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Netflix Content Strategy Analyser",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Strategy Analyser")
st.subheader("🎯 Interactive Dashboard for Netflix Content Analysis and Machine Learning")
st.markdown("---")


# ------------------------------------------
# LOAD DATA
# ------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("netflix_milestone3_output.csv")

df = load_data()


# ------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------

st.sidebar.header("🎛 Dashboard Filters")

selected_type = st.sidebar.multiselect(
    "Content Type",
    options=df["type"].dropna().unique(),
    default=df["type"].dropna().unique()
)

selected_country = st.sidebar.multiselect(
    "Country",
    options=sorted(df["country"].dropna().unique()),
    default=sorted(df["country"].dropna().unique())
)

selected_genre = st.sidebar.multiselect(
    "Genre",
    options=sorted(df["listed_in"].dropna().unique()),
    default=sorted(df["listed_in"].dropna().unique())
)

selected_year = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (
        int(df["release_year"].min()),
        int(df["release_year"].max())
    )
)


filtered_df = df[
    (df["type"].isin(selected_type)) &
    (df["country"].isin(selected_country)) &
    (df["listed_in"].isin(selected_genre)) &
    (df["release_year"] >= selected_year[0]) &
    (df["release_year"] <= selected_year[1])
]



# ------------------------------------------
# KPI CARDS
# ------------------------------------------

st.markdown("## 📊 Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Titles", len(filtered_df))

with col2:
    st.metric(
        "Movies",
        len(filtered_df[filtered_df["type"] == "Movie"])
    )

with col3:
    st.metric(
        "TV Shows",
        len(filtered_df[filtered_df["type"] == "TV Show"])
    )

with col4:
    st.metric(
        "Countries",
        filtered_df["country"].nunique()
    )

    st.markdown("---")

tab1, tab2 = st.tabs([
    "📊 Strategic Catalog Insights",
    "🤖 Advanced AI Analytics"
])

with tab1:

    st.subheader("Content Type Distribution")

    fig = px.pie(
        filtered_df,
        names="type",
        title="Movies vs TV Shows"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Content Growth by Year")

    year_df = (
        filtered_df
        .groupby("release_year")
        .size()
        .reset_index(name="Count")
    )

    fig = px.line(
        year_df,
        x="release_year",
        y="Count",
        markers=True,
        title="Netflix Content Growth"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Genres")

    genre_df = (
        filtered_df["listed_in"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    genre_df.columns = ["Genre", "Count"]

    fig = px.bar(
        genre_df,
        x="Genre",
        y="Count",
        color="Count",
        title="Top 10 Genres"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Countries")

    country_df = (
        filtered_df["country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    country_df.columns = ["Country", "Count"]

    fig = px.bar(
        country_df,
        x="Country",
        y="Count",
        color="Count",
        title="Top 10 Countries"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("🎯 Interactive Prediction")

    release_year = st.number_input(
    "Release Year",
    min_value=1925,
    max_value=2025,
    value=2020
    )

    duration = st.number_input(
    "Duration (minutes)",
    min_value=1,
    max_value=300,
    value=90
    )

    if st.button("Predict"):
        st.success("Prediction feature is ready.")

    st.header("📋 Filtered Dataset")

    st.dataframe(filtered_df)



# -----------------------------------------
# MACHINE LEARNING ANALYTICS
# -----------------------------------------
with tab2:

    st.header("🤖 Machine Learning Analytics")

    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png", use_container_width=True)

    st.subheader("Feature Importance")
    st.image("feature_importance.png", use_container_width=True)

    st.subheader("Clusters Visualization")
    st.image("clusters_pca.png", use_container_width=True)

    st.subheader("Elbow Method")
    st.image("elbow_method.png", use_container_width=True)










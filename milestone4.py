import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import os

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Netflix Content Strategy Analyser",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Strategy Analyser")
st.markdown("Interactive Dashboard for Netflix Content Analysis and Machine Learning")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflix_cleaned.csv")

df = load_data()


# ----------------------------
# LOAD MODEL
# ----------------------------
model = None
if os.path.exists("classifier.pkl"):
    model = joblib.load("classifier.pkl")

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("Dashboard Filters")

type_filter = st.sidebar.multiselect(
    "Content Type",
    options=df["type"].dropna().unique(),
    default=df["type"].dropna().unique()
)

country_filter = st.sidebar.multiselect(
    "Country",
    options=sorted(df["country"].dropna().unique()),
    default=sorted(df["country"].dropna().unique())
)

genre_filter = st.sidebar.multiselect(
    "Genre",
    options=sorted(df["listed_in"].dropna().unique()),
    default=sorted(df["listed_in"].dropna().unique())
)

year_filter = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (
        int(df["release_year"].min()),
        int(df["release_year"].max())
    )
)

# ----------------------------
# FILTER DATA
# ----------------------------
filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["country"].isin(country_filter)) &
    (df["listed_in"].isin(genre_filter)) &
    (df["release_year"] >= year_filter[0]) &
    (df["release_year"] <= year_filter[1])
]

# ----------------------------
# TABS
# ----------------------------
tab1, tab2 = st.tabs([
    "📊 Strategic Catalog Insights",
    "🤖 AI Analytics"
])

# =====================================================
# TAB 1
# =====================================================
with tab1:

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Titles", len(filtered_df))

    if "country" in filtered_df.columns:
        col2.metric("Countries", filtered_df["country"].nunique())

    if "listed_in" in filtered_df.columns:
        col3.metric("Genres", filtered_df["listed_in"].nunique())

    st.divider()

    st.subheader("Content Type")

    fig = px.pie(
        filtered_df,
        names="type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Titles Released per Year")

    year_data = (
        filtered_df.groupby("release_year")
        .size()
        .reset_index(name="Count")
    )

    fig = px.line(
        year_data,
        x="release_year",
        y="Count",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Genres")

    genre_data = (
        filtered_df["listed_in"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    genre_data.columns = ["Genre", "Count"]

    fig = px.bar(
        genre_data,
        x="Genre",
        y="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Countries")

    country_data = (
        filtered_df["country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    country_data.columns = ["Country", "Count"]

    fig = px.bar(
        country_data,
        x="Country",
        y="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2
# =====================================================
with tab2:

    st.header("Machine Learning Analytics")

    if os.path.exists("clusters_pca.png"):
        st.subheader("Cluster Visualization")
        st.image("clusters_pca.png")

    if os.path.exists("feature_importance.png"):
        st.subheader("Feature Importance")
        st.image("feature_importance.png")

    if os.path.exists("confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("confusion_matrix.png")

    st.divider()

    st.subheader("Interactive Prediction")

    release_year = st.number_input(
        "Release Year",
        1950,
        2035,
        2020
    )

    duration = st.number_input(
        "Duration (minutes)",
        30,
        300,
        120
    )

if st.button("Predict Content Type"):

 if model is None:

    st.error("classifier.pkl not found.")

else:
    country_count = 1
    genre_count = 1

    sample = np.array([[release_year, duration, country_count, genre_count]])

    prediction = model.predict(sample)

    if prediction[0] == 0:
        result = "Movie"
    else:
        result = "TV Show"

    st.success(f"🎬 Predicted Content Type: {result}")

st.divider()

st.subheader("Filtered Dataset")

st.dataframe(filtered_df)
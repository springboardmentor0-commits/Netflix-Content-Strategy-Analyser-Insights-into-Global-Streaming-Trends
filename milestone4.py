import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Netflix Content Strategy Analyser",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Strategy Analyser")
st.markdown("Interactive dashboard for Netflix catalog insights and AI analytics.")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("netflix_cleaned.csv")

df = load_data()

# ==========================================
# LOAD MODEL
# ==========================================
try:
    model = joblib.load("classification_model.pkl")
except:
    model = None

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("Dashboard Filters")

country = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].dropna().unique())
)

genre = st.sidebar.multiselect(
    "Genre",
    sorted(df["listed_in"].dropna().unique())
)

content = st.sidebar.multiselect(
    "Content Type",
    sorted(df["type"].unique())
)

year = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (
        int(df["release_year"].min()),
        int(df["release_year"].max())
    )
)

filtered = df.copy()

if country:
    filtered = filtered[filtered["country"].isin(country)]

if genre:
    filtered = filtered[filtered["listed_in"].isin(genre)]

if content:
    filtered = filtered[filtered["type"].isin(content)]

filtered = filtered[
    (filtered["release_year"] >= year[0]) &
    (filtered["release_year"] <= year[1])
]

# ==========================================
# TABS
# ==========================================
tab1, tab2 = st.tabs([
    "📊 Strategic Catalog Insights",
    "🤖 Advanced AI Analytics"
])

# ==========================================
# TAB 1
# ==========================================
with tab1:

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Titles", len(filtered))
    c2.metric("Countries", filtered["country"].nunique())
    c3.metric("Genres", filtered["listed_in"].nunique())

    st.divider()

    st.subheader("Content Distribution")

    fig = px.histogram(
        filtered,
        x="type",
        color="type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Release Trend")

    trend = (
        filtered.groupby("release_year")
        .size()
        .reset_index(name="Count")
    )

    fig2 = px.line(
        trend,
        x="release_year",
        y="Count",
        markers=True
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top Genres")

    top = (
        filtered["listed_in"]
        .value_counts()
        .head(10)
    )

    fig3 = px.bar(
        x=top.values,
        y=top.index,
        orientation="h"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# TAB 2
# ==========================================
with tab2:

    st.subheader("Cluster Visualization")

    if os.path.exists("clusters_pca.png"):
        st.image("clusters_pca.png")

    st.subheader("Feature Importance")

    if os.path.exists("feature_importance.png"):
        st.image("feature_importance.png")

    st.subheader("Confusion Matrix")

    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png")

    st.divider()

    st.subheader("🎯 Interactive Prediction")

input_country = st.selectbox(
    "Country",
    sorted(df["country"].dropna().unique())
)

input_genre = st.selectbox(
    "Genre",
    sorted(df["listed_in"].dropna().unique())
)

duration = st.number_input(
    "Duration (minutes)",
    min_value=30,
    max_value=300,
    value=90
)

release_year = st.number_input(
    "Release Year",
    min_value=1925,
    max_value=2025,
    value=2020
)

if st.button("Predict Content Type"):

    if model is None:
        st.error("❌ Model file not found!")

    else:

        try:
            country_code = abs(hash(input_country)) % 100
            genre_code = abs(hash(input_genre)) % 100

            sample = np.array([[
                release_year,
                country_code,
                genre_code,
                duration
            ]])

            prediction = model.predict(sample)[0]

            if prediction == 0:
                st.success("🎬 Predicted Content Type: Movie")
            else:
                st.success("📺 Predicted Content Type: TV Show")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
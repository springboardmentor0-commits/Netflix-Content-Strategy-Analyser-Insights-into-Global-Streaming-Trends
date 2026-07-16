import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import joblib

classifier = joblib.load("model.pkl")
cluster_model = joblib.load("cluster_model.pkl")

country_encoder = joblib.load("country_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("netflix_feature_engineered.csv")

df = load_data()

st.title("🎬 Netflix Analytics Dashboard")
st.markdown("Milestone 4 - Dashboard, Integration & Deployment")

# ---------------- Sidebar ---------------- #

st.sidebar.header("Filters")

country = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].dropna().unique())
)

genre = st.sidebar.multiselect(
    "Genre",
    sorted(df["listed_in"].dropna().unique())
)

content_type = st.sidebar.multiselect(
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

filtered_df = df.copy()

if country:
    filtered_df = filtered_df[filtered_df["country"].isin(country)]

if genre:
    filtered_df = filtered_df[filtered_df["listed_in"].isin(genre)]

if content_type:
    filtered_df = filtered_df[filtered_df["type"].isin(content_type)]

filtered_df = filtered_df[
    (filtered_df["release_year"] >= year[0]) &
    (filtered_df["release_year"] <= year[1])
]

tab1, tab2 = st.tabs(["📊Strategic CatalogInsights", "🤖 AI Analytics"])

with tab1:

    st.subheader("Content Distribution")

    fig = px.pie(
        filtered_df,
        names="type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Content Growth")

    growth = filtered_df.groupby("release_year").size().reset_index(name="Count")

    fig = px.line(
        growth,
        x="release_year",
        y="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Countries")

    top_country = filtered_df["country"].value_counts().head(10)

    fig = px.bar(
        top_country,
        x=top_country.index,
        y=top_country.values
    )

    st.plotly_chart(fig, use_container_width=True)
    cluster_model = joblib.load("cluster_model.pkl")
classifier = joblib.load("model.pkl")
st.subheader("Cluster Distribution")
st.write("columns:",filtered_df.columns)

cluster_counts = filtered_df["Cluster"].value_counts()


fig = px.bar(
    cluster_counts,
    x=cluster_counts.index,
    y=cluster_counts.values
)

st.plotly_chart(fig)
st.subheader("Predict Content Type")

duration = st.number_input("Duration", 30, 300, 120)

country = st.selectbox(
    "Country",
    sorted(df["country"].dropna().unique())
)

genre = st.selectbox(
    "Genre",
    sorted(df["listed_in"].dropna().unique())
)

release_year = st.number_input(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    2020
)

if st.button("Predict"):

    country_encoded = country_encoder.transform([country])[0]
    genre_encoded = genre_encoder.transform([genre])[0]

    sample = [[
        country_encoded,
        genre_encoded,
        duration,
        release_year
    ]]

    prediction = classifier.predict(sample)

    st.success(f"Predicted Type: {prediction[0]}")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

##############################################################
# PAGE CONFIG
##############################################################

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide"
)

##############################################################
# LOAD DATA
##############################################################

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_netflix_titles.csv")

df = load_data()

##############################################################
# LOAD MODELS
##############################################################

try:
    classifier = joblib.load("classification_model.pkl")
except:
    classifier = None

try:
    cluster_model = joblib.load("clustering_model.pkl")
except:
    cluster_model = None

##############################################################
# SIDEBAR
##############################################################

st.sidebar.title("Dashboard Filters")

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
    int(df.release_year.min()),
    int(df.release_year.max()),
    (
        int(df.release_year.min()),
        int(df.release_year.max())
    )
)

##############################################################
# FILTERING
##############################################################

filtered = df.copy()

if country:
    filtered = filtered[filtered["country"].isin(country)]

if genre:
    filtered = filtered[filtered["listed_in"].isin(genre)]

if content:
    filtered = filtered[filtered["type"].isin(content)]

filtered = filtered[
    (filtered.release_year >= year[0]) &
    (filtered.release_year <= year[1])
]

##############################################################
# TITLE
##############################################################

st.title("🎬 Netflix Analytics Dashboard")

st.markdown("---")

##############################################################
# KPI CARDS
##############################################################

c1,c2,c3,c4 = st.columns(4)

c1.metric("Total Titles", len(filtered))
c2.metric("Movies", len(filtered[filtered.type=="Movie"]))
c3.metric("TV Shows", len(filtered[filtered.type=="TV Show"]))
c4.metric("Countries", filtered.country.nunique())

st.markdown("---")

##############################################################
# TABS
##############################################################

tab1,tab2 = st.tabs(["📈 EDA","🤖 Machine Learning"])

##############################################################
# TAB 1
##############################################################

with tab1:

    st.subheader("Content Released Per Year")

    yearly = filtered.groupby("release_year").size().reset_index(name="Titles")

    fig = px.line(
        yearly,
        x="release_year",
        y="Titles",
        markers=True
    )

    st.plotly_chart(fig, width="stretch")

    ##########################################

    st.subheader("Top Genres")

    top_genre = (
        filtered["listed_in"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_genre.columns=["Genre","Count"]

    fig = px.bar(
        top_genre,
        x="Genre",
        y="Count",
        color="Count"
    )

    st.plotly_chart(fig,use_container_width=True)

    ##########################################

    st.subheader("Top Countries")

    top_country = (
        filtered["country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_country.columns=["Country","Titles"]

    fig = px.bar(
        top_country,
        x="Country",
        y="Titles",
        color="Titles"
    )

    st.plotly_chart(fig,use_container_width=True)

##############################################################
# TAB 2
##############################################################

with tab2:

    ##########################################################
    # CLUSTER VISUALIZATION
    ##########################################################

    if "cluster" in df.columns:

        st.subheader("Cluster Distribution")

        fig = px.histogram(
            filtered,
            x="cluster",
            color="cluster"
        )

        st.plotly_chart(fig,use_container_width=True)

        ##########################################

        st.subheader("Genre Word Cloud")

        cluster_id = st.selectbox(
            "Choose Cluster",
            sorted(filtered.cluster.unique())
        )

        text = " ".join(
            filtered[
                filtered.cluster==cluster_id
            ]["listed_in"]
        )

        wordcloud = WordCloud(
            width=900,
            height=400,
            background_color="white"
        ).generate(text)

        fig,ax=plt.subplots(figsize=(12,5))

        ax.imshow(wordcloud)

        ax.axis("off")

        st.pyplot(fig)

    ##########################################################
    # PREDICTOR
    ##########################################################

    st.subheader("Predict Content Type")

    year_input = st.number_input(
        "Release Year",
        1940,
        2035,
        2020
    )

    duration = st.number_input(
        "Duration (minutes)",
        1,
        300,
        120
    )

    original = st.selectbox(
        "Netflix Original",
        [0,1]
    )

    if st.button("Predict"):

        if classifier is None:

            st.warning("Classification model not found.")

        else:

            X = pd.DataFrame({
                "release_year":[year_input],
                "duration":[duration],
                "is_original":[original]
            })

            pred = classifier.predict(X)[0]

            if hasattr(classifier,"predict_proba"):

                prob = classifier.predict_proba(X).max()*100

                st.success(
                    f"Prediction : {pred} ({prob:.2f}% confidence)"
                )

            else:

                st.success(f"Prediction : {pred}")

    ##########################################################
    # FEATURE IMPORTANCE
    ##########################################################

    st.subheader("Feature Importance")

    try:

        fi = pd.read_csv("feature_importance.csv")

        fig = px.bar(
            fi,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(fig,use_container_width=True)

    except:

        st.info("feature_importance.csv not found.")
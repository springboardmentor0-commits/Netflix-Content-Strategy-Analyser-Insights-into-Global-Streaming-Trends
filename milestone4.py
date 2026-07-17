import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    layout="wide"
)

st.title("🎬 Netflix Analytics Dashboard")

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv("netflix_titles_cleaned.csv")

df = load_data()
df["country"] = df["country"].apply(lambda x: x.split(",")[0].strip())
df["listed_in"] = df["listed_in"].apply(lambda x: x.split(",")[0].strip())
# =====================================
# LOAD MODEL & ENCODERS
# =====================================
model = joblib.load("model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# =====================================
# SIDEBAR FILTERS
# =====================================
st.sidebar.header("Filters")

content_type = st.sidebar.multiselect(
    "Content Type",
    df["type"].dropna().unique(),
    default=df["type"].dropna().unique()
)

country = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].dropna().unique()),
    default=sorted(df["country"].dropna().unique())
)

genre = st.sidebar.multiselect(
    "Genre",
    sorted(df["listed_in"].dropna().unique()),
    default=sorted(df["listed_in"].dropna().unique())
)

filtered = df[
    (df["type"].isin(content_type)) &
    (df["country"].isin(country)) &
    (df["listed_in"].isin(genre))
]

# =====================================
# TABS
# =====================================
tab1, tab2 = st.tabs([
    "📊 Strategic Insights",
    "🤖 AI Analytics"
])
with tab1:

    st.subheader("Content Distribution")

    fig = px.histogram(
        filtered,
        x="type",
        color="type",
        title="Movies vs TV Shows"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Titles by Release Year")

    yearly = (
        filtered.groupby("release_year")
        .size()
        .reset_index(name="Count")
    )

    fig2 = px.line(
        yearly,
        x="release_year",
        y="Count",
        markers=True,
        title="Netflix Content Growth Over Time"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Countries")

    top_country = (
        filtered["country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_country.columns = ["Country", "Count"]

    fig3 = px.bar(
        top_country,
        x="Country",
        y="Count",
        color="Count",
        title="Top 10 Countries"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 10 Genres")

    top_genre = (
        filtered["listed_in"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_genre.columns = ["Genre", "Count"]

    fig4 = px.bar(
        top_genre,
        x="Genre",
        y="Count",
        color="Count",
        title="Top 10 Genres"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Rating Distribution")

    if "rating" in filtered.columns:

        fig5 = px.histogram(
            filtered,
            x="rating",
            color="rating",
            title="Ratings Distribution"
        )

        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Dataset Preview")

    st.dataframe(filtered.head(20), use_container_width=True)
    # =====================================
# AI ANALYTICS TAB
# =====================================
with tab2:

    st.subheader("🤖 Predict Content Type")

    # Get unique values from original dataset
    country_list = sorted(df["country"].dropna().unique())
    genre_list = sorted(df["listed_in"].dropna().unique())

    selected_country = st.selectbox(
    "Country",
    sorted(df["country"].dropna().apply(lambda x: x.split(",")[0].strip()).unique())
)
    selected_genre = st.selectbox(
    "Genre",
    sorted(df["listed_in"].dropna().apply(lambda x: x.split(",")[0].strip()).unique())
)
    duration = st.number_input(
        "Duration (minutes)",
        min_value=1,
        max_value=500,
        value=120
    )

if st.button("Predict"):
    try:
        # Encode inputs
        country_value = country_encoder.transform([selected_country])[0]
        genre_value = genre_encoder.transform([selected_genre])[0]

        prediction = model.predict(
            [[duration, country_value, genre_value]]
        )

        result = target_encoder.inverse_transform(prediction)

        st.success("Prediction Completed!")
        st.write("Predicted Content Type:", result[0])

    except Exception as e:
        st.error(e)
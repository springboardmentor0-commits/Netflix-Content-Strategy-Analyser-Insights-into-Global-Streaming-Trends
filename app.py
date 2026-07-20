import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Netflix Content Strategy Analyser",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_cleaned.csv")

    # Remove unwanted column
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Fill missing values
    df.fillna("Unknown", inplace=True)

    return df

df = load_data()
# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("netflix_model.pkl")

country_encoder = joblib.load("country_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")
type_encoder = joblib.load("type_encoder.pkl")

# ==========================================
# TITLE
# ==========================================

st.title("🎬 Netflix Content Strategy Analyser")
tab1, tab2 = st.tabs([
    "📊 Strategic Catalog Insights",
    "🧠 Advanced AI Analytics"
])
with tab1:
    st.markdown("### Interactive Dashboard for Netflix Content Analysis")


# ==========================================
# SIDEBAR FILTERS
# ==========================================

    st.sidebar.header("📊 Dashboard Filters")

# Content Type
    content_type = st.sidebar.multiselect(
    "Content Type",
    options=sorted(df["type"].unique()),
    default=sorted(df["type"].unique())
)

# Country
    country = st.sidebar.multiselect(
    "Country",
    options=sorted(df["country"].unique())
)

# Genre
    genre = st.sidebar.multiselect(
    "Genre",
    options=sorted(df["listed_in"].unique())
)

# Release Year
    year = st.sidebar.slider(
    "Release Year",
    min_value=int(df["release_year"].min()),
    max_value=int(df["release_year"].max()),
    value=(
        int(df["release_year"].min()),
        int(df["release_year"].max())
    )
)
# ==========================================
# FILTER DATASET
# ==========================================

    filtered_df = df.copy()

# Filter by Content Type
    if content_type:
     filtered_df = filtered_df[
        filtered_df["type"].isin(content_type)
    ]

# Filter by Country
    if country:
     filtered_df = filtered_df[
        filtered_df["country"].isin(country)
    ]

# Filter by Genre
    if genre:
     filtered_df = filtered_df[
        filtered_df["listed_in"].apply(
            lambda x: any(g in str(x) for g in genre)
        )
    ]

# Filter by Release Year
    filtered_df = filtered_df[
    filtered_df["release_year"].between(year[0], year[1])
]


# ==========================================
# DISPLAY FILTERED DATASET
# ==========================================
with tab1 :
    st.subheader("📄 Filtered Dataset")

    st.write(f"Total Records: {len(filtered_df)}")

    st.dataframe(filtered_df)
# ==========================================
# KPI CARDS
# ==========================================

    st.subheader("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
     st.metric("🎬 Total Titles", len(filtered_df))

    with col2:
     st.metric(
        "🎥 Movies",
        filtered_df[filtered_df["type"] == "Movie"].shape[0]
    )

    with col3:
     st.metric(
        "📺 TV Shows",
        filtered_df[filtered_df["type"] == "TV Show"].shape[0]
    )

    with col4:
     st.metric(
        "🌍 Countries",
        filtered_df["country"].nunique()
    )
    # ==========================================
# MOVIES VS TV SHOWS
# ==========================================

    import plotly.express as px

    st.subheader("🎬 Movies vs TV Shows")

    type_count = (
    filtered_df["type"]
    .value_counts()
    .reset_index()
)

    type_count.columns = ["Type", "Count"]

    fig = px.bar(
    type_count,
    x="Type",
    y="Count",
    color="Type",
    text="Count",
    title="Movies vs TV Shows"
)

    fig.update_layout(
    xaxis_title="Content Type",
    yaxis_title="Number of Titles"
)

    st.plotly_chart(fig, use_container_width=True)
# ==========================================
# CONTENT RATING DISTRIBUTION
# ==========================================

    st.subheader("📊 Content Rating Distribution")

    rating_count = (
    filtered_df["rating"]
    .value_counts()
    .reset_index()
)

    rating_count.columns = ["Rating", "Count"]

    fig = px.pie(
    rating_count,
    names="Rating",
    values="Count",
    title="Content Rating Distribution",
    hole=0.4
)

    fig.update_traces(textposition="inside", textinfo="percent+label")

    st.plotly_chart(fig, use_container_width=True)
# ==========================================
# TITLES RELEASED PER YEAR
# ==========================================

    st.subheader("📈 Titles Released per Year")

    year_count = (
    filtered_df["release_year"]
    .value_counts()
    .sort_index()
    .reset_index()
)

    year_count.columns = ["Release Year", "Count"]

    fig = px.line(
    year_count,
    x="Release Year",
    y="Count",
    markers=True,
    title="Titles Released per Year"
)

    fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Number of Titles"
)

    st.plotly_chart(fig, use_container_width=True)
# ==========================================
# TOP 10 GENRES
# ==========================================

    st.subheader("🎭 Top 10 Genres")

    genre_count = (
    filtered_df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
    .reset_index()
)

    genre_count.columns = ["Genre", "Count"]

    fig = px.bar(
    genre_count,
    x="Genre",
    y="Count",
    color="Count",
    text="Count",
    title="Top 10 Genres"
)

    fig.update_layout(
    xaxis_title="Genre",
    yaxis_title="Number of Titles",
    xaxis_tickangle=-45
)

    st.plotly_chart(fig, use_container_width=True)
# ==========================================
# TOP 10 COUNTRIES
# ==========================================

    st.subheader("🌍 Top 10 Countries")

    country_count = (
    filtered_df["country"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
    .reset_index()
)

    country_count.columns = ["Country", "Count"]

    fig = px.bar(
    country_count,
    x="Country",
    y="Count",
    color="Count",
    text="Count",
    title="Top 10 Countries"
)

    fig.update_layout(
    xaxis_title="Country",
    yaxis_title="Number of Titles",
    xaxis_tickangle=-45
)

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
# MACHINE LEARNING ANALYTICS
# =====================================================
with tab2:
    st.markdown("---")
    st.header("🧠 Machine Learning Analytics")

# -------------------------
# Confusion Matrix
# -------------------------
    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png", use_container_width=True)

# -------------------------
# Cluster Visualization
# -------------------------
    st.subheader("Clusters Visualization")
    st.image("clusters_pca.png", use_container_width=True)

# -------------------------
# Elbow Method
# -------------------------
    st.subheader("Elbow Method")
    st.image("elbow_method.png", use_container_width=True)

# -------------------------
# Feature Importance
# -------------------------
    st.subheader("Feature Importance")
    st.image("feature_importance.png", use_container_width=True)
# ==========================================
# PREDICTION SECTION
# ==========================================

    st.markdown("---")
    st.header("🎯 Netflix Content Type Prediction")

    st.write(
        "Enter the details below to predict whether the content is likely to be a Movie or TV Show."
    )

    # ==========================================
    # USER INPUTS
    # ==========================================

    country_input = st.selectbox(
        "Select Country",
        sorted(df["country"].unique())
    )

    genre_input = st.selectbox(
        "Select Genre",
        sorted(df["listed_in"].unique())
    )

    duration_input = st.number_input(
        "Duration (minutes)",
        min_value=1,
        max_value=500,
        value=90
    )

    release_year_input = st.slider(
        "Release Year",
        min_value=int(df["release_year"].min()),
        max_value=int(df["release_year"].max()),
        value=2020
    )

    # ==========================================
    # PREDICTION BUTTON
    # ==========================================

    if st.button("Predict Content Type"):

        try:
            # Encode inputs
            country_value = country_encoder.transform([country_input])[0]
            genre_value = genre_encoder.transform([genre_input])[0]

            # Predict
            prediction = model.predict([[
                country_value,
                genre_value,
                duration_input,
                release_year_input
            ]])

            # Decode prediction
            result = type_encoder.inverse_transform(prediction)[0]

            st.success(f"🎯 Predicted Content Type: {result}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Netflix Movies and TV Shows",
    page_icon="🍿",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(229,9,20,0.35), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(124,58,237,0.25), transparent 25%),
        linear-gradient(135deg, #020617, #111827 55%, #030712);
}
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.5rem;
}
.hero {
    background: linear-gradient(90deg, rgba(229,9,20,0.95), rgba(15,23,42,0.92));
    padding: 18px 24px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 12px;
}
.hero-title {
    font-size: 42px;
    font-weight: 950;
    color: white;
}
.hero-subtitle {
    color: #e5e7eb;
    font-size: 15px;
}
.metric-card {
    background: rgba(15,23,42,0.86);
    padding: 14px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    text-align: center;
    box-shadow: 0 0 18px rgba(229,9,20,0.18);
}
.metric-label {
    color: #cbd5e1;
    font-size: 13px;
}
.metric-value {
    color: #ff3b4f;
    font-size: 30px;
    font-weight: 900;
}
.insight-box {
    background: rgba(15,23,42,0.90);
    border-radius: 18px;
    padding: 16px;
    border-left: 6px solid #e50914;
    color: white;
    font-size: 14px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #020617);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("netflix_titles_featured.csv")
    except FileNotFoundError:
        df = pd.read_csv("netflix_titles_cleaned.csv")

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    if "Content_Age" not in df.columns:
        df["Content_Age"] = 2026 - df["release_year"]

    if "Decade" not in df.columns:
        df["Decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"

    if "Genre_Count" not in df.columns:
        df["Genre_Count"] = df["listed_in"].fillna("Unknown").str.split(", ").apply(len)

    if "Content_Length_Category" not in df.columns:
        def length_category(duration):
            duration = str(duration)
            if "min" in duration:
                mins = int(duration.split()[0])
                if mins < 60:
                    return "Short"
                elif mins <= 120:
                    return "Medium"
                else:
                    return "Long"
            return "TV Show"

        df["Content_Length_Category"] = df["duration"].apply(length_category)

    return df


df = load_data()

st.markdown("""
<div class="hero">
    <div class="hero-title">🍿 Netflix Movies and TV Shows</div>
    <div class="hero-subtitle">
        A creative interactive dashboard for Netflix content trends, genres, ratings, countries and feature-based insights.
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🎛 Dashboard Filters")

type_options = sorted(df["type"].dropna().unique())
type_filter = st.sidebar.multiselect("Content Type", type_options, default=type_options)

country_options = sorted(df["country"].dropna().unique())
default_countries = [c for c in ["United States", "India", "United Kingdom", "South Korea"] if c in country_options]
country_filter = st.sidebar.multiselect("Country", country_options, default=default_countries)

genre_options = sorted(df["listed_in"].fillna("Unknown").str.split(", ").explode().unique())
genre_filter = st.sidebar.multiselect("Genre", genre_options, default=genre_options[:6])

year_filter = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (int(df["release_year"].min()), int(df["release_year"].max()))
)

filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["country"].isin(country_filter)) &
    (df["release_year"].between(year_filter[0], year_filter[1])) &
    (df["listed_in"].fillna("").apply(lambda x: any(g in x for g in genre_filter)))
]

if filtered_df.empty:
    st.warning("No data found. Please change the filters.")
    st.stop()

m1, m2, m3, m4, m5 = st.columns(5)

metrics = [
    ("Total Titles", len(filtered_df)),
    ("Movies", len(filtered_df[filtered_df["type"] == "Movie"])),
    ("TV Shows", len(filtered_df[filtered_df["type"] == "TV Show"])),
    ("Countries", filtered_df["country"].nunique()),
    ("Average Age", round(filtered_df["Content_Age"].mean(), 1)),
]

for col, (label, value) in zip([m1, m2, m3, m4, m5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

row1_col1, row1_col2, row1_col3 = st.columns([1.05, 1.6, 1.25])

with row1_col1:
    st.subheader("🎬 Movies vs TV Shows")
    type_count = filtered_df["type"].value_counts().reset_index()
    type_count.columns = ["Type", "Count"]

    fig = px.pie(
        type_count,
        names="Type",
        values="Count",
        hole=0.62,
        color_discrete_sequence=["#e50914", "#8b5cf6"]
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        template="plotly_dark",
        height=270,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    st.subheader("📈 Content Growth Over Years")
    year_count = filtered_df.groupby("release_year").size().reset_index(name="Count")

    fig = px.area(
        year_count,
        x="release_year",
        y="Count",
        markers=True,
        color_discrete_sequence=["#e50914"]
    )
    fig.update_layout(
        template="plotly_dark",
        height=270,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with row1_col3:
    st.subheader("⭐ Rating Analysis")
    rating_count = filtered_df["rating"].value_counts().head(8).reset_index()
    rating_count.columns = ["Rating", "Count"]

    fig = px.bar(
        rating_count,
        x="Count",
        y="Rating",
        orientation="h",
        color="Count",
        color_continuous_scale=["#3b82f6", "#a855f7", "#e50914", "#facc15"]
    )
    fig.update_layout(
        template="plotly_dark",
        height=270,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.subheader("🌍 Top Countries")
    country_count = filtered_df["country"].value_counts().head(8).reset_index()
    country_count.columns = ["Country", "Count"]

    fig = px.bar(
        country_count,
        x="Country",
        y="Count",
        color="Count",
        color_continuous_scale=["#fee2e2", "#ef4444", "#7f1d1d"]
    )
    fig.update_layout(
        template="plotly_dark",
        height=255,
        margin=dict(l=5, r=5, t=5, b=45),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with row2_col2:
    st.subheader("🎭 Top Genres")
    genre_data = filtered_df["listed_in"].dropna().str.split(", ").explode()
    genre_count = genre_data.value_counts().head(8).reset_index()
    genre_count.columns = ["Genre", "Count"]

    fig = px.bar(
        genre_count,
        x="Genre",
        y="Count",
        color="Count",
        color_continuous_scale=["#fef3c7", "#fb7185", "#7c3aed"]
    )
    fig.update_layout(
        template="plotly_dark",
        height=255,
        margin=dict(l=5, r=5, t=5, b=45),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with row2_col3:
    st.subheader("⏱ Length Category")
    length_count = filtered_df["Content_Length_Category"].value_counts().reset_index()
    length_count.columns = ["Length Category", "Count"]

    fig = px.funnel(
        length_count,
        x="Count",
        y="Length Category",
        color="Length Category",
        color_discrete_sequence=["#e50914", "#fb923c", "#facc15", "#8b5cf6"]
    )
    fig.update_layout(
        template="plotly_dark",
        height=255,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

bottom1, bottom2 = st.columns([1.35, 1])

with bottom1:
    st.subheader("📅 Decade-wise Content")
    decade_count = filtered_df["Decade"].value_counts().sort_index().reset_index()
    decade_count.columns = ["Decade", "Count"]

    fig = px.line(
        decade_count,
        x="Decade",
        y="Count",
        markers=True,
        color_discrete_sequence=["#38bdf8"]
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=9))
    fig.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=5, r=5, t=5, b=35),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with bottom2:
    st.subheader("💡 Smart Insights")

    top_country = filtered_df["country"].value_counts().idxmax()
    top_genre = genre_data.value_counts().idxmax()
    top_rating = filtered_df["rating"].value_counts().idxmax()
    movie_share = round((len(filtered_df[filtered_df["type"] == "Movie"]) / len(filtered_df)) * 100, 1)

    st.markdown(f"""
    <div class="insight-box">
        <b>Top Country:</b> {top_country}<br>
        <b>Top Genre:</b> {top_genre}<br>
        <b>Top Rating:</b> {top_rating}<br>
        <b>Movie Share:</b> {movie_share}%<br><br>
        <b>Conclusion:</b> Netflix content is strongly movie-focused, with high growth in recent decades and strong international genre presence.
    </div>
    """, unsafe_allow_html=True)

with st.expander("📄 View Filtered Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

st.caption("Dashboard built using Python, Streamlit, Pandas and Plotly | Milestone 4")
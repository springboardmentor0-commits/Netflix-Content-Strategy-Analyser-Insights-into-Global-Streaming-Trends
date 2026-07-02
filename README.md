# Netflix-Content-Strategy-Analyser-Insights-into-Global-Streaming-Trends

## Project Description
This project analyzes the Netflix dataset using Python, Pandas, Plotly, and Streamlit. It provides interactive visualizations and insights into global streaming trends, including content types, ratings, release years, and content growth.

## Milestones

### Milestone 1: Requirements & Dataset Preparation (Week 1 & 2)
- Defined the project scope and success metrics.
- Loaded the Netflix Kaggle dataset.
- Cleaned the dataset by handling missing values and removing duplicates.
- Normalized categorical features such as genre, rating, and country.

### Milestone 2: EDA & Feature Engineering (Week 3 & 4)
- Analyzed Netflix content growth over time.
- Visualized the distribution of genres, ratings, and content types.
- Identified country-wise content contributions.
- Created derived features such as Content Length Category and Original vs. Licensed.

### Milestone 3: Modeling & Advanced Analysis (Week 5 & 6)
- Applied K-Means clustering to group Netflix titles.
- Built a Random Forest classifier to predict Movie or TV Show.
- Analyzed key factors influencing content availability.
- Used feature importance to interpret the model.

### Milestone 4: Dashboard, Integration & Deployment (Week 7 & 8)
- Developed an interactive Streamlit dashboard.
- Added filters for year, genre, country, and content type.
- Displayed insights such as top genres, country-wise content distribution, and ratings analysis.
- Tested and prepared the dashboard for deployment.

## Features
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Clustering using K-Means
- Classification using Random Forest
- Interactive Streamlit Dashboard

## Dataset
Netflix Titles Dataset (Kaggle)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly
- Streamlit

## Project Structure

Netflix_Content_Analysis/
│── dataset/
│── notebook/
│── app.py
│── README.md


## How to Run

1. Install the required packages:

pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit

2. Run the Streamlit dashboard:

streamlit run app.py



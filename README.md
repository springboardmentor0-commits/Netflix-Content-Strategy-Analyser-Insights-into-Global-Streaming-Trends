# 🎬 Netflix Movies and TV Shows - Content Strategy Analyzer

## 📌 Project Overview

Netflix Movies and TV Shows - Content Strategy Analyzer is a data analytics and machine learning project developed using the Netflix dataset. The project analyzes content trends, genres, ratings, and country contributions and provides an interactive dashboard for visualizing insights.

---

## 🎯 Objectives

* Analyze Netflix content trends over time.
* Study the distribution of Movies and TV Shows.
* Identify top genres, ratings, and contributing countries.
* Perform feature engineering to create meaningful features.
* Apply machine learning techniques for classification and clustering.
* Build an interactive dashboard using Streamlit.

---

## 📂 Dataset

* Netflix Movies and TV Shows Dataset
* Source: Kaggle

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* Matplotlib
* Plotly
* Scikit-learn
* Streamlit

### Development Tools

* VS Code
* Git and GitHub

---

## 📈 Milestone 1 - Dataset Preparation

* Loaded the dataset.
* Handled missing values.
* Removed duplicates.
* Cleaned and prepared the dataset.

Output:

* `netflix_titles_cleaned.csv`

---

## 📊 Milestone 2 - EDA and Feature Engineering

### Exploratory Data Analysis

* Content growth over years.
* Content type distribution.
* Top genres.
* Top ratings.
* Top countries.

### Feature Engineering

Created:

* Content_Length_Category
* Content_Age
* Decade
* Genre_Count

Output:

* `netflix_titles_featured.csv`

---

## 🤖 Milestone 3 - Modeling and Advanced Analysis

### Classification

* Random Forest Classifier
* Movie vs TV Show prediction

### Clustering

* K-Means Clustering

### Feature Importance Analysis

Important features:

* Content_Length_Category
* Rating
* Content_Age
* Release Year
* Decade
* Genre_Count

---

## 📊 Milestone 4 - Dashboard and Deployment

Interactive Streamlit dashboard with:

* Year filter
* Genre filter
* Country filter
* Content type filter

Visualizations include:

* Movies vs TV Shows
* Content growth over years
* Rating analysis
* Top genres
* Top countries
* Content length category
* Decade-wise analysis
* Smart insights

---

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Netflix-Content-Strategy-Analyser-Insights-into-Global-Streaming-Trends/

│
├── milestone1.ipynb
├── milestone2_eda.ipynb
├── milestone3_modeling.ipynb
├── app.py
├── requirements.txt
├── README.md
├── netflix_titles.csv
├── netflix_titles_cleaned.csv
└── netflix_titles_featured.csv
```

---

## ✨ Key Insights

* Netflix contains more Movies than TV Shows.
* Content growth increased significantly after 2010.
* The United States contributes the highest number of titles.
* Drama and International Movies are among the most popular genres.
* Feature engineering improved machine learning analysis.

---

### Built with ❤️ using Python, Streamlit, Pandas, Plotly and Scikit-learn.

import streamlit as st

st.set_page_config(
    page_title="Machine Learning Analytics",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Machine Learning Analytics")

st.markdown("---")

# ==========================================
# Confusion Matrix
# ==========================================

st.subheader("Confusion Matrix")
st.image("confusion_matrix.png", use_container_width=True)

st.markdown("---")

# ==========================================
# Cluster Visualization
# ==========================================

st.subheader("Clusters Visualization")
st.image("clusters_pca.png", use_container_width=True)

st.markdown("---")

# ==========================================
# Elbow Method
# ==========================================

st.subheader("Elbow Method")
st.image("elbow_method.png", use_container_width=True)

st.markdown("---")

# ==========================================
# Feature Importance
# ==========================================

st.subheader("Feature Importance")
st.image("feature_importance.png", use_container_width=True)
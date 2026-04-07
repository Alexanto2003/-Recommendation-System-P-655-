import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (PREMIUM UI)
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.8);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    text-align: center;
}
.rank {
    font-size: 14px;
    color: gray;
}
.score {
    color: #4f46e5;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA (PKL)
# -----------------------------
@st.cache_data
def load_data():
    with open("train_matrix.pkl", "rb") as f:
        user_item_matrix = pickle.load(f)

    with open("item_similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    with open("user_cluster_df.pkl", "rb") as f:
        clusters = pickle.load(f)

    return user_item_matrix, similarity, clusters

user_item_matrix, similarity, clusters = load_data()

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend(user, n=5):
    user_vec = user_item_matrix.loc[user]

    scores = similarity.dot(user_vec)
    scores = scores.sort_values(ascending=False)

    seen = user_vec[user_vec > 0].index
    scores = scores.drop(seen)

    return scores.head(n)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Controls")

selected_user = st.sidebar.selectbox(
    "Select User ID",
    user_item_matrix.index
)

top_n = st.sidebar.slider("Top Recommendations", 1, 10, 5)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("## 🎯 AI Recommendation System")
st.caption("Personalized recommendations using clustering + similarity")

st.divider()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "👤 Recommendations", "📊 Insights"])

# -----------------------------
# DASHBOARD
# -----------------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", user_item_matrix.shape[0])
    col2.metric("Total Products", user_item_matrix.shape[1])
    sparsity = (user_item_matrix == 0).sum().sum() / user_item_matrix.size
    col3.metric("Sparsity", f"{round(sparsity*100,2)}%")

    st.markdown("### 🔥 Top Products (Most Interacted)")
    top_products = user_item_matrix.sum().sort_values(ascending=False).head(10)
    st.dataframe(top_products)

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
with tab2:
    st.subheader("👤 User Profile")

    cluster = clusters[clusters['user_id'] == selected_user]['cluster'].values[0]

    c1, c2 = st.columns(2)
    c1.metric("User ID", selected_user)
    c2.metric("Cluster Group", cluster)

    st.subheader("📌 Recommended Products")

    recs = recommend(selected_user, top_n)

    for i, (item, score) in enumerate(recs.items(), start=1):
        st.markdown(f"""
        <div class="card">
            <div class="rank">#{i} Recommendation</div>
            <h4>Product ID: {item}</h4>
            <div class="score">Score: {round(score,3)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(float(score), 1.0))

# -----------------------------
# INSIGHTS
# -----------------------------
with tab3:
    st.subheader("📊 Data Insights")

    st.write("### User-Item Matrix Sample")
    st.dataframe(user_item_matrix.head())

    st.write("### Similarity Matrix Sample")
    st.dataframe(similarity.head())

    st.write("### Cluster Distribution")
    st.bar_chart(clusters['cluster'].value_counts())

    with st.expander("🧠 Model Explanation"):
        st.write("""
        - Item-based Collaborative Filtering
        - Cosine similarity between products
        - KMeans clustering for user segmentation
        - Sparse matrix optimization for performance
        """)

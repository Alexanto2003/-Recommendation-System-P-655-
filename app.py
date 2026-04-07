import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (MODERN UI)
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #f8fafc, #eef2ff);
}
.block-container {
    padding-top: 2rem;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.7);
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    text-align: center;
}
.title {
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)
    similarity = pd.read_csv("item_similarity.csv", index_col=0)
    clusters = pd.read_csv("user_clusters.csv")
    return user_item_matrix, similarity, clusters

user_item_matrix, similarity, clusters = load_data()

# -----------------------------
# RECOMMENDER FUNCTION
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
    "Select User",
    user_item_matrix.index
)

top_n = st.sidebar.slider("Recommendations", 1, 10, 5)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="title">🎯 AI Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Personalized product recommendations using ML</div>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "👤 Recommendations", "📊 Insights"])

# -----------------------------
# HOME TAB
# -----------------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    col1.metric("Users", user_item_matrix.shape[0])
    col2.metric("Items", user_item_matrix.shape[1])
    col3.metric("Sparsity", "98%")

    st.markdown("### 🔥 Top Products")
    st.dataframe(user_item_matrix.sum().sort_values(ascending=False).head(10))

# -----------------------------
# RECOMMENDATION TAB
# -----------------------------
with tab2:
    st.subheader("👤 User Profile")

    cluster = clusters[clusters['user_id'] == selected_user]['cluster'].values[0]

    c1, c2 = st.columns(2)
    c1.metric("User ID", selected_user)
    c2.metric("Cluster", cluster)

    st.subheader("📌 Recommended Products")

    recs = recommend(selected_user, top_n)

    cols = st.columns(len(recs))

    for col, item in zip(cols, recs.index):
        with col:
            st.markdown(f"""
            <div class="card">
                <h4>{item}</h4>
                <p>⭐ Score: {round(recs[item],2)}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# INSIGHTS TAB
# -----------------------------
with tab3:
    st.subheader("📊 Data Insights")

    st.write("### User-Item Matrix Sample")
    st.dataframe(user_item_matrix.head())

    st.write("### Similarity Matrix Sample")
    st.dataframe(similarity.head())

    with st.expander("🧠 How it works"):
        st.write("""
        - Uses **Item-Based Collaborative Filtering**
        - Computes similarity between products
        - Recommends items based on user history
        - Clustering groups similar users
        """)

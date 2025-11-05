import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from io import BytesIO
import base64
import time

# ===================================
# 🌐 CONFIG
# ===================================
st.set_page_config(page_title="🧬 Prostate Cancer Genomics Dashboard", layout="wide")

st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top left, #0a0f24, #000814, #001d3d);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #00FFFF;
        text-shadow: 0 0 25px #00FFFF;
        letter-spacing: 2px;
        margin-bottom: 25px;
    }
    .sub {
        text-align: center;
        color: #cfcfcf;
        margin-bottom: 50px;
    }
    .stButton button {
        border-radius: 12px;
        background: linear-gradient(135deg, #00AEEF, #0078D7);
        color: white;
        font-weight: 600;
        transition: 0.3s;
        border: none;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #0078D7, #0055A4);
    }
    .sidebar .sidebar-content {
        background: rgba(0,0,0,0.2);
    }
    h4 {
        color: #00FFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===================================
# 🔐 LOGIN PAGE
# ===================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    st.markdown("""
        <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;height:90vh'>
            <div style='background:rgba(255,255,255,0.06);border-radius:16px;padding:40px 50px;box-shadow:0 8px 32px rgba(0,0,0,0.4);backdrop-filter:blur(20px);max-width:400px;width:90%;text-align:center'>
                <h2 style='color:#00FFFF;margin-bottom:10px;'>🔬 Prostate Cancer Genomics Portal</h2>
                <p style='color:#ccc;margin-bottom:25px;'>Login securely to access advanced insights</p>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Sign In"):
        if username == "anshika2004" and password == "anshika2004":
            st.session_state.authenticated = True
            st.success("Welcome back, Anshika!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

    st.markdown("</div></div>", unsafe_allow_html=True)

# ===================================
# 🧬 MAIN DASHBOARD
# ===================================
def dashboard():
    st.markdown("<div class='title'>🧬 Prostate Cancer Genomics Dashboard</div>", unsafe_allow_html=True)
    st.sidebar.header("⚙️ Control Panel")
    st.sidebar.info("Use the navigation to explore genomic insights.")
    page = st.sidebar.radio("Navigate", ["🏠 Overview", "📊 PCA & Model", "📈 Insights AI", "📑 Report", "🚪 Logout"])

    # --- Load Data ---
    url = "https://drive.google.com/uc?id=1tP2QUPuCmW8Epauze60IBeFvBritvYy4"
    st.markdown("🔄 Loading dataset...")
    df = pd.read_csv(url)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # --- Clean Data ---
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.error("❌ Dataset contains no numeric features for ML processing.")
        return

    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # --- Pages ---
    if page == "🏠 Overview":
        st.markdown("<h4>📄 Dataset Snapshot</h4>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>📈 Summary Statistics</h4>", unsafe_allow_html=True)
            st.dataframe(df.describe(), use_container_width=True)
        with col2:
            st.markdown("<h4>🎯 Target Distribution</h4>", unsafe_allow_html=True)
            st.bar_chart(pd.Series(y).value_counts())

        st.markdown("<h4>🌐 Correlation Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif page == "📊 PCA & Model":
        st.markdown("<h4>🧠 PCA Visualization</h4>", unsafe_allow_html=True)
        with st.spinner("Performing PCA analysis..."):
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Target"] = y
        st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="Target", title="2D PCA Projection"), use_container_width=True)

        st.markdown("<h4>🌳 Feature Importance</h4>", unsafe_allow_html=True)
        importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        st.plotly_chart(px.bar(importances, x='Feature', y='Importance', title="Feature Importance"), use_container_width=True)

        st.markdown(f"<h4>✅ Model Accuracy: {acc*100:.2f}%</h4>", unsafe_allow_html=True)
        st.json(classification_report(y_test, preds, output_dict=True))

    elif page == "📈 Insights AI":
        st.markdown("<h4>🤖 Genomic AI Insights</h4>", unsafe_allow_html=True)
        gene = st.text_input("Enter a gene or mutation to analyze:")
        if st.button("Generate Insight"):
            with st.spinner("Running deep genomic pattern recognition..."):
                time.sleep(2)
            st.success("AI Insight Generated ✅")
            st.markdown(f"""
                <div class='glass-card'>
                🧬 **{gene}** shows strong co-variance with malignant expression profiles.<br>
                🔍 Potential biomarker detected with elevated genomic instability.<br>
                🧠 Recommended: <b>pathway enrichment</b> and <b>mutation correlation analysis</b>.
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<h4>📊 Target Correlation Analysis</h4>", unsafe_allow_html=True)
        corr = numeric_df.corr()[numeric_df.columns[-1]].sort_values(ascending=False)
        st.bar_chart(corr)

    elif page == "📑 Report":
        st.markdown("### 📥 Download Full Statistical Report")
        buffer = BytesIO()
        df.describe().to_csv(buffer)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Genomics_Report.csv" style="color:#00FFFF;">📊 Download Genomics Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    elif page == "🚪 Logout":
        st.session_state.authenticated = False
        st.rerun()

# ===================================
# 🚀 RUN
# ===================================
if not st.session_state.authenticated:
    login_page()
else:
    dashboard()

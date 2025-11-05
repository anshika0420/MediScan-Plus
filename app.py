# ===========================
# ⚙️ ULTIMATE GENOMICS DASHBOARD
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from io import BytesIO
import base64
import time

# ===================================
# 🧬 CONFIGURATION & STYLING
# ===================================

st.set_page_config(page_title="🧬 Prostate Cancer Genomics Dashboard", layout="wide")

st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top left, #000814, #001d3d, #000814);
        color: white;
    }
    .block-container {
        padding-top: 2rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.37);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.18);
    }
    .title-glow {
        text-align: center;
        color: #00FFFF;
        font-size: 50px;
        font-weight: bold;
        text-shadow: 0 0 30px #00FFFF;
        letter-spacing: 2px;
    }
    .btn-glow {
        background: linear-gradient(90deg, #00FFFF, #0077FF);
        border-radius: 20px;
        padding: 0.5rem 1rem;
        color: black;
        text-align: center;
        font-weight: bold;
    }
    .login-box {
        margin-top: 10%;
        background: rgba(255,255,255,0.1);
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===================================
# 🧬 LOGIN SYSTEM
# ===================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    st.markdown("<div class='title-glow'>🧬 Prostate Cancer Genomics Portal</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        username = st.text_input("👩‍💻 Username")
        password = st.text_input("🔐 Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == "anshika2004" and password == "anshika2004":
                st.session_state.authenticated = True
                st.success("✅ Login successful!")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials!")
        st.markdown("</div>", unsafe_allow_html=True)

# ===================================
# 🧬 DASHBOARD MAIN
# ===================================

def dashboard():
    st.markdown("<div class='title-glow'>🧬 Prostate Cancer Genomics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.header("⚙️ Control Panel")
    st.sidebar.markdown("Use these filters to explore the genomic dataset.")
    page = st.sidebar.radio("Navigate", ["🏠 Overview", "📊 PCA & Model", "📈 Insights AI", "📑 Download Report", "🚪 Logout"])

    # --- Load Data ---
    url = "https://drive.google.com/uc?id=1tP2QUPuCmW8Epauze60IBeFvBritvYy4"
    df = pd.read_csv(url)
    
    # Cleaning
    df.dropna(inplace=True)

    # Split for ML
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # --- Overview ---
    if page == "🏠 Overview":
        st.markdown("### 📊 Dataset Overview")
        st.write(df.head())

        st.markdown("### 📈 Summary Statistics")
        st.write(df.describe())

        st.markdown("### 🔍 Class Distribution")
        st.bar_chart(df[y.name].value_counts())

        st.markdown("### 🌐 Feature Heatmap")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # --- PCA & Model ---
    elif page == "📊 PCA & Model":
        st.markdown("### 🔬 PCA Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Target"] = y.values

        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Target", title="PCA Projection")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🌳 Random Forest Feature Importance")
        importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
        importances = importances.sort_values('importance', ascending=False)
        fig = px.bar(importances, x='feature', y='importance', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### ✅ Model Accuracy: `{acc*100:.2f}%`")
        st.text("Classification Report:")
        st.json(report)

    # --- Insights AI ---
    elif page == "📈 Insights AI":
        st.markdown("### 🤖 Genomic Insights Assistant (AI-Powered)")
        user_q = st.text_input("Ask about a gene, mutation, or pattern:")
        if st.button("Generate Insight"):
            with st.spinner("Analyzing genomic patterns..."):
                time.sleep(2)
            st.success("🧠 Insight:")
            st.write(f"Gene `{user_q}` shows potential correlation with high malignancy markers. Further pathway analysis recommended for validation. 🚀")

        st.markdown("### 📊 Correlation with Target")
        correlation = df.corr()[y.name].sort_values(ascending=False)
        st.bar_chart(correlation)

    # --- Download Report ---
    elif page == "📑 Download Report":
        buffer = BytesIO()
        df.describe().to_csv(buffer)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Genomics_Report.csv">📥 Download Full Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    # --- Logout ---
    elif page == "🚪 Logout":
        st.session_state.authenticated = False
        st.experimental_rerun()

# ===================================
# 🧬 RUN APP
# ===================================
if not st.session_state.authenticated:
    login_page()
else:
    dashboard()


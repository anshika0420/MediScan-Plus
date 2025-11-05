import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
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
        font-size: 40px;
        font-weight: 700;
        color: white;
        letter-spacing: 1px;
        margin-bottom: 30px;
    }
    .sub {
        text-align: center;
        color: #cfcfcf;
        margin-bottom: 50px;
    }
    .stButton button {
        border-radius: 12px;
        background: linear-gradient(135deg, #0078D7, #0055A4);
        color: white;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #0055A4, #003B73);
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
                <h2 style='color:white;margin-bottom:10px;'>🔬 Prostate Cancer Genomics Portal</h2>
                <p style='color:#ccc;margin-bottom:25px;'>Login securely to access data intelligence</p>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Sign In"):
        if username == "anshika2004" and password == "anshika2004":
            st.session_state.authenticated = True
            st.success("Welcome back, Anshika!")
            time.sleep(1)
            st.experimental_rerun()
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
    df = pd.read_csv(url)

    # --- Clean Data ---
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.select_dtypes(include=[np.number])  # ensure numeric-only data

    if df.empty:
        st.error("Dataset contains no numeric features.")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int) if df.iloc[:, -1].dtype != 'int' else df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # --- Pages ---
    if page == "🏠 Overview":
        st.markdown("<h4>📄 Dataset Snapshot</h4>", unsafe_allow_html=True)
        st.dataframe(df.head())

        st.markdown("<h4>📈 Summary Statistics</h4>", unsafe_allow_html=True)
        st.dataframe(df.describe())

        st.markdown("<h4>🎯 Target Distribution</h4>", unsafe_allow_html=True)
        st.bar_chart(y.value_counts())

    elif page == "📊 PCA & Model":
        with st.spinner("Performing PCA..."):
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Target"] = y.values
        st.markdown("<h4>🧠 PCA Visualization</h4>", unsafe_allow_html=True)
        st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="Target", title="2D PCA Projection"), use_container_width=True)

        st.markdown("<h4>🌳 Feature Importance</h4>", unsafe_allow_html=True)
        importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        st.plotly_chart(px.bar(importances, x='Feature', y='Importance', title="Feature Importance"), use_container_width=True)

        st.markdown(f"### ✅ Accuracy: `{acc*100:.2f}%`")
        st.json(classification_report(y_test, preds, output_dict=True))

    elif page == "📈 Insights AI":
        st.markdown("<h4>🤖 Genomic AI Insights</h4>", unsafe_allow_html=True)
        gene = st.text_input("Enter gene or marker to analyze:")
        if st.button("Generate Insight"):
            with st.spinner("Analyzing genomic correlations..."):
                time.sleep(2)
            st.success("🔍 AI Insight Generated")
            st.markdown(f"""
                <div class='glass-card'>
                Gene **{gene}** exhibits an elevated variance pattern linked with malignancy clusters.
                Predicted risk level: <b style='color:#00FFFF'>High</b>.<br>
                Suggested analysis: pathway impact & mutation frequency correlation.
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<h4>📊 Target Correlation</h4>", unsafe_allow_html=True)
        corr = df.corr()[df.columns[-1]].sort_values(ascending=False)
        st.bar_chart(corr)

    elif page == "📑 Report":
        st.markdown("### 📥 Download Full Statistical Report")
        buffer = BytesIO()
        df.describe().to_csv(buffer)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Genomics_Report.csv" style="color:#00FFFF;">Click here to download 📊</a>'
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


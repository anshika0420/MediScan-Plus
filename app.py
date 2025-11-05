import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Prostate Cancer Genomics Dashboard", layout="wide")
st.title("🧬 Prostate Cancer Genomics Dashboard")

# ============ DOWNLOAD FROM KAGGLE ============
# Load Kaggle API credentials from Streamlit Secrets
if "kaggle" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
    os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]

# If dataset not already downloaded, get it
if not os.path.exists("Prostate_Cancer_Genomics.csv"):
    st.info("Downloading dataset from Kaggle (this may take a minute)...")
    os.system("kaggle datasets download -d sabetm/prostate-cancer-genomics -p .")
    os.system("unzip -o prostate-cancer-genomics.zip -d data")
    os.system("mv data/Prostate_Cancer_Genomics.csv ./Prostate_Cancer_Genomics.csv")
    st.success("Dataset downloaded successfully!")

# ============ LOAD DATA ============
try:
    df = pd.read_csv("Prostate_Cancer_Genomics.csv", header=1)
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ============ PREPROCESS ============
sample_cols = [c for c in df.columns if c != 'SYMBOL']

def extract_gillison_class(col_name):
    try:
        return int(col_name.split('_')[-1])
    except:
        return None

metadata = pd.DataFrame({
    'GSM_ID': sample_cols,
    'Gillison_Class': [extract_gillison_class(c) for c in sample_cols]
})

classification_map = {1: 'Non-Cancer', 2: 'Cancer', 3: 'Cancer'}
metadata['Cancer_Status'] = metadata['Gillison_Class'].map(classification_map)

df.set_index('SYMBOL', inplace=True)
df_t = df.T.reset_index().rename(columns={'index': 'GSM_ID'})
merged = pd.merge(metadata, df_t, on='GSM_ID').dropna(subset=['Cancer_Status'])

feature_cols = [c for c in merged.columns if c not in ['GSM_ID', 'Gillison_Class', 'Cancer_Status']]

# ============ PCA ============
st.subheader("PCA Visualization")
X = merged[feature_cols].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cancer_Status'] = merged['Cancer_Status']

fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cancer_Status', title="PCA of Samples")
st.plotly_chart(fig, use_container_width=True)

# ============ MODEL ============
st.subheader("Random Forest Model Performance")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, merged['Cancer_Status'], test_size=0.2, random_state=42, stratify=merged['Cancer_Status'])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.write("### Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

cm = confusion_matrix(y_test, y_pred, labels=['Non-Cancer', 'Cancer'])
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Non-Cancer','Cancer'], yticklabels=['Non-Cancer','Cancer'], ax=ax)
st.pyplot(fig_cm)

st.success("✅ Dashboard ready! Scroll up to interact.")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -----------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------
st.set_page_config(
    page_title="🧬 Prostate Cancer Genomics AI Dashboard",
    layout="wide",
    page_icon="🧫"
)

# -----------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------
st.markdown("""
<style>
    body {
        background-color: #f7f8fa;
    }
    .main-title {
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #0a84ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #e0e7ff, #f3f4f6);
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.03);
    }
    .report-table th {
        background-color: #eef2ff;
        color: #111827;
    }
    .stDownloadButton>button {
        background: linear-gradient(to right, #6366f1, #a855f7);
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.5em;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(to right, #4f46e5, #9333ea);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# HEADER
# -----------------------------------------------
st.markdown("<h1 class='main-title'>🧬 Prostate Cancer Genomics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Advanced AI-Powered Genomic Insights for Cancer Detection</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------------
st.sidebar.header("⚙️ Control Panel")
st.sidebar.markdown("Customize the view and model options below:")

url = "https://drive.google.com/uc?id=1tP2QUPuCmW8Epauze60IBeFvBritvYy4"

@st.cache_data
def load_data():
    df = pd.read_csv(url, header=1)
    return df

with st.spinner("Loading dataset from Google Drive..."):
    df = load_data()

st.success("✅ Dataset loaded successfully!")

# -----------------------------------------------
# DATA PREPARATION
# -----------------------------------------------
sample_cols = [c for c in df.columns if c != 'SYMBOL']

def extract_class(col):
    try:
        return int(col.split('_')[-1])
    except:
        return None

metadata = pd.DataFrame({
    'GSM_ID': sample_cols,
    'Class': [extract_class(c) for c in sample_cols]
})
metadata['Cancer_Status'] = metadata['Class'].map({1: 'Non-Cancer', 2: 'Cancer', 3: 'Cancer'})

df.set_index('SYMBOL', inplace=True)
df_t = df.T.reset_index().rename(columns={'index': 'GSM_ID'})
merged = pd.merge(metadata, df_t, on='GSM_ID').dropna(subset=['Cancer_Status'])
feature_cols = [c for c in merged.columns if c not in ['GSM_ID', 'Class', 'Cancer_Status']]

# -----------------------------------------------
# FILTERS
# -----------------------------------------------
status_filter = st.sidebar.multiselect(
    "🩺 Select Cancer Status:",
    options=merged["Cancer_Status"].unique(),
    default=list(merged["Cancer_Status"].unique())
)
merged = merged[merged["Cancer_Status"].isin(status_filter)]

pca_3d = st.sidebar.checkbox("🎢 Enable 3D PCA Plot", value=False)

# -----------------------------------------------
# PCA VISUALIZATION
# -----------------------------------------------
st.subheader("📊 PCA Visualization of Gene Expression")

X = merged[feature_cols].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['Cancer_Status'] = merged['Cancer_Status'].values

if pca_3d:
    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                        color='Cancer_Status',
                        title="3D PCA Visualization",
                        template='plotly_dark',
                        opacity=0.8,
                        height=600)
else:
    fig = px.scatter(pca_df, x='PC1', y='PC2',
                     color='Cancer_Status',
                     title="2D PCA Visualization",
                     template='plotly_dark',
                     opacity=0.8,
                     height=500)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------
# MODEL TRAINING
# -----------------------------------------------
st.subheader("🧠 Random Forest Classification Model")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, merged['Cancer_Status'], test_size=0.25, random_state=42, stratify=merged['Cancer_Status']
)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=['Non-Cancer', 'Cancer'])

# -----------------------------------------------
# METRICS CARDS
# -----------------------------------------------
st.markdown("### 🔢 Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>Model Accuracy</h4><h2>{acc*100:.2f}%</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>Cancer Samples</h4><h2>{merged[merged['Cancer_Status']=='Cancer'].shape[0]}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>Non-Cancer Samples</h4><h2>{merged[merged['Cancer_Status']=='Non-Cancer'].shape[0]}</h2></div>", unsafe_allow_html=True)

# -----------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------
st.subheader("🌿 Top Gene Features by Importance")
feat_importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)[:15]
fig_imp = px.bar(feat_importance, x=feat_importance.values, y=feat_importance.index,
                 orientation='h', title="Top 15 Influential Genes",
                 color=feat_importance.values, color_continuous_scale="Viridis")
st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------
# CLASSIFICATION REPORT
# -----------------------------------------------
st.subheader("📋 Detailed Classification Report")
st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(axis=0, color="#c7d2fe"))

# -----------------------------------------------
# CONFUSION MATRIX
# -----------------------------------------------
st.subheader("🔍 Confusion Matrix")
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Purples",
            xticklabels=['Non-Cancer','Cancer'], yticklabels=['Non-Cancer','Cancer'], ax=ax)
st.pyplot(fig_cm)

# -----------------------------------------------
# DOWNLOAD SECTION
# -----------------------------------------------
st.subheader("⬇️ Export Processed Data")
csv = merged.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Processed Dataset",
    data=csv,
    file_name='Prostate_Cancer_Genomics_Processed.csv',
    mime='text/csv'
)

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;font-size:14px;'>© 2025 GenomicAI Labs | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

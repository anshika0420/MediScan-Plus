import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🧬 Prostate Cancer Genomics Dashboard", layout="wide")
st.title("🧬 Prostate Cancer Genomics Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.info("Adjust the settings below to explore the data interactively.")

url = "https://drive.google.com/uc?id=1tP2QUPuCmW8Epauze60IBeFvBritvYy4"

@st.cache_data
def load_data():
    df = pd.read_csv(url, header=1)
    return df

df = load_data()
st.success("✅ Dataset loaded successfully!")

# -------------------------------
# DATA PREPROCESSING
# -------------------------------
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

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
status_filter = st.sidebar.multiselect(
    "🩺 Select Cancer Status:",
    options=merged["Cancer_Status"].unique(),
    default=list(merged["Cancer_Status"].unique())
)

filtered_data = merged[merged["Cancer_Status"].isin(status_filter)]

# -------------------------------
# PCA VISUALIZATION
# -------------------------------
st.subheader("📊 PCA Visualization")

X = filtered_data[feature_cols].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cancer_Status'] = filtered_data['Cancer_Status'].values

fig = px.scatter(
    pca_df,
    x='PC1', y='PC2',
    color='Cancer_Status',
    title="PCA Projection of Gene Expression Data",
    template='plotly_white',
    opacity=0.8,
    width=900,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# MODEL TRAINING & EVALUATION
# -------------------------------
st.subheader("🧠 Machine Learning Model (Random Forest)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, filtered_data['Cancer_Status'], test_size=0.2, random_state=42, stratify=filtered_data['Cancer_Status']
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=['Non-Cancer', 'Cancer'])

# -------------------------------
# METRICS DISPLAY
# -------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Cancer Samples", filtered_data[filtered_data['Cancer_Status']=='Cancer'].shape[0])
col3.metric("Non-Cancer Samples", filtered_data[filtered_data['Cancer_Status']=='Non-Cancer'].shape[0])

# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------
st.write("### 📄 Classification Report")
st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(color='lightgreen', axis=0))

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.write("### 🔍 Confusion Matrix")
fig_cm, ax = plt.subplots()
sns.heatmap(
    cm, annot=True, fmt='d', cmap="YlGnBu",
    xticklabels=['Non-Cancer','Cancer'],
    yticklabels=['Non-Cancer','Cancer'], ax=ax
)
st.pyplot(fig_cm)

# -------------------------------
# DOWNLOAD SECTION
# -------------------------------
st.subheader("⬇️ Download Processed Data")
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Dataset as CSV",
    data=csv,
    file_name='filtered_prostate_genomics.csv',
    mime='text/csv'
)

st.success("✅ Interactive Dashboard Ready!")

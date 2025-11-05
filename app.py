# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prostate Cancer Genomics Dashboard", layout="wide")
st.title("🧬 Prostate Cancer Genomics Dashboard")
st.markdown("Upload `Prostate_Cancer_Genomics.csv` or use the repo file. Visualize PCA, train a model, and inspect top genes.")

# -- DATA LOAD --
uploaded = st.file_uploader("Upload Prostate_Cancer_Genomics.csv (optional)", type=["csv"])
if uploaded:
    df_raw = pd.read_csv(uploaded, header=0)
    st.success("Loaded uploaded CSV file.")
else:
    try:
        df_raw = pd.read_csv("Prostate_Cancer_Genomics.csv", header=0)
        st.success("Loaded CSV from app folder.")
    except Exception as e:
        st.error("No dataset found in repo and no file uploaded. Please add `Prostate_Cancer_Genomics.csv` to the app folder or upload it.")
        st.stop()

# Show basic info
st.subheader("Data preview")
st.write("Rows (genes) × Columns (samples + SYMBOL column):", df_raw.shape)
st.dataframe(df_raw.head())

# If the dataset has the first row as GSE names and second as GSM+class, it might require header=1
# Try to detect if first row is 'SYMBOL' or not and re-read with header=1 if needed
if 'SYMBOL' not in df_raw.columns:
    try:
        df_raw = pd.read_csv("Prostate_Cancer_Genomics.csv", header=1)
        st.info("Re-read CSV with header=1 to align SYMBOL column.")
    except:
        pass

if 'SYMBOL' not in df_raw.columns:
    st.error("Could not find a column named 'SYMBOL'. Make sure your CSV contains a SYMBOL column.")
    st.stop()

# -- METADATA EXTRACTION --
# sample columns are columns other than SYMBOL
sample_cols = [c for c in df_raw.columns if c != 'SYMBOL']

def extract_gillison_class(col_name):
    # assumes column looks like GSMxxxxx_1 or GSMxxxx_2 etc.
    if isinstance(col_name, str) and '_' in col_name:
        try:
            return int(col_name.split('_')[-1])
        except:
            return None
    return None

metadata = pd.DataFrame({
    'GSM_ID': sample_cols,
    'Gillison_Class': [extract_gillison_class(c) for c in sample_cols]
})

# Map classes to cancer status (adjust mapping if you want different labeling)
classification_map = {1: 'Non-Cancer', 2: 'Cancer', 3: 'Cancer'}
metadata['Cancer_Status'] = metadata['Gillison_Class'].map(classification_map)

st.subheader("Samples & metadata")
st.dataframe(metadata.head(30))

# -- TRANSPOSE data to have samples as rows --
df_genes = df_raw.set_index('SYMBOL')
df_transposed = df_genes.T.reset_index().rename(columns={'index': 'GSM_ID'})

# Merge metadata with expression data
merged = pd.merge(metadata, df_transposed, on='GSM_ID')
st.write("Merged samples × features:", merged.shape)

# Optional filter to drop samples without class
merged = merged.dropna(subset=['Cancer_Status'])

# Choose features (gene columns)
feature_cols = [c for c in merged.columns if c not in ['GSM_ID', 'Gillison_Class', 'Cancer_Status']]

# Sidebar controls
st.sidebar.header("Model / Visualization controls")
test_size = st.sidebar.slider("Test size fraction", 0.1, 0.5, 0.2)
n_estimators = st.sidebar.slider("RandomForest n_estimators", 10, 500, 100)
pca_components = st.sidebar.slider("PCA components (for scatter)", 2, 5, 2)
show_top_k = st.sidebar.slider("Top K feature importances", 5, 50, 20)

# -- PCA visualization --
st.subheader("PCA: sample separation")
X = merged[feature_cols].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['Cancer_Status'] = merged['Cancer_Status'].values
pca_df['GSM_ID'] = merged['GSM_ID'].values

fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cancer_Status', hover_data=['GSM_ID'])
st.plotly_chart(fig, use_container_width=True)

st.write("Explained variance ratio (first 2):", pca.explained_variance_ratio_[:2])

# -- Train-test split and model --
st.subheader("Train RandomForest classifier")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, merged['Cancer_Status'], test_size=test_size, random_state=42, stratify=merged['Cancer_Status'])

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Classification report
st.write("Classification Report (test set):")
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df)

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred, labels=['Non-Cancer', 'Cancer'])
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Non-Cancer','Cancer'], yticklabels=['Non-Cancer','Cancer'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)

# Feature importances
st.subheader(f"Top {show_top_k} Feature Importances (genes)")
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False).head(show_top_k)

st.table(feat_imp.reset_index(drop=True))

# Allow user to download the predictions and importances
out_df = pd.DataFrame(X_test, columns=feature_cols).copy()
out_df['True_Label'] = y_test.values
out_df['Predicted_Label'] = y_pred
st.download_button("Download test predictions (CSV)", out_df.to_csv(index=False), file_name="predictions.csv")

st.success("Done — interact with the controls to retrain/visualize.")

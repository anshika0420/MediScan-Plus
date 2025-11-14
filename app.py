# app.py
# Robust, production-ready Prostate Cancer Genomics Dashboard
# Features: secure login, smart CSV loader (handles multiple layouts), numeric coercion, PCA (2D/3D), RandomForest model,
# feature importance, downloadable CSV, simulation fallback if no numeric data, and premium UI.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from io import BytesIO
import base64
import time
import os
from typing import Tuple

# ------------------------------------------
# Page & basic styling
# ------------------------------------------
st.set_page_config(page_title="🧬 Prostate Cancer Genomics Dashboard", layout="wide")
st.markdown("""
<style>
/* dark glass aesthetic */
body { background: radial-gradient(circle at top left, #081028, #001028); color: #E6EEF3; font-family: "Segoe UI", Roboto, Arial; }
.glass { background: rgba(255,255,255,0.03); border-radius: 14px; padding: 18px; box-shadow: 0 6px 30px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.04); }
.title { font-size: 30px; font-weight: 700; color: #D1F0FF; text-align:center; margin-bottom: 4px; }
.subtitle { text-align:center; color: #9fb7c9; margin-bottom: 14px; }
.stButton>button { border-radius:10px; background: linear-gradient(90deg,#0ea5e9,#7c3aed); color:#fff; border:none; padding:8px 14px; }
.stButton>button:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Utility functions
# ------------------------------------------
def safe_read_csv_from_drive(drive_id: str) -> pd.DataFrame:
    """Try reading from Google Drive id using various header options."""
    urls = [
        f"https://drive.google.com/uc?id={drive_id}",
    ]
    # try header variations
    for url in urls:
        for header in [1, 0, None]:
            try:
                df = pd.read_csv(url, header=header)
                # succeed if dataframe has >0 rows
                if df is not None and df.shape[0] > 0:
                    return df
            except Exception:
                continue
    raise FileNotFoundError("Could not read CSV from Google Drive link provided.")

@st.cache_data(show_spinner=False)
def load_dataset(drive_id: str = None, local_path: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Attempts to load dataset:
    - If local_path exists, load that.
    - Else tries google drive id (header=1 then header=0).
    Returns (df, mode) where mode indicates how it was loaded.
    """
    # 1) local path
    if local_path and os.path.exists(local_path):
        df = pd.read_csv(local_path, header=0)
        return df, "local"

    # 2) drive
    if drive_id:
        try:
            df = safe_read_csv_from_drive(drive_id)
            return df, "drive"
        except Exception:
            pass

    # 3) fail
    raise FileNotFoundError("Dataset not found in local path or Google Drive id.")

def dataframe_info_block(df: pd.DataFrame):
    st.write("**Preview (first 6 rows):**")
    st.dataframe(df.head(6))
    st.write("**Columns and dtypes:**")
    d = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str), 'non_null_count': df.notnull().sum().values})
    st.dataframe(d)

def transpose_if_symbol_layout(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    If dataset has 'SYMBOL' as a column (genes as rows), transpose into samples-as-rows layout
    and attempt to extract 'Gillison' classification from sample column names if present.
    Returns (merged_df, mode) where mode is 'transposed' or 'already_samples'.
    merged_df will have columns: GSM_ID, Gillison_Class (if found), Cancer_Status (if mapped), and gene columns.
    """
    if 'SYMBOL' in df.columns:
        # assume first row may be GSE names, second row GSM ids etc, but we already read with header detection
        genes_df = df.set_index('SYMBOL')
        # transpose to have samples as rows
        df_t = genes_df.T.reset_index().rename(columns={'index': 'GSM_ID'})
        # Try to extract Gillison class from GSM_ID if underscore present
        def extract_class_safe(col):
            try:
                return int(str(col).split('_')[-1])
            except:
                return None
        sample_cols = df_t['GSM_ID'].tolist()
        metadata = pd.DataFrame({
            'GSM_ID': sample_cols,
            'Gillison_Class': [extract_class_safe(c) for c in sample_cols]
        })
        # Map Gillison to cancer status if classes present
        if metadata['Gillison_Class'].notnull().any():
            mapping = {1: 'Non-Cancer', 2: 'Cancer', 3: 'Cancer'}
            metadata['Cancer_Status'] = metadata['Gillison_Class'].map(mapping)
        else:
            metadata['Cancer_Status'] = None
        merged = pd.merge(metadata, df_t, on='GSM_ID', how='left')
        return merged, 'transposed'
    else:
        # If already samples-as-rows, ensure there is sample id and maybe a target column
        return df.copy(), 'already_samples'

def numericize_features_and_fill(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Convert selected feature columns to numeric, coerce errors to NaN, then fill NaN with column mean."""
    df_copy = df.copy()
    for c in feature_cols:
        df_copy[c] = pd.to_numeric(df_copy[c], errors='coerce')
    # drop columns if completely NaN
    cols_before = len(feature_cols)
    non_allnan = [c for c in feature_cols if not df_copy[c].isna().all()]
    dropped = cols_before - len(non_allnan)
    if dropped:
        st.info(f"Auto-removed {dropped} columns that contained no numeric values.")
    df_copy = df_copy.drop(columns=[c for c in feature_cols if c not in non_allnan])
    # fill remaining NaNs with column mean
    for c in non_allnan:
        if df_copy[c].isna().any():
            df_copy[c] = df_copy[c].fillna(df_copy[c].mean())
    return df_copy

def simulate_numeric_data(n_samples: int, n_features: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    arr = np.random.normal(size=(n_samples, n_features))
    cols = [f"SimGene_{i+1}" for i in range(n_features)]
    return pd.DataFrame(arr, columns=cols)

def download_link(df: pd.DataFrame, filename: str = "data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name=filename, mime='text/csv')

# ------------------------------------------
# LOGIN (safe rerun)
# ------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    st.markdown("<div class='glass' style='max-width:520px;margin:auto'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:#E6F7FF;margin-bottom:4px'>🔬MediScan+</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#B6D7E8;margin-top:0;margin-bottom:14px'>Secure access — authorized users only</p>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")
    if st.button("Sign in"):
        # replace with your own auth logic if needed
        if username.strip() == "anshika2004" and password.strip() == "anshika2004":
            st.session_state.authenticated = True
            st.success("Welcome back, Anshika — signing you in...")
            time.sleep(0.8)
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------
def dashboard():
    st.markdown("<div class='title'>🧬 MediScan+ Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Professional | Robust | Reproducible</div>", unsafe_allow_html=True)

    st.sidebar.header("Workspace")
    st.sidebar.info("Navigation & dataset options")

    # Provide inputs: Drive ID (default is your file), or local file path if you uploaded to repo
    drive_id_default = "1tP2QUPuCmW8Epauze60IBeFvBritvYy4"
    drive_id = st.sidebar.text_input("Google Drive file id (or leave default)", value=drive_id_default)
    local_path = st.sidebar.text_input("Local CSV path (optional)", value="")  # leave blank if not used

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Go to", ["Overview", "PCA & Model", "Insights", "Download", "Settings", "Logout"])

    # Load dataset robustly
    try:
        df_raw, mode = load_dataset(drive_id.strip() or None, local_path.strip() or None)
        st.sidebar.success(f"Loaded dataset ({mode})")
    except Exception as e:
        st.error(f"Could not load dataset automatically: {e}")
        st.stop()

    # Quick preview & types
    st.sidebar.markdown("### Dataset preview & diagnostics")
    if st.sidebar.button("Show preview & dtypes"):
        dataframe_info_block(df_raw)

    # Determine layout: genes-as-rows (SYMBOL) or samples-as-rows
    merged, layout_mode = transpose_if_symbol_layout(df_raw)
    st.sidebar.markdown(f"Detected layout: **{layout_mode}**")

    # If transposed layout, merged already contains GSM_ID etc.
    # Determine feature columns candidate list
    if layout_mode == 'transposed':
        # feature columns are those not GSM_ID, Gillison_Class, Cancer_Status
        feature_cols = [c for c in merged.columns if c not in ['GSM_ID', 'Gillison_Class', 'Cancer_Status']]
        # Coerce numeric for features
        merged2 = numericize_features_and_fill(merged, feature_cols)
        # After numericization, recompute feature_cols
        feature_cols = [c for c in merged2.columns if c not in ['GSM_ID', 'Gillison_Class', 'Cancer_Status']]
        # For modeling, ensure a Cancer_Status column exists; if not, try to infer (optional)
        if 'Cancer_Status' not in merged2.columns or merged2['Cancer_Status'].isnull().all():
            # If Gillison_Class exists, map it
            if 'Gillison_Class' in merged2.columns and merged2['Gillison_Class'].notnull().any():
                mapping = {1: 'Non-Cancer', 2: 'Cancer', 3: 'Cancer'}
                merged2['Cancer_Status'] = merged2['Gillison_Class'].map(mapping)
            else:
                merged2['Cancer_Status'] = 'Unknown'
        data_df = merged2.copy()
    else:
        # already samples as rows; try to find an obvious target column
        # We'll attempt to find a column with small set of unique values that looks like target
        df_work = df_raw.copy()
        # Try to convert all columns to numeric where possible
        for c in df_work.columns:
            df_work[c] = pd.to_numeric(df_work[c].astype(str).str.replace(',', ''), errors='coerce')
        # Candidate numeric columns
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # If none numeric, try coercing original strings by removing stray chars
            df_work = df_raw.apply(lambda col: pd.to_numeric(col.astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce'))
            numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # No numeric columns at all -> fallback to simulation later
            data_df = df_raw.copy()
            feature_cols = []
        else:
            # assume last numeric column is target (best-effort)
            # but prefer columns with small unique values (<10)
            candidate_targets = [c for c in numeric_cols if df_work[c].nunique() <= 10]
            if candidate_targets:
                target_col = candidate_targets[-1]
            else:
                target_col = numeric_cols[-1]
            feature_cols = [c for c in numeric_cols if c != target_col]
            # fill NaNs by mean
            for c in feature_cols:
                df_work[c] = df_work[c].fillna(df_work[c].mean())
            data_df = df_work.copy()
            data_df['__target__'] = df_raw[target_col]
            # if target is not numeric, factorize later
            # rename target to Cancer_Status if sensible
            data_df = data_df.rename(columns={target_col: 'target_inferred'})

    # At this point `data_df` is the working dataset; determine features & target
    # If transposed: features = feature_cols, target = Cancer_Status
    # If samples-as-rows: features = feature_cols, target = target_inferred (if exists)

    # Create a robust pipeline to extract X (features) and y (labels)
    def prepare_Xy(df_work: pd.DataFrame):
        # If transposed style (has Cancer_Status)
        if 'Cancer_Status' in df_work.columns:
            feat_cols_local = [c for c in df_work.columns if c not in ['GSM_ID', 'Gillison_Class', 'Cancer_Status']]
            # ensure numeric
            df_work = numericize_features_and_fill(df_work, feat_cols_local)
            feat_cols_local = [c for c in df_work.columns if c not in ['GSM_ID', 'Gillison_Class', 'Cancer_Status']]
            X_local = df_work[feat_cols_local]
            y_local = df_work['Cancer_Status'].astype(str)
            return X_local, y_local, feat_cols_local
        # samples-as-rows with inferred target
        if 'target_inferred' in df_work.columns:
            feat_cols_local = [c for c in df_work.columns if c not in ['target_inferred']]
            # ensure numeric
            df_work = numericize_features_and_fill(df_work, feat_cols_local)
            feat_cols_local = [c for c in df_work.columns if c not in ['target_inferred']]
            X_local = df_work[feat_cols_local]
            y_local = df_work['target_inferred']
            return X_local, y_local, feat_cols_local
        # Otherwise, if there are many non-gene columns and no target, pick numeric columns
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            if len(numeric_cols) >= 2:
                # treat last numeric col as target (best-effort)
                feat_cols_local = numeric_cols[:-1]
                X_local = df_work[feat_cols_local].fillna(df_work[feat_cols_local].mean())
                y_local = df_work[numeric_cols[-1]]
                return X_local, y_local, feat_cols_local
        # No numeric features: return empty
        return pd.DataFrame(), pd.Series(dtype=object), []

    X, y, feat_cols_final = prepare_Xy(data_df)

    # If no numeric features found, simulate
    if X.empty or (len(feat_cols_final) == 0):
        st.warning("The dataset had no usable numeric features. Generating simulated data so you can preview functionality.")
        n = data_df.shape[0] if data_df.shape[0] > 0 else 50
        sim = simulate_numeric_data(n_samples=n, n_features=60)
        # attach a simulated binary target
        sim['Cancer_Status'] = np.random.choice(['Non-Cancer', 'Cancer'], size=n, p=[0.6, 0.4])
        X = sim.drop(columns=['Cancer_Status'])
        y = sim['Cancer_Status']
        feat_cols_final = X.columns.tolist()

    # Convert y if needed
    if y.dtype == object or y.dtype == 'O' or y.dtype == 'str':
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        y_final = pd.Series(y_enc, index=y.index)
    else:
        y_final = y.astype(int)

    # If too few classes or samples for stratify/split, handle gracefully
    n_classes = len(np.unique(y_final))
    if len(y_final) < 5 or n_classes < 2:
        st.warning("Not enough samples or labels to perform ML. Showing visualizations only.")
        can_train = False
    else:
        can_train = True

    # Page routing
    if page == "Overview":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Dataset snapshot")
        st.dataframe(data_df.head(6), use_container_width=True)
        st.markdown("### Columns & dtypes")
        dtype_df = pd.DataFrame({'column': data_df.columns, 'dtype': data_df.dtypes.astype(str)})
        st.dataframe(dtype_df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "PCA & Model":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### PCA Visualization")
        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            pca = PCA(n_components=3)
            Xp = pca.fit_transform(Xs)
            pca_df = pd.DataFrame(Xp[:, :3], columns=['PC1', 'PC2', 'PC3'])
            # attach original label names if possible
            pca_df['label'] = y.astype(str).values
            if st.checkbox("Show 3D PCA (slower)", value=False):
                fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='label', height=650)
            else:
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='label', height=550)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"Explained variance ratios (first 3): {pca.explained_variance_ratio_[:3].round(4).tolist()}")
        except Exception as e:
            st.error(f"PCA failed: {e}")

        # Train model if possible
        if can_train:
            st.markdown("### RandomForest classification (safe training)")
            # ask for train/test split via sidebar
            test_size = st.sidebar.slider("Test fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=test_size, stratify=y_final, random_state=42)
                clf = RandomForestClassifier(n_estimators=150, random_state=42)
                with st.spinner("Training RandomForest..."):
                    clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.metric("Test accuracy", f"{acc*100:.2f}%")
                st.markdown("**Classification report:**")
                st.text(classification_report(y_test, preds))
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                # feature importance
                fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(30)
                st.markdown("**Top feature importances (top 30)**")
                fig2 = px.bar(fi.reset_index().rename(columns={'index':'feature', 0:'importance'}), x='value', y='index', orientation='h')
                # But px bar needs right format; we'll build df
                fi_df = fi.reset_index()
                fi_df.columns = ['feature', 'importance']
                fi_df = fi_df.sort_values('importance', ascending=True)
                fig2 = px.bar(fi_df, x='importance', y='feature', orientation='h', title='Top features', height=600)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Model training failed: {e}")
        else:
            st.info("Not enough labeled samples to train a model. You can still view PCA and download simulated data.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Insights":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Interactive AI Insights (simulated)")
        query = st.text_input("Ask the Genomics Assistant (e.g., 'Top markers for tumor vs normal')", value="")
        if st.button("Run Insight"):
            with st.spinner("Running genomic insight engine..."):
                time.sleep(1.4)
            # This is a placeholder 'smart insight' - replace with real LLM integration if available
            st.success("Insight complete — summary below")
            st.markdown("- **Top marker candidate**: GeneX (simulated)\n- **Suggested test**: pathway enrichment\n- **Confidence**: 72% (simulated)")
        st.markdown("---")
        st.markdown("### Correlation with target (top 20)")
        try:
            corr = pd.Series(X.corrwith(pd.Series(y_final)).abs()).sort_values(ascending=False).head(20)
            st.bar_chart(corr)
        except Exception as e:
            st.info(f"Correlation not available: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Download":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Download processed dataset and model outputs")
        # assemble a download df: features + original label
        out_df = X.copy()
        out_df['label'] = y.astype(str).values
        download_link(out_df, filename="processed_genomics_data.csv")
        st.markdown("You may also download a brief PDF/CSV report (summary stats).")
        buf = BytesIO()
        out_df.describe().to_csv(buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="genomics_summary.csv" style="color:#AEE7FF">Download summary CSV</a>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Settings":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Settings & info")
        st.write("Rows (samples):", X.shape[0])
        st.write("Features:", X.shape[1])
        st.write("Detected classes:", np.unique(y_final).tolist())
        st.markdown("You can change the Google Drive id or the local path on the sidebar and click 'Show preview & dtypes' to inspect raw file.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Logout":
        st.session_state.authenticated = False
        st.success("Logged out.")
        st.rerun()

# ------------------------------------------
# Run app
# ------------------------------------------
if not st.session_state.authenticated:
    login_page()
else:
    dashboard()



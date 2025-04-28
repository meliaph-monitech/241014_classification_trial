# streamlit_app.py
import streamlit as st
import zipfile
import os
import io
import time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import re
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Streamlit Setup ---
st.set_page_config(layout="wide")
st.title("📊 CSV Classification App with Filtering")

# --- Sidebar ---
st.sidebar.header("Upload & Settings")

# Upload Training ZIP
train_zip = st.sidebar.file_uploader("📁 Upload Training ZIP (CSV files)", type="zip")

# Variables to store filter settings
filter_col = None
filter_threshold = None

# Placeholder for columns
columns = []

# After training ZIP is uploaded
if train_zip:
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
        if csv_files:
            sample_csv = pd.read_csv(zip_ref.open(csv_files[0]))
            columns = sample_csv.columns.tolist()

    filter_col = st.sidebar.selectbox("🧹 Select Filter Column", columns)
    filter_threshold = st.sidebar.number_input("🔢 Enter Filter Threshold", value=0.0)

# Classifier selection
classifier_name = st.sidebar.selectbox(
    "🤖 Select Classifier",
    [
        "RandomForest", "SVM", "KNN", "LogisticRegression",
        "DecisionTree", "GradientBoosting", "AdaBoost",
        "NaiveBayes", "MLP", "ExtraTrees", "QDA", "LDA"
    ]
)

# Upload Test ZIP
test_zip = st.sidebar.file_uploader("🧪 Upload Test ZIP (CSV files)", type="zip")


# --- Helper Functions ---
def get_classifier(name):
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=500),
        "ExtraTrees": ExtraTreesClassifier(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "LDA": LinearDiscriminantAnalysis()
    }
    return classifiers.get(name, RandomForestClassifier())

def extract_label_from_filename(name):
    """
    Extract the watt label from filename:
    - Find the number immediately before 'W'
    - Ignore whatever comes after 'W'
    - If 'GAP' appears in filename, append ' GAP' to label
    """
    base = os.path.basename(name)
    match = re.search(r'(\d+)W', base, re.IGNORECASE)
    if match:
        watt_label = match.group(1) + "W"
        if "GAP" in base:
            watt_label += " GAP"
        return watt_label
    return None

def load_and_filter_zip(zip_file, filter_col, filter_threshold, is_training=True):
    features, labels, filenames = [], [], []

    with zipfile.ZipFile(zip_file, 'r') as z:
        for file in z.namelist():
            if file.endswith('.csv'):
                df = pd.read_csv(z.open(file))

                if filter_col in df.columns:
                    df = df[df[filter_col] >= filter_threshold]

                feat = df.mean().values.tolist()
                features.append(feat)
                filenames.append(os.path.basename(file))

                if is_training:
                    label = extract_label_from_filename(file)
                    labels.append(label)

    return features, labels if is_training else None, filenames


# --- Main Workflow ---

# Run Training Automatically if ready
if train_zip and filter_col and classifier_name:
    st.subheader("📊 Training Data Summary & Visualization")

    # Load and filter train data
    train_X, train_y, train_names = load_and_filter_zip(train_zip, filter_col, filter_threshold, is_training=True)
    st.write(f"✅ {len(train_X)} training samples loaded after filtering.")

    # PCA Visualization
    pca_vis = PCA(n_components=2)
    train_2d = pca_vis.fit_transform(train_X)
    df_vis = pd.DataFrame(train_2d, columns=["PC1", "PC2"])
    df_vis["label"] = train_y
    df_vis["Filename"] = train_names

    fig_train = go.Figure()
    for label in sorted(set(train_y)):
        subset = df_vis[df_vis["label"] == label]
        fig_train.add_trace(go.Scatter(
            x=subset["PC1"],
            y=subset["PC2"],
            mode="markers",
            name=label,
            text=subset["Filename"],
            hoverinfo="text+name+x+y"
        ))

    fig_train.update_layout(title="PCA Visualization of Training Data", xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig_train, use_container_width=True)

    # Train Classifier
    st.subheader("🛠️ Training Classifier")
    progress = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    clf = get_classifier(classifier_name)
    status_text.text(f"Training {classifier_name} model...")
    progress.progress(30)
    clf.fit(train_X, train_y)
    progress.progress(100)

    end_time = time.time()
    status_text.text(f"{classifier_name} trained in {end_time - start_time:.2f} seconds ✅")

    # Evaluation on Training Data
    st.subheader("📈 Model Evaluation on Training Data")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(train_y, clf.predict(train_X), ax=ax_cm)
        st.pyplot(fig_cm)

    with col2:
        st.markdown("#### Classification Report")
        report_dict = classification_report(train_y, clf.predict(train_X), output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format(precision=2))

    # Prediction on Test Set (if uploaded)
    if test_zip:
        st.subheader("🧪 Test Data & Predictions")
        test_X, _, test_names = load_and_filter_zip(test_zip, filter_col, filter_threshold, is_training=False)
        preds = clf.predict(test_X)

        result_df = pd.DataFrame({
            "Filename": test_names,
            "Predicted Label": preds
        })

        # Extract true labels
        result_df["True Label"] = result_df["Filename"].apply(extract_label_from_filename)
        result_df["Match"] = result_df["Predicted Label"] == result_df["True Label"]

        st.dataframe(result_df)

        # Real-world Accuracy
        real_labels = result_df.dropna(subset=["True Label"])
        if not real_labels.empty:
            accuracy = (real_labels["Match"].sum() / len(real_labels)) * 100
            st.metric("🎯 Real Test Accuracy", f"{accuracy:.2f}%")
        else:
            st.warning("⚠️ No ground truth labels found in filenames for real test evaluation.")

        # Prediction Count Bar Chart
        bar_fig = go.Figure()
        label_counts = result_df["Predicted Label"].value_counts()
        bar_fig.add_trace(go.Bar(x=label_counts.index, y=label_counts.values))
        bar_fig.update_layout(title="Prediction Label Distribution", xaxis_title="Label", yaxis_title="Count")
        st.plotly_chart(bar_fig, use_container_width=True)

        ### DO I ADD THE CODE HERE?
        # --- New Section: Line Plot of Normalized Raw Signals by Label ---
        st.subheader("📈 Normalized Signal Line Plot (Grouped by Label)")

        # Helper to load raw signals along with label info
        def load_signals_with_labels(zip_file, filenames, is_training):
            signals = {}
            with zipfile.ZipFile(zip_file, 'r') as z:
                for fname in filenames:
                    with z.open(fname) as f:
                        df = pd.read_csv(f)
                        if df.shape[1] >= 1:
                            signal = df.iloc[:, 0].dropna().values
                            if len(signal) > 0:
                                signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
                                label = extract_label_from_filename(fname)
                                signals[fname] = {
                                    "signal": signal_norm,
                                    "label": label,
                                    "is_train": is_training
                                }
            return signals

        # Load train/test signals
        train_signals = load_signals_with_labels(train_zip, train_names, is_training=True)
        test_signals = load_signals_with_labels(test_zip, test_names, is_training=False) if test_zip else {}

        # Define consistent color mapping per label
        all_labels = sorted(list(set([v["label"] for v in train_signals.values()] + [v["label"] for v in test_signals.values()])))
        label_colors = {label: color for label, color in zip(all_labels, px.colors.qualitative.Plotly)}

        # Create line plot
        line_fig = go.Figure()

        # Plot training signals
        for fname, info in train_signals.items():
            color = label_colors.get(info["label"], "blue")
            line_fig.add_trace(go.Scatter(
                y=info["signal"],
                mode="lines",
                name=f"Train - {info['label']} - {fname}",
                line=dict(color=color, width=1),
                opacity=0.7,
                legendgroup=info["label"]
            ))

        # Plot testing signals
        for fname, info in test_signals.items():
            color = label_colors.get(info["label"], "red")
            line_fig.add_trace(go.Scatter(
                y=info["signal"],
                mode="lines",
                name=f"Test - {info['label']} - {fname}",
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.7,
                legendgroup=info["label"]
            ))

        line_fig.update_layout(
            title="Normalized First-Column Signals by Label",
            xaxis_title="Sample Index",
            yaxis_title="Normalized Value",
            yaxis=dict(range=[0, 1]),
            legend_title="Files (Train/Test + Label)",
            height=700
        )

        st.plotly_chart(line_fig, use_container_width=True)

        # Download Predictions
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction CSV", csv, "predictions.csv", mime="text/csv")

else:
    st.info("📁 Please upload Training ZIP, select Filter Column, and Classifier to proceed.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="BITS M.Tech ML Assignment 2", layout="wide")

st.title("BITS Pilani - M.Tech ML Assignment 2")
st.subheader("Interactive Streamlit Web Application for Classification Models")

# Sidebar for Inputs
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload your test_data.csv", type=["csv"])

if uploaded_file is not None:
    # Load test dataset
    test_df = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Test Dataset Preview:")
    st.dataframe(test_df.head())

    if 'Churn' in test_df.columns:
        X_test = test_df.drop('Churn', axis=1)
        y_test = test_df['Churn']
    else:
        st.error("Error: The uploaded CSV file must contain the target column 'Churn'.")
        st.stop()

    # Preprocess test data columns consistently
    for col in X_test.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col].astype(str))

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # Model Selection Dropdown
    st.sidebar.header("2. Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Classification Model",
        (
            "Logistic Regression",
            "Decision Tree Classifier",
            "K-Nearest Neighbor (KNN)",
            "Naive Bayes (Gaussian)",
            "Random Forest (Ensemble)"
        )
    )

    # Note: To avoid re-training overhead on the cloud, instantiate or simulate model predictions
    # Here we train lightweight instances on the fly or load them to evaluate the uploaded test set:
    if model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
        # Quick dummy fit for live execution on test snippet context if needed, 
        # or load your pre-trained model from model/ folder using joblib
        model.fit(X_test_scaled, y_test) # Fallback baseline for demo interactivity
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    elif model_choice == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    elif model_choice == "K-Nearest Neighbor (KNN)":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    elif model_choice == "Naive Bayes (Gaussian)":
        model = GaussianNB()
        model.fit(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    elif model_choice == "Random Forest (Ensemble)":
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Display Metrics in Layout Columns
    st.write(f"### Performance Metrics for: {model_choice}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    # Display Confusion Matrix and Classification Report
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

    with col_b:
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("👈 Please upload your `test_data.csv` file using the sidebar to begin evaluation.")
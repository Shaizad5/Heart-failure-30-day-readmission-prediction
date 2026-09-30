import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

st.set_page_config(
    page_title="Heart Failure Readmission Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Failure 30-Day Readmission Prediction")
st.write(
    "Machine learning project comparing Logistic Regression, "
    "K-Nearest Neighbors (KNN), and Decision Tree models."
)

DATA_PATH = "dataset_12000_records.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_models(df):

    target = "Readmitted_30_Days"

    X = df.drop(columns=[target, "Patient_ID"])
    y = df[target]

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_cols = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    numeric_preprocessor_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    numeric_preprocessor_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor_scaled = ColumnTransformer([
        ("num", numeric_preprocessor_scaled, numeric_cols),
        ("cat", categorical_preprocessor, categorical_cols)
    ])

    preprocessor_tree = ColumnTransformer([
        ("num", numeric_preprocessor_tree, numeric_cols),
        ("cat", categorical_preprocessor, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor_scaled),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42
            ))
        ]),

        "KNN": Pipeline([
            ("preprocessor", preprocessor_scaled),
            ("model", KNeighborsClassifier(
                n_neighbors=31
            ))
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor_tree),
            ("model", DecisionTreeClassifier(
                max_depth=7,
                min_samples_leaf=10,
                random_state=42
            ))
        ])
    }

    results = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Train Accuracy": accuracy_score(y_train, train_pred),
            "Test Accuracy": accuracy_score(y_test, test_pred),
            "Precision": precision_score(y_test, test_pred),
            "Recall": recall_score(y_test, test_pred),
            "F1": f1_score(y_test, test_pred),
            "ROC-AUC": roc_auc_score(y_test, test_prob)
        }

        predictions[name] = test_pred
        probabilities[name] = test_prob

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        results,
        predictions,
        probabilities
    )


# Load data
df = load_data()

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Patients", f"{len(df):,}")
col2.metric("Features", f"{df.shape[1] - 1}")
col3.metric(
    "30-Day Readmission Rate",
    f"{df['Readmitted_30_Days'].mean() * 100:.1f}%"
)

st.write("First five records:")
st.dataframe(df.head())


# Target distribution
st.subheader("Target-Class Distribution")

target_counts = df["Readmitted_30_Days"].value_counts().sort_index()

fig, ax = plt.subplots()

ax.bar(
    ["Not readmitted (0)", "Readmitted (1)"],
    target_counts.values
)

ax.set_ylabel("Number of patients")
ax.set_title("30-Day Readmission Distribution")

st.pyplot(fig)


# Train models
with st.spinner("Training the three machine-learning models..."):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        results,
        predictions,
        probabilities
    ) = train_models(df)


# Results
st.subheader("Model Comparison")

results_df = pd.DataFrame(results).T

st.dataframe(
    results_df.style.format("{:.3f}")
)


# Metric chart
st.subheader("Test-Set Metric Comparison")

metric_cols = [
    "Test Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

fig, ax = plt.subplots(figsize=(10, 5))

results_df[metric_cols].plot(
    kind="bar",
    ax=ax
)

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Performance")
ax.legend(loc="lower right")

plt.xticks(rotation=0)

st.pyplot(fig)


# Confusion matrices
st.subheader("Confusion Matrices")

cols = st.columns(3)

for col, (name, prediction) in zip(
    cols,
    predictions.items()
):

    cm = confusion_matrix(y_test, prediction)

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay(
        confusion_matrix=cm
    ).plot(ax=ax)

    ax.set_title(name)

    col.pyplot(fig)


# ROC curves
st.subheader("ROC Curves")

fig, ax = plt.subplots(figsize=(9, 6))

for name, probability in probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    ax.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Chance"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Test Set")
ax.legend()

st.pyplot(fig)


# Train/test comparison
st.subheader("Training vs Test Accuracy")

comparison = results_df[
    ["Train Accuracy", "Test Accuracy"]
]

fig, ax = plt.subplots(figsize=(8, 5))

comparison.plot(
    kind="bar",
    ax=ax
)

ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Test Accuracy")

plt.xticks(rotation=0)

st.pyplot(fig)


st.success(
    "The application has successfully trained and evaluated "
    "Logistic Regression, KNN, and Decision Tree models."
)

st.caption(
    "This application is an educational machine-learning demonstration "
    "and is not intended for clinical decision-making."
)

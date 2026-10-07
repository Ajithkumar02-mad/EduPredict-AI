import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: white;
    margin-bottom: 25px;
}
.hero h1 {font-size: 42px; margin-bottom: 5px;}
.hero p {font-size: 18px; opacity: .9;}
.metric-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,.25);
    text-align: center;
}
.result-pass {
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    background: rgba(34,197,94,.12);
    border: 1px solid rgba(34,197,94,.45);
}
.result-fail {
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    background: rgba(239,68,68,.12);
    border: 1px solid rgba(239,68,68,.45);
}
.small-note {font-size: 13px; opacity: .75;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA GENERATION
# =========================================================
@st.cache_data
def generate_dataset(n=600, seed=42):
    rng = np.random.default_rng(seed)

    study = np.clip(rng.normal(6.2, 2.5, n), 0.5, 12)
    attendance = np.clip(rng.normal(78, 12, n), 45, 100)
    previous = np.clip(rng.normal(67, 15, n), 25, 100)
    assignments = np.clip(rng.normal(76, 14, n), 25, 100)
    sleep = np.clip(rng.normal(7.0, 1.1, n), 4, 10)
    participation = np.clip(rng.normal(68, 17, n), 20, 100)

    # Synthetic educational-risk score used only to create a realistic
    # demonstration dataset with some overlap/noise.
    score = (
        0.30 * (study / 12 * 100)
        + 0.22 * attendance
        + 0.22 * previous
        + 0.12 * assignments
        + 0.07 * participation
        + 0.07 * (sleep / 10 * 100)
        + rng.normal(0, 8, n)
    )

    result = np.where(score >= 67, "Pass", "Fail")

    return pd.DataFrame({
        "StudyHours": np.round(study, 1),
        "Attendance": np.round(attendance, 1),
        "PreviousMark": np.round(previous, 1),
        "AssignmentScore": np.round(assignments, 1),
        "SleepHours": np.round(sleep, 1),
        "Participation": np.round(participation, 1),
        "Result": result
    })

df = generate_dataset()

FEATURES = [
    "StudyHours",
    "Attendance",
    "PreviousMark",
    "AssignmentScore",
    "SleepHours",
    "Participation"
]
TARGET = "Result"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# MODELS
# =========================================================
models = {
    "SVM (Linear)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", C=1.0, probability=True, random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42
    )
}

trained_models = {}
evaluation_rows = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index("Pass")]

    evaluation_rows[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label="Pass"),
        "Recall": recall_score(y_test, pred, pos_label="Pass"),
        "F1 Score": f1_score(y_test, pred, pos_label="Pass"),
        "ROC AUC": roc_auc_score((y_test == "Pass").astype(int), proba)
    }
    trained_models[name] = model

metrics_df = pd.DataFrame(evaluation_rows).T

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎓 EduPredict AI")
st.sidebar.caption("Advanced Student Performance Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Model Comparison",
        "📈 Data Analytics",
        "🧪 Model Evaluation",
        "📜 Prediction History",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()
st.sidebar.write("### Dataset")
st.sidebar.write(f"Records: **{len(df)}**")
st.sidebar.write(f"Features: **{len(FEATURES)}**")
st.sidebar.write(f"Target: **Pass / Fail**")

# =========================================================
# SESSION HISTORY
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🎓 EduPredict AI</h1>
    <p>Advanced Student Performance Prediction & Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.subheader("📌 Executive Dashboard")

    best_model = metrics_df["F1 Score"].idxmax()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Students Analyzed", len(df))
    c2.metric("Best Model", best_model)
    c3.metric("Best Accuracy", f"{metrics_df.loc[best_model, 'Accuracy']*100:.1f}%")
    c4.metric("Best ROC-AUC", f"{metrics_df.loc[best_model, 'ROC AUC']:.3f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("🎯 Overall Result Distribution")
        counts = df["Result"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Number of Students")
        ax.set_title("Pass vs Fail")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("📚 Study Hours vs Previous Mark")
        fig, ax = plt.subplots()
        for label in ["Pass", "Fail"]:
            subset = df[df["Result"] == label]
            ax.scatter(
                subset["StudyHours"],
                subset["PreviousMark"],
                label=label,
                alpha=0.65
            )
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Previous Mark")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

# =========================================================
# PREDICTION
# =========================================================
elif page == "🔮 Prediction":

    st.subheader("🔮 Individual Student Prediction")
    st.write("Enter student information and compare predictions across multiple ML models.")

    model_name = st.selectbox(
        "Select Prediction Model",
        list(trained_models.keys()),
        index=1
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        study = st.slider("📚 Study Hours / Day", 0.5, 12.0, 6.0, 0.5)
        attendance = st.slider("🗓️ Attendance (%)", 40.0, 100.0, 78.0, 1.0)

    with col2:
        previous = st.slider("📝 Previous Mark", 0.0, 100.0, 65.0, 1.0)
        assignments = st.slider("📑 Assignment Score", 0.0, 100.0, 75.0, 1.0)

    with col3:
        sleep = st.slider("😴 Sleep Hours", 3.0, 10.0, 7.0, 0.5)
        participation = st.slider("🙋 Participation (%)", 0.0, 100.0, 70.0, 1.0)

    student = pd.DataFrame([{
        "StudyHours": study,
        "Attendance": attendance,
        "PreviousMark": previous,
        "AssignmentScore": assignments,
        "SleepHours": sleep,
        "Participation": participation
    }])

    if st.button("🚀 Predict Performance", use_container_width=True):
        selected_model = trained_models[model_name]
        prediction = selected_model.predict(student)[0]
        probabilities = selected_model.predict_proba(student)[0]
        pass_index = list(selected_model.classes_).index("Pass")
        pass_probability = probabilities[pass_index]

        if prediction == "Pass":
            st.markdown(
                f'<div class="result-pass"><h2>🎉 PREDICTED: PASS</h2>'
                f'<p>Pass Probability: <b>{pass_probability*100:.1f}%</b></p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-fail"><h2>⚠️ PREDICTED: FAIL</h2>'
                f'<p>Pass Probability: <b>{pass_probability*100:.1f}%</b></p></div>',
                unsafe_allow_html=True
            )

        st.subheader("📊 Prediction Confidence")
        st.progress(float(pass_probability))

        history_record = student.copy()
        history_record["Model"] = model_name
        history_record["Prediction"] = prediction
        history_record["PassProbability"] = round(pass_probability * 100, 2)
        st.session_state.history.append(history_record)

        st.subheader("Student Profile")
        st.dataframe(student, use_container_width=True)

        st.info(
            "This prediction is an educational demonstration and should not "
            "be used as the sole basis for academic decisions."
        )

# =========================================================
# MODEL COMPARISON
# =========================================================
elif page == "📊 Model Comparison":

    st.subheader("📊 Machine Learning Model Comparison")

    display_metrics = metrics_df.copy() * 100
    st.dataframe(
        display_metrics.style.format("{:.2f}%"),
        use_container_width=True
    )

    st.subheader("🏆 Performance Comparison")

    metric_choice = st.selectbox(
        "Select metric",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    )

    fig, ax = plt.subplots()
    ax.bar(metrics_df.index, metrics_df[metric_choice])
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_choice)
    ax.set_title(f"Model Comparison — {metric_choice}")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig, clear_figure=True)

    st.success(
        f"Best model according to {metric_choice}: "
        f"**{metrics_df[metric_choice].idxmax()}**"
    )

# =========================================================
# DATA ANALYTICS
# =========================================================
elif page == "📈 Data Analytics":

    st.subheader("📈 Student Data Analytics")

    feature = st.selectbox("Select feature for analysis", FEATURES)

    c1, c2 = st.columns(2)

    with c1:
        st.write(f"### Distribution of {feature}")
        fig, ax = plt.subplots()
        ax.hist(df[feature], bins=20)
        ax.set_xlabel(feature)
        ax.set_ylabel("Students")
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.write(f"### {feature} by Result")
        grouped = df.groupby("Result")[feature].mean()
        fig, ax = plt.subplots()
        ax.bar(grouped.index, grouped.values)
        ax.set_ylabel(f"Average {feature}")
        st.pyplot(fig, clear_figure=True)

    st.subheader("🔗 Feature Correlation Matrix")
    corr = df[FEATURES].corr()

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(FEATURES)
    ax.set_title("Feature Correlation")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig, clear_figure=True)

    st.subheader("📋 Descriptive Statistics")
    st.dataframe(df[FEATURES].describe().T, use_container_width=True)

# =========================================================
# MODEL EVALUATION
# =========================================================
elif page == "🧪 Model Evaluation":

    st.subheader("🧪 Detailed Model Evaluation")

    selected = st.selectbox("Select model", list(trained_models.keys()))
    model = trained_models[selected]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index("Pass")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, predictions)*100:.2f}%")
    c2.metric("Precision", f"{precision_score(y_test, predictions, pos_label='Pass')*100:.2f}%")
    c3.metric("Recall", f"{recall_score(y_test, predictions, pos_label='Pass')*100:.2f}%")
    c4.metric("F1 Score", f"{f1_score(y_test, predictions, pos_label='Pass')*100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions, labels=["Fail", "Pass"])
    fig, ax = plt.subplots()
    image = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(selected)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(
        (y_test == "Pass").astype(int),
        probabilities
    )
    auc = roc_auc_score((y_test == "Pass").astype(int), probabilities)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# =========================================================
# HISTORY
# =========================================================
elif page == "📜 Prediction History":

    st.subheader("📜 Prediction History")

    if st.session_state.history:
        history = pd.concat(st.session_state.history, ignore_index=True)
        st.dataframe(history, use_container_width=True)

        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Prediction History CSV",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions have been made in this session yet.")

# =========================================================
# ABOUT
# =========================================================
elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About the Project")

    st.markdown("""
### Student Performance Prediction Using Machine Learning

**EduPredict AI** is an advanced academic micro-project demonstrating how
machine learning can be used to classify student performance.

#### 🎯 Objective
Predict whether a student is likely to **Pass or Fail** based on multiple
academic and behavioral indicators.

#### 🤖 Machine Learning Models
- Support Vector Machine — Linear Kernel
- Support Vector Machine — RBF Kernel
- Random Forest
- Logistic Regression
- Gradient Boosting

#### 📥 Features
- Study Hours
- Attendance
- Previous Mark
- Assignment Score
- Sleep Hours
- Participation

#### 📊 Evaluation
The application evaluates models using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Classification Report

#### 🛠️ Technology Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

#### ⚠️ Academic Disclaimer
The included dataset is synthetically generated for demonstration.
It is not a validated educational dataset and must not be used for real
student assessment or institutional decision-making.
""")

st.divider()
st.caption("EduPredict AI | Advanced AI/ML Micro Project")

# 🎓 EduPredict AI — Advanced Student Performance Prediction

## Overview

EduPredict AI is an advanced Streamlit-based machine learning micro-project
for predicting student performance as **Pass / Fail**.

Unlike a basic SVM demonstration, this version provides a complete ML workflow:

- Synthetic dataset generation with 600 student records
- Six student-related features
- Multiple machine learning models
- SVM Linear and RBF kernels
- Model comparison
- Accuracy, Precision, Recall and F1 Score
- ROC-AUC and ROC curves
- Confusion matrices
- Dataset analytics
- Feature correlation analysis
- Individual student prediction
- Prediction probability/confidence
- Session prediction history
- CSV export
- Multi-page dashboard navigation

## Project Structure

```text
EduPredict_AI/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Workflow

```text
Synthetic Student Dataset
          ↓
Data Preparation
          ↓
Train / Test Split
          ↓
Feature Scaling
          ↓
Multiple ML Models
          ↓
Model Evaluation
          ↓
Best Model Selection
          ↓
New Student Input
          ↓
Pass / Fail + Probability
```

## Models

1. SVM — Linear Kernel
2. SVM — RBF Kernel
3. Random Forest
4. Logistic Regression
5. Gradient Boosting

## Features

- Study Hours
- Attendance
- Previous Mark
- Assignment Score
- Sleep Hours
- Participation

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Main Dashboard Sections

### Dashboard
Provides high-level project metrics, dataset distribution and visual analysis.

### Prediction
Allows the user to enter a student's details and generate a Pass/Fail prediction
with probability.

### Model Comparison
Compares all models using multiple evaluation metrics.

### Data Analytics
Explores distributions, averages, correlations and descriptive statistics.

### Model Evaluation
Provides confusion matrix, classification report, ROC curve and detailed metrics.

### Prediction History
Stores predictions during the current session and allows CSV export.

## Important Academic Note

The dataset is synthetically generated to make the project self-contained and
easy to deploy. It is intended for demonstration and academic presentation,
not real student assessment.

For a production system, use a sufficiently large, representative and
validated dataset, perform proper data governance, assess fairness, and
validate the model with domain experts.

## Suggested Presentation Flow

1. Problem Statement
2. Objective
3. Existing System
4. Proposed System
5. Dataset and Features
6. Data Preprocessing
7. SVM Algorithm
8. Other Models
9. Model Evaluation
10. Dashboard Demonstration
11. Prediction Demonstration
12. Limitations
13. Future Scope
14. Conclusion

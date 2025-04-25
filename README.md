# Credit Risk Prediction for Loan Applicants(Crisk)

## Overview

This repository contains two main programs for predicting credit risk using the German Credit dataset:

- **Jupyter/Colab Notebook**: For in-depth data exploration, model development, and performance analysis.
- **Streamlit Web App (`credit_fraud_ui.py`)**: For interactive data upload, model training, and evaluation via a user-friendly web interface.

Both solutions aim to classify loan applicants as **good** or **bad** credit risks, helping financial institutions minimize defaults and improve lending decisions.

---

## Features

- **Data Exploration & Preprocessing**
  - Handles missing values and outliers
  - One-hot encoding for categorical variables
  - SMOTE oversampling to address class imbalance
- **Model Development**
  - Supports LightGBM, XGBoost, Random Forest, and Gradient Boosting (GBM)
  - Hyperparameter tuning via GridSearchCV
  - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Model Interpretation**
  - Feature importance visualization
  - Confusion matrix and ROC curve plots
- **Error Analysis**
  - Misclassification analysis and discussion of data/model limitations
- **Actionable Insights**
  - Recommendations for improving credit evaluation

---

## Getting Started

### 1. Requirements

- Python 3.7+
- Recommended: Google Colab or Jupyter Notebook for the notebook version
- For the Streamlit app:
  - Streamlit (`pip install streamlit`)
  - scikit-learn, pandas, numpy, matplotlib, imbalanced-learn, xgboost, lightgbm

### 2. Running the Notebook

1. Open the notebook in Google Colab or Jupyter.
2. Upload the `german_credit_data.csv` dataset.
3. Execute the cells in order:
   - Data loading and preprocessing
   - Model training and evaluation (LightGBM, Random Forest, XGBoost, GBM)
   - Visualizations and insights

### 3. Running the Streamlit App

1. Install dependencies:
   ```bash
   pip install streamlit scikit-learn pandas numpy matplotlib imbalanced-learn xgboost lightgbm
   ```
2. Run the app:
   ```bash
   streamlit run credit_fraud_ui.py
   ```
3. Upload your credit data CSV through the web interface.
4. Select preprocessing options, model type, and hyperparameters.
5. Train the model and view performance metrics and visualizations.

---

## File Structure

```
.
├── paste.txt                # Jupyter/Colab notebook code for credit risk prediction
├── credit_fraud_ui.py       # Streamlit web application
├── german_credit_data.csv   # (You must provide this dataset)
```

---

## Model Choices

- **LightGBM**: Fast, efficient, handles categorical data well, robust to missing values.
- **XGBoost**: High recall, strong for complex feature interactions, requires careful tuning.
- **Random Forest**: Reliable baseline, handles non-linearities, robust to outliers.
- **GBM**: Flexible, good for sequential error correction, slower but effective.

Each model is hyperparameter-tuned and evaluated using standard classification metrics.

---

## Insights & Recommendations

- **Key features**: Credit amount, loan duration, age, account status, employment history.
- **Improvements**: Integrate alternative data sources, use model stacking, and dynamic thresholding.
- **Benefits**: Reduces financial risk, improves accuracy, and enhances customer trust.

---

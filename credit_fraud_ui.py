# credit_fraud_ui.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# ======================
# UI Configuration
# ======================
st.set_page_config(
    page_title="Credit Fraud Analytics",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stSelectbox label { font-weight: bold; color: #2c3e50; }
    .metric-box { padding: 15px; border-radius: 10px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ======================
# Core Functions
# ======================


def load_data(uploaded_file):
    """Load and preprocess data"""
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=lambda x: x.strip().replace(' ', '_').lower())
    return df


def preprocess_data(df, handle_nan):
    """Handle missing values"""
    if handle_nan == "Drop NA":
        return df.dropna()
    else:
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df


def train_model(model_type, X_train, y_train, params):
    """Model training wrapper"""
    models = {
        'LightGBM': lgb.LGBMClassifier,
        'XGBoost': xgb.XGBClassifier,
        'Random Forest': RandomForestClassifier,
        'GBM': GradientBoostingClassifier
    }
    model = models[model_type](**params)
    model.fit(X_train, y_train)
    return model

# Clean feature names for LightGBM compatibility


def clean_feature_names(df):
    # Remove special JSON characters: , { } [ ] " :
    df.columns = df.columns.str.replace(r'[:]', '', regex=True)
    # Replace spaces and other special characters with underscores
    # Ensure unique names after cleaning
    df.columns = [f'{col}_{i}' if df.columns[:i].tolist().count(col) > 0 else col
                  for i, col in enumerate(df.columns)]
    return df


# ======================
# Main Application
# ======================
def main():
    st.title("🔍 Credit Fraud Detection Analytics Workbench(By Sayandeep Dey)")
    st.markdown("---")

    # ======================
    # Data Upload Section
    # ======================
    with st.expander("📁 Data Upload & Preprocessing", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload Credit Data (CSV)", type="csv")

        if uploaded_file:
            df = load_data(uploaded_file)
            st.session_state.raw_data = df

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Preview")
                st.dataframe(df.head().style.highlight_null(
                    color='#ffcccb'), height=250)

            with col2:
                st.subheader("Data Summary")
                st.write(f"Total Records: {df.shape[0]}")
                st.write(f"Features: {df.shape[1]}")
                st.write("Missing Values:")
                st.write(df.isna().sum())

            # Preprocessing controls
            st.subheader("Preprocessing Options")
            handle_nan = st.radio("Handle Missing Values:",
                                  ["Drop NA", "Impute Median"], horizontal=True)

            df = preprocess_data(df, handle_nan)
            st.session_state.processed_data = df

    # ======================
    # Model Configuration
    # ======================
    if 'processed_data' in st.session_state:
        st.markdown("---")
        with st.expander("⚙️ Model Configuration", expanded=True):
            model_type = st.selectbox("Select Model:",
                                      ["LightGBM", "XGBoost", "Random Forest", "GBM"])

            # Dynamic parameter controls
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                n_estimators = st.slider("Number of Trees", 50, 500, 100)
                max_depth = st.slider("Max Depth", 3, 15, 5)

            with param_col2:
                learning_rate = st.slider(
                    "Learning Rate", 0.01, 0.5, 0.1, step=0.01)
                subsample = st.slider("Subsample Ratio", 0.5, 1.0, 0.8)

            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'random_state': 42
            }

            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    # Prepare data
                    train = st.session_state.processed_data.sample(frac=0.9)
                    test = st.session_state.processed_data.drop(train.index)

                    X_train = train.drop(columns=['risk'])
                    y_train = train['risk'].map({'bad': 0, 'good': 1})
                    X_test = test.drop(columns=['risk'])
                    y_test = test['risk'].map({'bad': 0, 'good': 1})
                    X_train = clean_feature_names(X_train)
                    X_test = clean_feature_names(X_test)

                    # Handle categorical features
                    X_train = pd.get_dummies(X_train)
                    X_test = pd.get_dummies(X_test)
                    X_train, X_test = X_train.align(
                        X_test, join='left', axis=1)
                    X_test = X_test.fillna(0)

                    # Handle class imbalance
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)

                    # Train model
                    model = train_model(model_type, X_train, y_train, params)
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.success("Model trained successfully!")

    # ======================
    # Results Visualization
    # ======================
    if 'model' in st.session_state:
        st.markdown("---")
        with st.expander("📊 Model Evaluation", expanded=True):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            # Generate predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(
                model, "predict_proba") else y_pred

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            with col2:
                st.metric(
                    "Precision", f"{precision_score(y_test, y_pred):.2%}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
            with col4:
                st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")

            # Visualizations
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
                st.pyplot(fig)

            with fig_col2:
                st.subheader("ROC Curve")
                fig, ax = plt.subplots()
                RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
                st.pyplot(fig)

            # Feature Importance
            st.subheader("Feature Importance")
            if hasattr(model, 'feature_importances_'):
                X_train = st.session_state.get('X_train')
                if X_train is not None:
                    importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.bar_chart(importance.set_index('Feature'))
                else:
                    st.warning(
                        "Training data not found. Please train the model first.")
            else:
                st.warning(
                    "Feature importance not available for this model type")


if __name__ == "__main__":
    main()

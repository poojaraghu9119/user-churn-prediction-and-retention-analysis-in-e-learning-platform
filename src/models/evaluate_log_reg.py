# IMPORTING THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, 
precision_score, recall_score, f1_score)

# PATHS
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "saved_model"

# LOAD THE DATASET
X_test = pd.read_csv(DATA_DIR/ "X_test.csv")
y_test = pd.read_csv(DATA_DIR/ "y_test.csv")
X_train = pd.read_csv(DATA_DIR/ "X_train.csv")

# LOAD THE TRANSFORMERS
from src.features.custom_transformers import (
    FeatureEngineer,
    MissingValueImputer,
    MappingEncoder,
    FrequencyEncoder,
    OutlierHandler,
    FeatureDropper
)

# IMPORTING THE PREPROCESSER OF TRANSFORMING AND SCALING FROM train_log_reg.py
from src.models.train_log_reg import preprocessor

# LOAD THE MODEL
model_path = MODEL_DIR / "logistic_regression_base_model.joblib"
log_reg_base_model = joblib.load(model_path)

# PREDICTION
y_pred = log_reg_base_model.predict(X_test)
y_pred_proba = log_reg_base_model.predict_proba(X_test)[:, 1]

# EVALUATION
print("Evaluating the model with the default threshold 0.5:")
print("---------------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# PREDICTION AND EVALUATION USING THRESHOLD = 0.9
y_pred_09 = (y_pred_proba >= 0.9).astype(int)

print("Evaluating the model with threshold = 0.9:")
print("---------------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred_09))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred_09))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_09))

# The results are better when the threshold = 0.9:

# Pasting the results here:

log_reg_metrics = {
    "model": "Logistic Regression",
    "accuracy": 0.9788789982759429,
    "roc_auc": 0.9929508592095224,

    "precision_0": 1.00,
    "recall_0": 0.98,
    "f1_0": 0.99,

    "precision_1": 0.57,
    "recall_1": 0.92,
    "f1_1": 0.71,

    "TN": 183072,
    "FP": 3641,
    "FN": 414,
    "TP": 4862
}


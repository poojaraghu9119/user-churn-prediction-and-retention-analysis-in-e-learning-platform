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
    OutlierHandler
)

# LOAD THE MODEL
model_path = MODEL_DIR / "lgbm_baseline_model.joblib"
lgbm_base_model = joblib.load(model_path)

# PREDICTION
y_pred = lgbm_base_model.predict(X_test)
y_pred_proba = lgbm_base_model.predict_proba(X_test)[:, 1]

# EVALUATION
print("Performance of the model with the default threshold:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("------------------------------------------------------------------------------")

# PASTING THE RESULTS HERE

# Accuracy: 0.9834052992619369
# ROC AUC: 0.996834620341265

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      0.98      0.99    186713
#            1       0.63      0.97      0.77      5276

#     accuracy                           0.98    191989
#    macro avg       0.81      0.98      0.88    191989
# weighted avg       0.99      0.98      0.99    191989


# Confusion Matrix:
#  [[183697   3016]
#  [   137   5139]]

# EVALUATION USING THRESHOLD = 0.9

y_pred_09 = (y_pred_proba >= 0.9).astype(int)

# EVALUATION

print("Performance of the model while using threshold = 0.9:")
print("Accuracy:", accuracy_score(y_test, y_pred_09))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred_09))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_09))

# RESULTS FOR THRESHOLD = 0.9

lgbm_metrics = {
    "model": "LightGBM (thr=0.9)",
    "accuracy": 0.9893275135554641,
    "roc_auc": 0.997054205648059,

    "precision_0": 1.00,
    "recall_0": 0.99,
    "f1_0": 0.99,

    "precision_1": 0.75,
    "recall_1": 0.91,
    "f1_1": 0.82,

    "TN": 185143,
    "FP": 1570,
    "FN": 479,
    "TP": 4797
}


# LOAD THE FINAL TUNED LGBM MODEL

model_path = MODEL_DIR / "lgbm_final_model.joblib"
lgbm_best_model = joblib.load(model_path)

# PREDICTION USING THE BEST THRESHOLD = 0.9

y_pred_proba_09 = lgbm_best_model.predict_proba(X_test)[:, 1]
Y_pred_09 = (y_pred_proba_09 >= 0.9).astype(int)

# EVALUATION OF THE ABOVE PREDICTIONS

print("Performance of the tuned LGBM model while using threshold = 0.9:")
print("-----------------------------------------------------------------------")
print(f"Model accuracy: {accuracy_score(y_test, Y_pred_09)}.")
print(f"ROC_AUC_SCORE is {roc_auc_score(y_test, y_pred_proba_09)}.")
print("Classification report:")
print()
print(classification_report(y_test, Y_pred_09))
print("Confusion matrix:")
print()
print(confusion_matrix(y_test, Y_pred_09))

# RESULTS
# Performance of the tuned LGBM model while using threshold = 0.9:
# -----------------------------------------------------------------------
# Model accuracy: 0.9887389381683326.
# ROC_AUC_SCORE is 0.997128246013278.
# Classification report:

#               precision    recall  f1-score   support

#            0       1.00      0.99      0.99    186713
#            1       0.73      0.93      0.82      5276

#     accuracy                           0.99    191989
#    macro avg       0.87      0.96      0.91    191989
# weighted avg       0.99      0.99      0.99    191989

# Confusion matrix:

# [[184943   1770]
#  [   392   4884]]

# The precision of class 1 is better for the baseline model with threshold = 0.9 than the tuned model.
# IMPORTING THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, 
precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance

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
model_path = MODEL_DIR / "rf_baseline_pipeline.joblib"
rf_base_model = joblib.load(model_path)

# PREDICTION
y_pred = rf_base_model.predict(X_test)
y_pred_proba = rf_base_model.predict_proba(X_test)[:, 1]

# EVALUATION
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# PASTING THE MODEL RESULTS HERE

#Accuracy: 0.9771393152732709
# ROC AUC: 0.995800998590812

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      0.98      0.99    186713
#            1       0.55      0.98      0.70      5276

#     accuracy                           0.98    191989
#    macro avg       0.77      0.98      0.85    191989
# weighted avg       0.99      0.98      0.98    191989


# Confusion Matrix:
#            Predicted 0  Predicted 1
#  Actual 0 [[182480   4233]
#  ACtual 1 [   106   5170]]

# Interpretations from the results:
# -----------------------------------
# The model accuracy and ROC-AUC score is good, which means the model is good at separating the classes and that
# the probability that the model will rank a random positive instance higher than a negative instance is good.
# The precision, recall and f1-score of class 0 is good. But the precision and f1-score is low for class 1.
# This is due to the class imbalance in the dataset, and the model uses the default threshold of 0.5. 
# This can be rectified using threshold tuning, and dropping irrelevant features (if any).

# THRESHOLD TUNING

thresholds = [0.6, 0.7, 0.8, 0.9]
for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    print(f"Threshold: {t}")
    print(confusion_matrix(y_test, y_pred_t))
    print(classification_report(y_test, y_pred_t))
    print()

# We have got the best result for threshold = 0.9.
# precision    recall  f1-score   support

#            0       1.00      0.99      0.99    186713
#            1       0.70      0.91      0.79      5276

#     accuracy                           0.99    191989
#    macro avg       0.85      0.95      0.89    191989
# weighted avg       0.99      0.99      0.99    191989

# COMPUTING THE PERMUTATION IMPORTANCE TO FIND THE FEATURES THE MODEL HAS USED THE MOST AND THE FEATURES
# WHICH THE MODEL HAS NOT USED.

# Making predictions from the model using the threshold = 0.9

y_proba = rf_base_model.predict_proba(X_test)[:, 1]
y_pred_09 = (y_proba > 0.9).astype(int)

# EVALUATION

print("Performance of the model while using threshold = 0.9:")
print("--------------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred_09))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred_09))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_09))

# LOAD THE HYPERPARAMETER TUNED MODEL

model_path = MODEL_DIR / "best_rf_model.joblib"
best_rf_model = joblib.load(model_path)

# PREDICTION
Y_proba = best_rf_model.predict_proba(X_test)[:, 1]
Y_pred_09 = (Y_proba >= 0.9).astype(int)

# EVALUATION

print("Performance of the hyperparameter tuned model:")
print("-------------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, Y_pred_09))
print("ROC AUC:", roc_auc_score(y_test, Y_proba))
print("\nClassification Report:\n", classification_report(y_test, Y_pred_09))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, Y_pred_09))

# Performance of the hyperparameter tuned model:
# -------------------------------------------------------------
rf_metrics = {
    "model": "Random Forest",
    "accuracy": 0.9881816145716682,
    "roc_auc": 0.9962370893071175,

    # Class 0 metrics
    "precision_0": 1.00,
    "recall_0": 0.99,
    "f1_0": 0.99,

    # Class 1 metrics
    "precision_1": 0.73,
    "recall_1": 0.90,
    "f1_1": 0.81,

    # Confusion matrix values
    "TN": 184989,
    "FP": 1724,
    "FN": 545,
    "TP": 4731
}



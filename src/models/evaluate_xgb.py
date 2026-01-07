# IMPORTING THE REQUIRED LIBRARIES

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)

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
model_path = MODEL_DIR / "xgb_baseline_model.joblib"
xgb_base_model = joblib.load(model_path)

# PREDICTION USING THE DEFAULT THRESHOLD
y_pred_proba = xgb_base_model.predict_proba(X_test)[:, 1]
y_pred = xgb_base_model.predict(X_test)

# EVALUATION

print("Evaluation using the default threshold, 0.5:")
print("--------------------------------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC_AUC_SCORE: {roc_auc_score(y_test, y_pred_proba)}")
print("Classification report of the model:")
print()
print(classification_report(y_test, y_pred))
print("Confusion matrix of the model is:")
print()
print(confusion_matrix(y_test, y_pred))

# PASTING THE RESULTS OF THE BASELINE MODEL

# Evaluation using the default threshold, 0.5:
# --------------------------------------------------
# Accuracy: 0.9876503341337264
# ROC_AUC_SCORE: 0.9971496032838518
# Classification report of the model:

#               precision    recall  f1-score   support

#            0       1.00      0.99      0.99    186713
#            1       0.70      0.95      0.81      5276

#     accuracy                           0.99    191989
#    macro avg       0.85      0.97      0.90    191989
# weighted avg       0.99      0.99      0.99    191989

# Confusion matrix of the model is:

# [[184601   2112]
#  [   281   4995]]

# The accuracy and roc_auc_score are good. That means, the model is able to separate the classes well and its
# probability of ranking a random positive instance higher than a random negative instance is good.
# The precision, recall, and f1-score for class 0 is good, while the precision, recall, and f1-score of class 1
# is low. 
# Class 1 means that the student will complete the course. Low precision of class 1 means that out of the total
# class 1 predictions, only 70% students actually completed the course. Remaining 30% did not complete the
# course. This is because of the parameter scale_pos_weight. This parameter penalizes more if the model predicts
# "NOT COMPLETED" for a student who actually completed the course. So, the model pushes many borderline
# probabilities like (0.51, 0.53) to COMPLETED. This can be fixed using tuning the threshold.

# THRESHOLD TUNING

threshold = [0.6, 0.7, 0.8, 0.9]
for t in threshold:
    y_pred_t = (y_pred_proba >= t).astype(int)
    print(f"Threshold: {t}")
    print(confusion_matrix(y_test, y_pred_t))
    print(classification_report(y_test, y_pred_t))
    print()

# The results for threshold 0.8 and 0.9 are good.

# Threshold: 0.8
# [[185270   1443]
#  [   513   4763]]
#               precision    recall  f1-score   support

#            0       1.00      0.99      0.99    186713
#            1       0.77      0.91      0.83      5276

#     accuracy                           0.99    191989
#    macro avg       0.88      0.95      0.91    191989
# weighted avg       0.99      0.99      0.99    191989


# Threshold: 0.9
# [[185584   1129]
#  [   716   4560]]
#               precision    recall  f1-score   support

#            0       1.00      0.99      1.00    186713
#            1       0.80      0.86      0.83      5276

#     accuracy                           0.99    191989
#    macro avg       0.90      0.93      0.91    191989
# weighted avg       0.99      0.99      0.99    191989

# For threshold = 0.8, precision for class 1 is 0.77, and recall is 0.9. While, for threshold = 0.9, 
# precision increases but recall drops below 0.9. This shows increase in beta errors, that is, the model
# predicts more students will actually complete the course as not completed. This error has to be minimized.
# However, we can decide between 0.8 and 0.9 after tuning the hyperparameters of the model.

# The hyperparameter tuning is done and the final model is saved.
# LOAD THE FINAL MODEL

model_path = MODEL_DIR / "xgb_final_model.joblib"
xgb_best_model = joblib.load(model_path)

# PREDICTION USING THRESHOLD = 0.8

y_pred_proba_08 = xgb_best_model.predict_proba(X_test)[:, 1]
y_pred_08 = (y_pred_proba_08 >= 0.8).astype(int)

# Evaluation of the above predictions
print("Model performance with threshold = 0.8")
print(f"Accuracy of the final xgboost model is {accuracy_score(y_test, y_pred_08)}.")
print(f"ROC_AUC_SCORE is {roc_auc_score(y_test, y_pred_proba_08)}.")
print("Classification report:")
print()
print(classification_report(y_test, y_pred_08))
print("Confusion matrix:")
print()
print(confusion_matrix(y_test, y_pred_08))

# PREDICTION USING THRESHOLD = 0.9

y_pred_proba_09 = xgb_best_model.predict_proba(X_test)[:, 1]
y_pred_09 = (y_pred_proba_09 >= 0.9).astype(int)

# Evaluation of the above predictions
print("Model performance with threshold = 0.9")
print(f"Accuracy of the final xgboost model is {accuracy_score(y_test, y_pred_09)}.")
print(f"ROC_AUC_SCORE is {roc_auc_score(y_test, y_pred_proba_09)}.")
print("Classification report:")
print()
print(classification_report(y_test, y_pred_09))
print("Confusion matrix:")
print()
print(confusion_matrix(y_test, y_pred_09))

# RESULTS

# Model performance with threshold = 0.8
xgb_08_metrics = {
    "model": "XGBoost (thr=0.8)",
    "accuracy": 0.9887389381683326,
    "roc_auc": 0.9972843807664706,

    "precision_0": 1.00,
    "recall_0": 0.99,
    "f1_0": 0.99,

    "precision_1": 0.73,
    "recall_1": 0.93,
    "f1_1": 0.82,

    "TN": 184888,
    "FP": 1825,
    "FN": 347,
    "TP": 4929
}

# Model performance with threshold = 0.9

xgb_09_metrics = {
    "model": "XGBoost (thr=0.9)",
    "accuracy": 0.9902286068472673,
    "roc_auc": 0.9972533569428744,

    "precision_0": 1.00,
    "recall_0": 0.99,
    "f1_0": 0.99,

    "precision_1": 0.78,
    "recall_1": 0.90,
    "f1_1": 0.84,

    "TN": 185347,
    "FP": 1366,
    "FN": 510,
    "TP": 4766
}


# Threshold = 0.9 gives better results.

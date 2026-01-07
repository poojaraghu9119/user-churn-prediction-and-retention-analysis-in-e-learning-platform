# THIS FILE IS USED TO TRAIN THE XGBOOST CLASSIFIER MODEL

# IMPORTING THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

# IMPORTING THE DATASETS

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR/ "data" / "processed"

X_train = pd.read_csv(DATA_DIR/ "X_train.csv")
X_test = pd.read_csv(DATA_DIR/ "X_test.csv")
y_train = pd.read_csv(DATA_DIR/ "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR/ "y_test.csv").squeeze()

# IMPORTING THE REQUIRED CUSTOM TRANSFORMERS
from src.features.custom_transformers import (FeatureEngineer, MissingValueImputer, MappingEncoder, 
                                              FrequencyEncoder, OutlierHandler)

# CREATING TRANSFORMER OBJECTS

feature_engineer = FeatureEngineer()
missing_imputer = MissingValueImputer(missing_num_cols = ['nevents', 'ndays_act', 'nchapters', 
                                                          'course_duration_days', 'student_age_at_start', 
                                                          'avg_events_per_active_day', 
                                                          'avg_chapters_per_active_day', 'activity_ratio'],
                                      missing_cat_cols = ["LoE_DI", "gender"])
map_encoder = MappingEncoder()
frequency_encoder = FrequencyEncoder(cols = ["course_id", "final_cc_cname_DI"])
outlier_handler = OutlierHandler()

# DEFINING THE BASELINE XGBOOST MODEL

# Calculating the scale_pos_weight
scale_pos_weight = ((y_train == 0).sum()) / ((y_train == 1).sum())
xgb_baseline = XGBClassifier(n_estimators = 300, max_depth = 8, subsample = 0.8, learning_rate = 0.1, 
                             colsample_bytree = 0.8, objective = "binary:logistic", eval_metric = "auc",
                             scale_pos_weight = scale_pos_weight, tree_method = "hist", n_jobs = -1,
                             random_state = 42)

# CREATING THE PIPELINE

xgb_baseline_pipe = Pipeline(steps = [("feature_engineering", feature_engineer),
                                      ("missing_imputatiom", missing_imputer),
                                      ("mapping_encoding", map_encoder),
                                      ("frequency_encoding", frequency_encoder),
                                      ("outlier_handling", outlier_handler),
                                      ("XGB_baseline_model", xgb_baseline),])

# TRAINING THE MODEL
xgb_baseline_pipe.fit(X_train, y_train)

# SAVING THE MODEL

MODEL_DIR = BASE_DIR / "saved_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "xgb_baseline_model.joblib"
joblib.dump(xgb_baseline_pipe, model_path)

print(f"Baseline XGBoost model saved at: {model_path}")

# TUNING THE HYPERPARAMETERS OF THE MODEL

# Redefining the model:
xgb_model = XGBClassifier(scale_pos_weight = scale_pos_weight, objective = "binary:logistic", 
                          eval_metric = "auc", tree_method = "hist", n_jobs = -1, random_state = 42)

# Defining the pipeline
xgb_model_pipe = Pipeline(steps = [("feature_engineering", feature_engineer),
                                   ("missing_imputation", missing_imputer),
                                   ("mapping_encoding", map_encoder),
                                   ("frequency_encoding", frequency_encoder),
                                   ("outlier_handling", outlier_handler),
                                   ("XGB_final_model", xgb_model),])

# Hyperparameter tuning
params = {"XGB_final_model__n_estimators": [200, 300, 400],
          "XGB_final_model__max_depth": [6, 8, 10],
          "XGB_final_model__learning_rate": [0.05, 0.1, 0.15],
          "XGB_final_model__subsample": [0.7, 0.8, 0.9],
          "XGB_final_model__colsample_bytree": [0.7, 0.8, 0.9],
          "XGB_final_model__min_child_weight": [1,3,5],}

random_search_xgb = RandomizedSearchCV(estimator = xgb_model_pipe, param_distributions=params, n_iter = 20,
                                       scoring = "f1", cv = 5, verbose = 2, n_jobs = -1, random_state = 42)

random_search_xgb.fit(X_train, y_train)

best_xgb_model = random_search_xgb.best_estimator_
best_xgb_params = random_search_xgb.best_params_

print("Best XGBoost params:", best_xgb_params)

# SAVING THE MODEL

MODEL_DIR = BASE_DIR / "saved_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "xgb_final_model.joblib"
joblib.dump(best_xgb_model, model_path)

print(f"Final XGBoost model saved at: {model_path}")







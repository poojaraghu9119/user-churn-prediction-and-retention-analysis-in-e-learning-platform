# IMPORTING THE REQUIRED LIBRARIES

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

# PATHS
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "saved_model"

# IMPORTING THE DATASETS
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# IMPORTING THE TRANSFORMER CLASSES FROM custom_transformers.py

from src.features.custom_transformers import (
    FeatureEngineer,
    MissingValueImputer,
    MappingEncoder,
    FrequencyEncoder,
    OutlierHandler
)

# CREATING TRANSFORMER OBJECTS

feature_engineer = FeatureEngineer()
missing_imputer = MissingValueImputer(missing_num_cols = ['nevents', 'ndays_act', 'nchapters', 
                                                          'course_duration_days', 'student_age_at_start', 
                                                          'avg_events_per_active_day', 
                                                          'avg_chapters_per_active_day', 'activity_ratio'],
                                      missing_cat_cols = ["LoE_DI", "gender"])
mapping_encoder = MappingEncoder()
frequency_encoder = FrequencyEncoder(cols = ["course_id", "final_cc_cname_DI"])
outlier_handler = OutlierHandler()

# DEFINING THE BASELINE LightGBM MODEL:
lgbm_baseline_model = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                                     max_depth=6, num_leaves=31,         # default value
                                     subsample=0.8, colsample_bytree=0.8, objective="binary",
                                     class_weight="balanced", n_jobs=-1, random_state=42)

# CREATING THE BASELINE PIPELINE FOR THE LightGBM MODEL

lgbm_baseline_pipeline = Pipeline(steps = [("feature_engineering", feature_engineer),
                                           ("missing_imputation", missing_imputer),
                                           ("map_encoding", mapping_encoder),
                                           ("frequency_encoding", frequency_encoder),
                                           ("outlier_handling", outlier_handler),
                                           ("lgbm_baseline_model", lgbm_baseline_model),])

# TRAINING THE MODEL
lgbm_baseline_pipeline.fit(X_train, y_train)

# SAVING THE MODEL

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "lgbm_baseline_model.joblib"
joblib.dump(lgbm_baseline_pipeline, model_path)

print(f"Baseline LGBM Model saved at: {model_path}")

# [LightGBM] [Info] Number of positive: 12409, number of negative: 434842
# [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.088685 seconds.
# You can set `force_row_wise=true` to remove the overhead.
# And if memory is not enough, you can set `force_col_wise=true`.
# [LightGBM] [Info] Total Bins 1596
# [LightGBM] [Info] Number of data points in the train set: 447251, number of used features: 28
# [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
# [LightGBM] [Info] Start training from score 0.000000
# [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
# [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
# [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
# Baseline LGBM Model saved at: C:\Projects\student-course-completion-prediction\saved_model\lgbm_baseline_model.joblib

# REDEFINING THE LightGBM MODEL
lgbm_model = LGBMClassifier(objective = "binary", class_weight = "balanced", n_jobs = -1, random_state = 42)

# DEFINING THE PIPELINE
lgbm_model_pipeline = Pipeline(steps = [("feature_engineering", feature_engineer),
                                        ("missing_imputation", missing_imputer),
                                        ("map_encoding", mapping_encoder),
                                        ("frequency_encoding", frequency_encoder),
                                        ("outlier_handling", outlier_handler),
                                        ("lgbm_model", lgbm_model),])

params = {
    "lgbm_model__n_estimators": [200, 300, 400, 600],
    "lgbm_model__learning_rate": [0.15, 0.05, 0.1],
    "lgbm_model__num_leaves": [31, 50, 80, 120],
    "lgbm_model__max_depth": [6, 8, 10],
    "lgbm_model__min_child_samples": [20, 40, 60],
    "lgbm_model__subsample": [0.7, 0.8, 0.9],
    "lgbm_model__colsample_bytree": [0.7, 0.8, 0.9]
}

random_search_lgbm = RandomizedSearchCV(estimator = lgbm_model_pipeline, param_distributions=params, n_iter = 20,
                                        scoring = "f1", cv = 5, n_jobs = -1, verbose = 2, random_state = 42)

# Training the model
random_search_lgbm.fit(X_train, y_train)

best_lgbm_model = random_search_lgbm.best_estimator_
best_params = random_search_lgbm.best_params_

print("Best LightGBM Parameters:")
print(best_params)

# SAVE MODEL
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "lgbm_final_model.joblib"
joblib.dump(best_lgbm_model, model_path)

print(f"Tuned LightGBM model saved at: {model_path}")

# Importing the required libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# Importing the datasets from the processed folder
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"

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

# DEFINING THE BASELINE RANDOM FOREST MODEL
rf_baseline_model = RandomForestClassifier(n_estimators = 200, max_depth = 10, max_features = "sqrt",
                                           class_weight = "balanced", n_jobs = -1, random_state = 42)

# CREATING THE PIPELINE

rf_baseline_pipeline = Pipeline(steps = [("feature_engineering", feature_engineer),
                                         ("missing_imputation", missing_imputer),
                                         ("map_encoding", mapping_encoder),
                                         ("frequency_encoding", frequency_encoder),
                                         ("outlier_handling", outlier_handler),
                                         ("baseline_model", rf_baseline_model),])

# TRAINING THE MODEL
rf_baseline_pipeline.fit(X_train, y_train)

# SAVING THE MODEL
MODEL_DIR = BASE_DIR / "saved_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "rf_baseline_pipeline.joblib"
joblib.dump(rf_baseline_pipeline, model_path)

print(f"Model saved at: {model_path}")

# TUNING THE HYPER PARAMETERS OF THE MODEL TO INCREASE THE MODEL PERFORMANCE

rf = RandomForestClassifier(class_weight = "balanced", n_jobs = -1, random_state = 42)
rf_pipeline = Pipeline(steps = [("feature_engineering", feature_engineer),
                                ("missing_imputation", missing_imputer),
                                ("map_encoding", mapping_encoder),
                                ("frequency_encoding", frequency_encoder),
                                ("outlier_handling", outlier_handler),
                                ("rf_model", rf),])

param_grid = {"rf_model__n_estimators": [200, 300, 400],
              "rf_model__max_depth": [8, 10, 12],
              "rf_model__min_samples_split" :  [2, 5, 10],
              "rf_model__min_samples_leaf": [1,2,5],
              "rf_model__max_features": ["sqrt", "log2"]}

random_search = RandomizedSearchCV(estimator = rf_pipeline, param_distributions=param_grid, n_iter=20,
                                   scoring="f1", cv=5, n_jobs=-1, verbose=2, random_state=42)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_parameters = random_search.best_params_

# SAVING THE MODEL
MODEL_DIR = BASE_DIR / "saved_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "best_rf_model.joblib"
joblib.dump(best_model, model_path)

print(f"The final RF model is saved at: {model_path}")
print(f"Best parameters: {best_parameters}")











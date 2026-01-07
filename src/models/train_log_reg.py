# IMPORTING THE REQUIRED LIBRARIES

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# PATHS
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "saved_model"

# LOAD THE DATASET
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# IMPORT THE TRANSFORMER CLASSES FROM THE custom_transformer.py file
from src.features.custom_transformers import (FeatureEngineer, MissingValueImputer, MappingEncoder, 
                                              FrequencyEncoder, OutlierHandler, FeatureDropper)
# CREATING THE TRANSFORMER OBJECTS
feature_engineer = FeatureEngineer()
missing_imputation = MissingValueImputer(missing_num_cols = ['ndays_act', 'nchapters', "nevents",
                                                          'course_duration_days', 'student_age_at_start', 
                                                          'avg_events_per_active_day', 
                                                          'avg_chapters_per_active_day', 'activity_ratio'],
                                         missing_cat_cols = ["LoE_DI", "gender"])
mapping_encoder = MappingEncoder()
frequency_encoder = FrequencyEncoder(cols = ["course_id", "final_cc_cname_DI"])
outlier_handler = OutlierHandler()
feature_dropper = FeatureDropper(drop_features = ["exploration_rate", "nevents"])

# DEFINING THE FEATURES WHICH NEED TO BE BOTH TRANSFORMED AND SCALED

num_power_scale_features = ['ndays_act', 'nchapters', 'course_duration_days', 'student_age_at_start',
                            'avg_events_per_active_day', 'avg_chapters_per_active_day', 'activity_ratio',
                            'nforum_posts', 'no_of_courses_registered', 'no_of_courses_explored']

num_scale_only_features = ["start_year"]

# Creating the pipeline to transform and then scale the features
power_and_scale = Pipeline(steps=[
    ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
    ("scaler", StandardScaler())
])

# Creating the pipeline to scale the features
scale_only = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num_power", power_and_scale, num_power_scale_features),
        ("num_scale", scale_only, num_scale_only_features)
    ],
    remainder="passthrough"
)

# CREATING THE FINAL PIPELINE

logistic_model = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1, solver="lbfgs")

logistic_pipeline = Pipeline(steps=[
    ("feature_engineering", feature_engineer),
    ("missing_imputation", missing_imputation),
    ("mapping_encoding", mapping_encoder),
    ("frequency_encoding", frequency_encoder),
    ("outlier_handling", outlier_handler),
    ("feature_dropping", feature_dropper),
    ("scaling", preprocessor),
    ("logistic_model", logistic_model)
])

logistic_pipeline.fit(X_train, y_train)

# SAVING THE MODEL

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "logistic_regression_base_model.joblib"
joblib.dump(logistic_pipeline, model_path)



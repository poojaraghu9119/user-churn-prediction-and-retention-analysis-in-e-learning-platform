# THIS IS A FILE WHICH STORES ALL THE CUSTOM TRANSFORMERS.

# Importing the required libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Creating a custom transformer called FeatureEngineer to create new features and drop irrelevant ones.

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived numerical features from existing columns.
    Assumes datetime columns are already parsed in loader.py.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Converting the datatypes of the date columns
        X["start_time_DI"] = pd.to_datetime(X["start_time_DI"], errors="coerce")
        X["last_event_DI"] = pd.to_datetime(X["last_event_DI"], errors="coerce")
        # -----------------------------------
        # Start year from start_time_DI
        # -----------------------------------
        X["start_year"] = X["start_time_DI"].dt.year

        # -----------------------------------
        # Course duration and activity ratio
        # -----------------------------------
        X["course_duration_days"] = (
            X["last_event_DI"] - X["start_time_DI"]
        ).dt.days

        X["activity_ratio"] = np.where(
            X["course_duration_days"] > 0,
            X["ndays_act"] / X["course_duration_days"],
            0
        )

        # -----------------------------------
        # Student age at course start
        # -----------------------------------
        X["student_age_at_start"] = X["start_year"] - X["YoB"]

        # -----------------------------------
        # Exploration rate
        # (safe as per your data guarantees)
        # -----------------------------------
        X["exploration_rate"] = (
            X["no_of_courses_explored"] / X["no_of_courses_registered"]
        )

        # -----------------------------------
        # Average activity metrics
        # -----------------------------------
        X["avg_events_per_active_day"] = np.where(
            X["ndays_act"] > 0,
            X["nevents"] / X["ndays_act"],
            0
        )

        X["avg_chapters_per_active_day"] = np.where(
            X["ndays_act"] > 0,
            X["nchapters"] / X["ndays_act"],
            0
        )

        # -----------------------------------
        # Drop raw columns no longer needed
        # -----------------------------------
        X.drop(
            columns=["YoB", "start_time_DI", "last_event_DI"],
            inplace=True,
            errors="ignore"
        )

        return X

# Creating a custom transformer to impute the missing values in the dataset.

class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_num_cols, missing_cat_cols):
        self.missing_num_cols = missing_num_cols
        self.missing_cat_cols = missing_cat_cols

    def fit(self, X, y=None):
        self.num_medians_ = X[self.missing_num_cols].median()
        self.cat_modes_ = X[self.missing_cat_cols].mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.missing_num_cols] = X[self.missing_num_cols].fillna(self.num_medians_)
        X[self.missing_cat_cols] = X[self.missing_cat_cols].fillna(self.cat_modes_)
        return X


# Creating a custom transformer called MappingEncoder to encode the gender and LoE_DI columns.
class MappingEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gender_map = {"m": 0, "f": 1, "o": 2}
        self.LoE_DI_map = {"Less than Secondary": 0, "Secondary": 1, "Bachelor's": 2, "Master's": 3, 
                            "Doctorate": 4}
    
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()
        X["gender"] = X["gender"].map(self.gender_map)
        X["LoE_DI"] = X["LoE_DI"].map(self.LoE_DI_map)
        return X
    
# Creating a custom transformer called FrequencyEncoder to do frequency encoding on the columns course_id
# and final_cc_cname_DI
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    
    def fit(self, X, y = None):
        self.freq_maps_ = {col: X[col].value_counts(normalize = True) for col in self.cols}
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0)
        return X
    
# Creating a custom transformer called OutlierHandler to handle the outliers:
class OutlierHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        def upper_iqr(col):
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            return q3 + 1.5 * (q3 - q1)
        
        def lower_iqr(col):
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            return q1 - 1.5 * (q3 - q1)
        self.caps_ = {
        # nevents
        "nevents_0.99": X["nevents"].quantile(0.99),
        "nevents_upper_iqr": upper_iqr("nevents"),

        # ndays_act
        "ndays_act_upper_iqr": upper_iqr("ndays_act"),

        # nchapters
        "nchapters_upper_iqr": upper_iqr("nchapters"),

        # course_duration_days
        "course_duration_days_upper_iqr": upper_iqr("course_duration_days"),

        # student_age_at_start
        "student_age_at_start_lower_iqr": lower_iqr("student_age_at_start"),
        "student_age_at_start_0.99": X["student_age_at_start"].quantile(0.99),

        # avg_events_per_active_day
        "avg_events_per_active_day_upper_iqr": upper_iqr("avg_events_per_active_day"),

        # avg_chapters_per_active_day
        "avg_chapters_per_active_day_0.99": X["avg_chapters_per_active_day"].quantile(0.99),
        "avg_chapters_per_active_day_upper_iqr": upper_iqr("avg_chapters_per_active_day"),

        # activity_ratio
        "activity_ratio_0.99": X["activity_ratio"].quantile(0.99),
        "activity_ratio_upper_iqr": upper_iqr("activity_ratio"),

        # nforum_posts
        "nforum_posts_0.995": X["nforum_posts"].quantile(0.995),
        "nforum_posts_upper_iqr": upper_iqr("nforum_posts"),

        # no_of_courses_registered
        "no_of_courses_registered_0.99": X["no_of_courses_registered"].quantile(0.99),
        "no_of_courses_registered_upper_iqr": upper_iqr("no_of_courses_registered"),

        # no_of_courses_explored
        "no_of_courses_explored_0.99": X["no_of_courses_explored"].quantile(0.99),
        "no_of_courses_explored_upper_iqr": upper_iqr("no_of_courses_explored"),
    }
        return self

    def transform(self, X):
        X = X.copy()
        
        # nevents
        X["nevents_outlier"] = (X["nevents"] > self.caps_["nevents_upper_iqr"]).astype(int)
        X["nevents"] = X["nevents"].clip(upper=self.caps_["nevents_0.99"])

        # ndays_act
        X["ndays_act_outlier"] = (X["ndays_act"] > self.caps_["ndays_act_upper_iqr"]).astype(int)

        # nchapters
        X["nchapters_outlier"] = (X["nchapters"] > self.caps_["nchapters_upper_iqr"]).astype(int)

        # course_duration_days 
        X["course_duration_days_outlier"] = (
                X["course_duration_days"] > self.caps_["course_duration_days_upper_iqr"]).astype(int)

        # student_age_at_start
        X["student_age_at_start"] = X["student_age_at_start"].clip(
            lower=self.caps_["student_age_at_start_lower_iqr"],
            upper=self.caps_["student_age_at_start_0.99"],)

        # avg_events_per_active_day
        X["avg_events_per_active_day"] = X["avg_events_per_active_day"].clip(
            upper=self.caps_["avg_events_per_active_day_upper_iqr"])

        # avg_chapters_per_active_day
        X["avg_chapters_per_active_day_outlier"] = (
            X["avg_chapters_per_active_day"] > self.caps_["avg_chapters_per_active_day_upper_iqr"]
            ).astype(int)
        X["avg_chapters_per_active_day"] = X["avg_chapters_per_active_day"].clip(
            upper=self.caps_["avg_chapters_per_active_day_0.99"])

        # activity_ratio
        X["activity_ratio_outlier"] = (
            X["activity_ratio"] > self.caps_["activity_ratio_upper_iqr"]).astype(int)
        X["activity_ratio"] = X["activity_ratio"].clip(upper=self.caps_["activity_ratio_0.99"])

        # nforum_posts
        X["nforum_posts_outlier"] = (
            X["nforum_posts"] > self.caps_["nforum_posts_upper_iqr"]).astype(int)
        X["nforum_posts"] = X["nforum_posts"].clip(upper=self.caps_["nforum_posts_0.995"])

        # no_of_courses_registered
        X["no_of_courses_registered_outlier"] = (
            X["no_of_courses_registered"] > self.caps_["no_of_courses_registered_upper_iqr"]).astype(int)
        X["no_of_courses_registered"] = X["no_of_courses_registered"].clip(
        upper=self.caps_["no_of_courses_registered_0.99"])

        # no_of_courses_explored
        X["no_of_courses_explored_outlier"] = (
            X["no_of_courses_explored"] > self.caps_["no_of_courses_explored_upper_iqr"]).astype(int)
        X["no_of_courses_explored"] = X["no_of_courses_explored"].clip(
            upper=self.caps_["no_of_courses_explored_0.99"])
        return X
    
# Creating a custom transformer called FeatureDropper to select the required features and drop irrelevant features
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_features):
        self.drop_features = drop_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.drop_features, errors="ignore")

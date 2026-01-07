import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """This function performs feature engineering on the input DataFrame.
       It creates new features based on existing ones and drops irrelevant features (like features
       with high missing values percentage, features with all unique values or a single unique value) 
       as necessary."""
    
    cols_to_drop = ["nplay_video", "roles", "incomplete_flag", "grade", "registered", "index"]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # We can find the number of courses each student has registered and out of those how many they have explored.
    # This can be done using the groupby function on the 'userid_DI' column. 
    df["no_of_courses_registered"] = df.groupby("userid_DI")["course_id"].transform("count") 
    df["no_of_courses_explored"] = df.groupby("userid_DI")["explored"].transform("sum")

    return df
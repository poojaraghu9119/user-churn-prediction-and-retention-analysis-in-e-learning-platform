import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit # Here, we use GroupShuffleSplit because the useid_DI column
# does not have all unique values, some ids are repeated because some students have registered to multiple
# courses. We want to make sure that all data for a given student is in either the training or testing set. 
# If we use train_test_split, we might end up with data for the same student in both sets, 
# which could lead to data leakage. So, we use GroupShuffleSplit here.
from src.data.loader import load_raw_data
from src.data_validation.validate_data import validate_data
from src.features.build_features import feature_engineering

def split_and_save_data():
    """This function is used to separate the features and target variable, split the data into training and 
       testing sets while ensuring that all data for a given student is in either the training or testing set,
       and save the resulting datasets to CSV files."""
    
    # Load and prepare the data
    df = load_raw_data("data/raw.csv")
    df = validate_data(df)
    df = feature_engineering(df)

    # The target variable is "certified", and we want to create groups based on "userid_DI".
    X = df.drop(columns= ["certified"])
    y = df["certified"]

    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups = df["userid_DI"]))

    # Creating the train and test sets
    X_train = X.iloc[train_idx].drop(columns=["userid_DI"])
    X_test = X.iloc[test_idx].drop(columns=["userid_DI"])
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    os.makedirs("data/processed", exist_ok=True)

    # Save the datasets to CSV files
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    split_and_save_data()
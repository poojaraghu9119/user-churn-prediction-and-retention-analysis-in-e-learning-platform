import pandas as pd
import numpy as np

def validate_data(df: pd.DataFrame) -> None:
    """This function checks for any impossible or inconsistent values in the DataFrame. Examples of inconsistent
       values include YoB being greater than or equal to the course start year, course start date greater than
       last event date, negative values in columns that should only have non-negative values, etc. 
       In such cases, this function deletes the inconsistent rows and logs the number of rows deleted for each check."""
    initial_row_count = df.shape[0]

    # Ensure datetime 
    df = df.copy()
    df["start_time_DI"] = pd.to_datetime(df["start_time_DI"], errors="coerce")
    df["last_event_DI"] = pd.to_datetime(df["last_event_DI"], errors="coerce")

    # Check for YoB greater than or equal to start year
    invalid_yob = df[df["YoB"] >= df["start_time_DI"].dt.year].shape[0]
    df.drop(df[df["YoB"] >= df["start_time_DI"].dt.year].index, inplace=True)

    # Check for start date greater than last event date
    invalid_dates = df[df["start_time_DI"] > df["last_event_DI"]].shape[0]
    df.drop(df[df["start_time_DI"] > df["last_event_DI"]].index, inplace=True)

    # Check for negative values in non-negative columns
    non_negative_columns = ["ndays_act", "nevents", "nchapters", "nplay_video"]
    for col in non_negative_columns:
        invalid_negatives = df[df[col] < 0].shape[0]
        df.drop(df[df[col] < 0].index, inplace=True)

    final_row_count = df.shape[0]
    total_rows_deleted = initial_row_count - final_row_count

    print(f"Rows deleted due to invalid YoB: {invalid_yob}")
    print(f"Rows deleted due to invalid dates: {invalid_dates}")
    print(f"Total rows deleted: {total_rows_deleted}")

    return df
    
    

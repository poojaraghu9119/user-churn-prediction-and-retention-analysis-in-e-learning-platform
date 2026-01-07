import pandas as pd
import numpy as np

def load_raw_data(file_path="data/raw.csv"):


    """This functions loads the raw data from the disk and removes empty strings in the grade column 
       (as explored in the EDA notebook) and
       converts its datatype to float."""
    

    df = pd.read_csv(file_path)
    df["grade"] = df["grade"].replace(" ", np.nan).astype(float)
    return df

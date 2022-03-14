from email.utils import decode_rfc2231
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

def load_drive_stats(filename, path) -> pd.DataFrame:
    """_summary_

    Args:
        filename (str, optional): _description_. Defaults to "ST4000DM000_history".

    Returns:
        pd.DataFrame: _description_
    """
    file = f"{path}/data/raw/{filename}.csv"
    df = pd.read_csv(file, parse_dates=["date"])
    return df

def countdown(df) -> pd.DataFrame:
    """create failure date column, calculate countdown
    """
    # Series of all the hdds the day they failed to obtain failure date
    failure = df[df.failure == 1]
    # Only use first failure per hdd
    #failure.sort_values('date', inplace=True)
    failure = failure.drop_duplicates(keep='first', subset="serial_number")
    # Assign failure dates
    df['date_failure'] = df['serial_number'].map(failure.set_index('serial_number')['date'])
    # Days to fail as int
    df["countdown"] = (df.date_failure - df.date).dt.days
    df = df[df.countdown >= 0]
    return df

def train_test_splitter(df, test_size, random_state, stratify=True):
    X = df.copy()
    y = X.pop("countdown")
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def drop_missing_cols(df, threshold=0.8) -> pd.DataFrame:
    cols_to_drop = df.columns[df.notna().sum() < (threshold * len(df))] # Columns that contain lot of NaNs
    #print("Number of columns to drop:", len(cols_to_drop))
    df = df.drop(cols_to_drop, axis=1) # Drop the cols
    #print("Shape of the dataframe", df.shape)
    return df

def drop_constant_cols(df) -> pd.DataFrame:
    # check columns which only contain 0 values and drop them from the data frame
    cols_to_drop = df.describe().T.query('std == 0').reset_index()['index'].to_list()
    #print(cols_to_drop)
    df = df.drop(cols_to_drop, axis=1)
    return df

def drop_missing_rows(df) -> pd.DataFrame:
    df = df.dropna(how="any")
    return df

def save_preprocessed_data(filename="ST4000DM000_history", path=os.getcwd()):
    df = load_drive_stats(filename, path)
    df = countdown(df)
    df = drop_missing_cols(df)
    df = drop_missing_rows(df)
    df = drop_constant_cols(df)
    file = f"{path}/data/processed/{filename}_preprocessed.csv"
    folder = f"{path}/data/processed/"
    if not os.path.exists(folder):
        os.mkdir(f"{os.getcwd()}/data/processed/")
    df.to_csv(file, index=False)
    return df

#save_preprocessed_data()
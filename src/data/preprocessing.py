from email.utils import decode_rfc2231
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

def load_drive_stats(filename, path) -> pd.DataFrame:
    """Load drive stats file

    Args:
        filename (_type_): Name of the csv file
        path (_type_): Path of the repo

    Returns:
        pd.DataFrame: Dataframe containing the drive stats
    """
    file = f"{path}/data/raw/{filename}.csv"
    df = pd.read_csv(file, parse_dates=["date"])
    return df

def countdown(df) -> pd.DataFrame:
    """Create column with failure date and calculate countdown

    Args:
        df (_type_): Drive stats file

    Returns:
        pd.DataFrame: Drive stats with countdown column
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

def train_test_splitter(df, test_size, random_state, stratify=True) -> pd.DataFrame:
    """Train test split of the drive data

    Args:
        df (_type_): Drive stats
        test_size (_type_): Size of the test set
        random_state (_type_): Random state for comparability over different runs
        stratify (bool, optional): Stratification. Defaults to True.

    Returns:
        pd.DataFrame: Split features and targets
    """
    X = df.copy()
    y = X.pop("countdown")
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def drop_missing_cols(df, threshold=0.8) -> pd.DataFrame:
    """Drop columns with missing values. A threshold allows to tune which columns are dropped.

    Args:
        df (_type_): Drive stats data
        threshold (float, optional): Percentage of missings in a column so that it is dropped. Defaults to 0.8.

    Returns:
        pd.DataFrame: Drive stats file with dropped columns
    """
    cols_to_drop = df.columns[df.notna().sum() < (threshold * len(df))] # Columns that contain lot of NaNs
    #print("Number of columns to drop:", len(cols_to_drop))
    df = df.drop(cols_to_drop, axis=1) # Drop the cols
    #print("Shape of the dataframe", df.shape)
    return df

def drop_constant_cols(df) -> pd.DataFrame:
    """Drop columns with constant values since they play no role for modeling

    Args:
        df (_type_): Drive stats data

    Returns:
        pd.DataFrame: Drive stats file with dropped columns
    """
    # check columns which only contain 0 values and drop them from the data frame
    cols_to_drop = df.describe().T.query('std == 0').reset_index()['index'].to_list()
    #print(cols_to_drop)
    df = df.drop(cols_to_drop, axis=1)
    return df

def drop_missing_rows(df) -> pd.DataFrame:
    """Drop rows with missing values (measurement errors, see EDA)

    Args:
        df (_type_): Drive stats

    Returns:
        pd.DataFrame: Drive stats data with removed rows
    """
    df = df.dropna(how="any")
    return df

def load_preprocess_data(filename="ST4000DM000_history_total", path=os.getcwd()) -> pd.DataFrame:
    """Load and preprocess drive stats data

    Args:
        filename (str, optional): Name of the csv file. Defaults to "ST4000DM000_history".
        path (_type_, optional): Path of the repo. Defaults to os.getcwd().

    Returns:
        pd.DataFrame: Dataframe with the drive stats data
    """
    df = load_drive_stats(filename, path)
    df = countdown(df)
    df = drop_missing_cols(df)
    df = drop_missing_rows(df)
    df = drop_constant_cols(df)
    return df

def save_preprocessed_data(filename="ST4000DM000_history_total", path=os.getcwd()):
    """Load and preprocess the drive stats data and store the result in a csv file

    Args:
        filename (str, optional): Name of the csv file. Defaults to "ST4000DM000_history".
        path (_type_, optional): Path of the repo. Defaults to os.getcwd().

    Returns:
        pd.DataFrame: Dataframe with the drive stats data
    """
    df = load_preprocess_data(filename=filename, path=path)
    file = f"{path}/data/processed/{filename}_preprocessed.csv"
    folder = f"{path}/data/processed/"
    if not os.path.exists(folder):
        os.mkdir(f"{os.getcwd()}/data/processed/")
    df.to_csv(file, index=False)
    return df

if __name__ == "__main__":
    _ = save_preprocessed_data()
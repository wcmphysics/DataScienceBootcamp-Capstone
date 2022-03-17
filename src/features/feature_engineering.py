import pandas as pd
import numpy as np
from src.data.preprocessing import drop_normalized_cols
from tqdm import tqdm

def ts_derivatives(df_in) -> pd.DataFrame:
    """Calculate the derivative of the features over time.

    Args:
        df_in (_type_): Dataframe with some features

    Returns:
        pd.DataFrame: Dataframe with derivative columns
    """
    # Dataframe with only the relevant data
    df = df_in.copy()
    df.drop(["model", "failure", "date_failure", "countdown"], axis=1, inplace=True)
    # Sort the values
    df.sort_values(["serial_number", "date"], inplace=True)
    # Drop date to supress warnings
    df.drop("date", inplace=True, axis=1)
    # Create grouped object
    grouped_data = df.groupby("serial_number")
    # Calculate derivative
    diff_series = grouped_data.diff()
    # Merge dataframes
    df_out = pd.merge(  left=df_in, right=diff_series, how="left", 
                        left_index=True, right_index=True, suffixes=("", "_diff"),
                        )
    return df_out

def ts_ema(df_in, interval=30) -> pd.DataFrame:
    """Calculate the EMA of the features over time.

    Args:
        df_in (_type_): Dataframe with some features

    Returns:
        pd.DataFrame: Dataframe with EMA columns
    """
    # Dataframe with only the relevant data
    df = df_in.copy()
    df.drop(["model", "failure", "date_failure", "countdown"], axis=1, inplace=True)
    # Sort the values
    df.sort_values(["serial_number", "date"], inplace=True)
    # Drop date to supress warnings
    df.drop("date", inplace=True, axis=1)
    # Create grouped object
    grouped_data = df.groupby("serial_number")
    # Calculate EMA
    int_series = grouped_data.ewm(span=interval).mean()
    # Extract index from multiindex
    index_series = int_series.index.get_level_values(1)
    # Fix indices
    int_series = pd.DataFrame(int_series.values, index=index_series, columns=int_series.columns)
    # Merge dataframes
    df_out = pd.merge(  left=df_in, right=int_series, how="left", 
                        left_index=True, right_index=True, suffixes=("", "_ema"),
                        )
    return df_out

def ts_rolling_d(df_in, interval=30) -> pd.DataFrame:
    """Calculate the rolling sum over derivative of the features over time.

    Args:
        df_in (_type_): Dataframe with some features

    Returns:
        pd.DataFrame: Dataframe with new columns
    """
    # Dataframe with only the relevant data
    df = df_in.copy()
    df.drop(["model", "failure", "date_failure", "countdown"], axis=1, inplace=True)
    # Sort the values
    df.sort_values(["serial_number", "date"], inplace=True)
    # Drop date to supress warnings
    df.drop("date", inplace=True, axis=1)
    # Create grouped object
    grouped_data = df.groupby("serial_number")
    # Calculate features
    diff_sum_series = grouped_data.diff().rolling(interval, min_periods=1).mean()
    # Merge dataframes
    df_out = pd.merge( left=df_in, right=diff_sum_series, 
                how="left", left_index=True, right_index=True, suffixes=("", "_rolling_d"),
                )
    return df_out

def unwrap_smart_7(df_in) -> pd.DataFrame:
    """Fix the jumps in the smart_7 feature

    Args:
        df_in (_type_): Drive stats data

    Returns:
        pd.DataFrame: Data with updated feature
    """
    # Copy input dataframe
    df = df_in.copy()
    df["smart_7_mod"] = df.smart_7_raw
    # Extract individual drives
    drives = df.serial_number.unique()
    for drive in tqdm(drives): # Loop over drives
        # Create dataframe with time series for drive, reindex and store the old index
        temp_data = df[df.serial_number == drive].sort_values("countdown", ascending=False).reset_index()
        # Calculate the derivate and use spikes to determine jumps
        jumps = temp_data.smart_7_raw.diff() < -5e8
        # Extract index of jumps
        jump_idx = jumps[jumps].index
        # Backup the smart_7_raw series
        smart_7_temp = temp_data.smart_7_raw.copy()
        for idx in jump_idx: # Loop over the jumps
            # Add the value before the jump to all the following  values
            temp_data.loc[idx:, "smart_7_raw"] += smart_7_temp[idx-1]
        # Restore the original index
        temp_data.set_index("index", inplace=True)
        # Update the dataframe with the unwrapped data for this drive
        df.smart_7_mod[temp_data.index] = temp_data.smart_7_raw
    return df

def calculate_smart_999(df_in, trigger_percentage=0.05) -> pd.DataFrame:
    """Calculate the smart_999 feature. If the raw differs from the EMA by more 
    than trigger_percent, the corresponding feature initiates a trigger. Smart_999
    sums over all those triggers.

    Args:
        df_in (_type_): Drive stats data
        trigger_percentage (float, optional): Percentage for triggering. Defaults to 0.05.

    Returns:
        pd.DataFrame: Dataframe with features
    """
    df = df_in.copy()
    # Select columns to use for the calculation
    cols_to_use =   ['smart_4_raw', 'smart_5_raw',# 'smart_7_raw',
                    'smart_12_raw', 'smart_183_raw', 'smart_184_raw',
                    'smart_187_raw', 'smart_188_raw', 'smart_189_raw',
                    'smart_193_raw', 'smart_192_raw', 'smart_197_raw',
                    'smart_198_raw', 'smart_199_raw',
                    ]
    # Loop over columns
    for col in cols_to_use:
        # Check if raw differs from ema by more than 5%
        df[col+"_trigger"] = 1/2 * np.abs((df[col] + df[col+"_ema"]) / df[col+"_ema"]) > (1+trigger_percentage)
    #print("Shape after calculation of EMA triggers:", df.shape)
    # Sum over all triggers
    sum_cols = []
    for col in df.columns:
        if 'trigger' in col:
            sum_cols.append(col)
    df["smart_999"] = df[sum_cols].sum(axis=1)
    #print("Shape after calculation of sum of EMA triggers:", df.shape)
    return df

def create_features(df_in, interval=30, trigger_percentage=0.05) -> pd.DataFrame:
    """Create the fancy features.

    Args:
        df_in (_type_): Dataframe as output by preprocessing script
        interval (int, optional): Time interval for EMA. Defaults to 30.
        trigger_percentage (float, optional): Normalized distance between raw and EMA. Defaults to 0.05.

    Returns:
        pd.DataFrame: Dataset with new features
    """
    df = df_in.copy()
    df = drop_normalized_cols(df)
    df = unwrap_smart_7(df)
    df = ts_ema(df, interval=interval)
    df = calculate_smart_999(df, trigger_percentage=trigger_percentage)
    #df.drop(["model", "failure", "date_failure"], axis=1, inplace=True)
    return df

if __name__ == "__main__":
    _ = create_features()
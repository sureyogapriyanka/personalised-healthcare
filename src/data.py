import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path
from typing import Optional

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, impute missing values, and add patient_id if missing."""
    # standardize column names
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    # numeric vs categorical
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    # impute numeric with median
    if len(num_cols) > 0:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    # impute categorical with "Unknown"
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    # add patient_id if missing
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", range(1, len(df) + 1))

    return df

def load_and_preprocess_data(raw_path: str, processed_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw patient data and perform initial preprocessing.
    
    Parameters:
    raw_path (str): Path to the raw CSV file
    processed_dir (str, optional): Directory to save processed data
    
    Returns:
    pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    # Load data
    df = load_data(raw_path)
    
    # Clean data
    df = clean_data(df)
    
    # Save processed data if directory provided
    if processed_dir:
        processed_path = Path(processed_dir) / 'patient_data_processed.csv'
        df.to_csv(processed_path, index=False)
        print(f"Processed data saved to: {processed_path}")
    
    return df
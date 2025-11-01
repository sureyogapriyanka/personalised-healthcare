import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add an age_group column if 'age' exists."""
    if "age" in df.columns:
        bins = [0, 30, 45, 60, 100]
        labels = ["Young Adult", "Middle Adult", "Older Adult", "Senior"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
    return df

def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk_score based on multiple health factors."""
    risk_factors = ["age", "recovery_time"]
    available_factors = [col for col in risk_factors if col in df.columns]
    
    if available_factors:
        df["risk_score"] = df[available_factors].sum(axis=1)
    else:
        df["risk_score"] = df["age"] if "age" in df.columns else 0
    
    return df

def create_treatment_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Create treatment_effectiveness indicator from treatment_response."""
    if "treatment_response" in df.columns:
        treatment_mapping = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}
        df["treatment_effectiveness"] = df["treatment_response"].map(treatment_mapping)
    return df

def add_chronic_condition_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add chronic_condition indicator based on medical history."""
    if "medical_history" in df.columns:
        chronic_conditions = ["diabetes", "hypertension", "heart disease", "asthma", "copd"]
        df["has_chronic_condition"] = df["medical_history"].str.lower().apply(
            lambda x: any(condition in str(x) for condition in chronic_conditions)
        ).astype(int)
    return df

def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables for machine learning."""
    # Create a copy to avoid modifying the original dataframe
    df_encoded = df.copy()
    
    # Get categorical columns
    cat_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Label encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        if col != 'patient_id':  # Don't encode patient_id
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    return df_encoded

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive feature engineering for patient healthcare data.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with patient healthcare data
    
    Returns:
    pd.DataFrame: DataFrame with engineered features
    """
    # Add age groups
    df = add_age_group(df)
    
    # Add risk scores
    df = add_risk_score(df)
    
    # Create treatment effectiveness indicators
    df = create_treatment_effectiveness(df)
    
    # Add chronic condition indicators
    df = add_chronic_condition_indicator(df)
    
    # Convert text columns to numeric features where possible
    text_cols = ["vital_signs", "laboratory_results"]
    for col in text_cols:
        if col in df.columns:
            # Extract numeric values from text columns
            # This is a simplified example - in practice, you'd want more sophisticated parsing
            df[f"{col}_parsed"] = df[col].str.len()  # Simple feature as example
    
    return df
import pandas as pd
import joblib
from typing import Dict, Any

def serve(model_path: str, input_dict: Dict[str, Any]) -> Any:  # type: ignore
    """
    Load a trained model and make a prediction for one patient sample.
    input_dict should be a dict with feature names as keys.
    
    Parameters:
    model_path (str): Path to the trained model
    input_dict (dict): Dictionary with feature names as keys and values
    
    Returns:
    Prediction result
    """
    # Load the trained model
    model = joblib.load(model_path)
    
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    return prediction

def serve_proba(model_path: str, input_dict: Dict[str, Any]) -> Any:  # type: ignore
    """
    Load a trained model and return prediction probabilities for one patient sample.
    
    Parameters:
    model_path (str): Path to the trained model
    input_dict (dict): Dictionary with feature names as keys and values
    
    Returns:
    Prediction probabilities
    """
    # Load the trained model
    model = joblib.load(model_path)
    
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Get prediction probabilities
    probabilities = model.predict_proba(df)[0]
    
    return probabilities
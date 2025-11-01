from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
from typing import Tuple, Any, Dict

def build_models():
    """
    Returns a dictionary of machine learning models suitable for healthcare predictions.
    """
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "logistic_regression": LogisticRegression(
            random_state=42, max_iter=1000
        )
    }
    return models

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Any, Any]:
    """
    Train the best performing machine learning model on the provided data.
    
    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target variable
    
    Returns:
    Tuple[Any, Any, Any]: Trained model, X_test, y_test
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and train the best model (Random Forest based on our healthcare analysis)
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model.
    
    Parameters:
    model (Any): Trained model
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test targets
    
    Returns:
    dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "y_test": y_test,
        "y_pred": y_pred
    }

def compare_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
    """
    Compare multiple models and return their performance metrics.
    
    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target variable
    
    Returns:
    Dict[str, Dict]: Dictionary with model names and their performance metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get models
    models = build_models()
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        # Use scaled data for Logistic Regression
        if name == "logistic_regression":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": y_pred
        }
    
    return results

def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Parameters:
    model (Any): Trained model to save
    filepath (str): Path where to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Parameters:
    filepath (str): Path to the saved model
    
    Returns:
    Any: Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model
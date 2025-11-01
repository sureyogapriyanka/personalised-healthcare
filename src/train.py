import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Now we can import from src modules
from models import build_models, train_model, evaluate_model, save_model, compare_models
from data import load_and_preprocess_data
from features import engineer_features
import joblib

def train(path: str, target: str = "treatment_response"):
    """
    Train machine learning models on the patient healthcare dataset located at `path`.
    
    Parameters:
    path (str): Path to the dataset
    target (str): Target column for prediction
    
    Returns:
    Trained model
    """
    # Load and preprocess data
    df = load_and_preprocess_data(path)
    
    # Engineer features
    df = engineer_features(df)
    
    # Check if target column exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")

    # Prepare features and target
    X = df.drop(columns=[target, "patient_id"])
    y = df[target]

    # Handle any remaining missing values
    X = X.fillna(X.median())

    # Compare multiple models
    print("Comparing multiple models...")
    model_results = compare_models(X, y)
    
    # Display model comparison
    print("\nModel Comparison:")
    print("=" * 30)
    for name, results in model_results.items():
        print(f"{name}: {results['accuracy']:.4f}")
    
    # Find best model
    best_model_name = max(model_results, key=lambda x: model_results[x]['accuracy'])
    best_model = model_results[best_model_name]['model']
    best_predictions = model_results[best_model_name]['predictions']
    
    # Get test data for detailed evaluation
    # Split the data to get test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nBest Model: {best_model_name} with accuracy {model_results[best_model_name]['accuracy']:.4f}")
    
    # Detailed evaluation of best model
    print(f"\nDetailed Evaluation of {best_model_name}:")
    print("=" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, best_predictions))
    
    # Save the best model
    model_path = "../models/healthcare_predictor.pkl"
    save_model(best_model, model_path)
    
    return best_model

if __name__ == "__main__":
    train("../data/raw/patient_data.csv", "treatment_response")
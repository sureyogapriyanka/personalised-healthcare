#!/usr/bin/env python3
"""
Main execution script for the Personalized Healthcare Analysis Project.

This script runs the complete analysis pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Results reporting
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import functions from our modules
import pandas as pd

# Import modules with relative imports to avoid circular imports
import data
import features
import models

def main():
    """Execute the complete personalized healthcare analysis pipeline."""
    print("=== Personalized Healthcare Analysis Pipeline ===")
    print("Starting analysis pipeline...\n")
    
    # Define paths
    project_root = src_path.parent
    raw_data_path = project_root / "data" / "raw" / "patient_data.csv"
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    
    # Create directories if they don't exist
    processed_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    if not raw_data_path.exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please ensure the patient dataset is located at data/raw/patient_data.csv")
        return 1
    
    df = data.load_and_preprocess_data(str(raw_data_path), str(processed_dir))
    print(f"Loaded {len(df)} patients with {len(df.columns)} features\n")
    
    # Step 2: Feature engineering
    print("Step 2: Engineering features...")
    df = features.engineer_features(df)
    print(f"Feature engineering complete. Dataset now has {len(df.columns)} features\n")
    
    # Step 3: Model training
    print("Step 3: Training model...")
    target = "treatment_response"
    
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found in dataset")
        return 1
    
    # Prepare features and target
    X = df.drop(columns=[target, "patient_id"])
    y = df[target]
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    # Train the model
    model, X_test, y_test = models.train_model(X, y)
    print("Model training complete\n")
    
    # Step 4: Model evaluation
    print("Step 4: Evaluating model...")
    results = models.evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {results['accuracy']:.4f}\n")
    
    # Print detailed classification report
    print("Classification Report:")
    print("-" * 50)
    for label, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"{label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            print()
    
    # Step 5: Save model
    print("Step 5: Saving model...")
    model_path = models_dir / "healthcare_predictor.pkl"
    models.save_model(model, str(model_path))
    
    # Save processed data
    processed_path = processed_dir / "patient_data_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to: {processed_path}\n")
    
    # Step 6: Generate summary report
    print("Step 6: Generating summary report...")
    summary_stats = {
        'metric': [
            'Total Patients',
            'Age Range',
            'Mean Age',
            'Male Percentage',
            'Female Percentage',
            'Average Recovery Time',
            'Treatment Response Rate',
            'High Readmission Risk Percentage',
            'Most Common Condition',
            'Model Accuracy'
        ],
        'value': [
            len(df),
            f"{df['age'].min()}-{df['age'].max()}",
            f"{df['age'].mean():.1f}",
            f"{(df['gender'] == 'Male').mean() * 100:.1f}%",
            f"{(df['gender'] == 'Female').mean() * 100:.1f}%",
            f"{df['recovery_time'].mean():.1f}",
            f"{(df['treatment_response'] == 'Good').mean() * 100:.1f}%",
            f"{(df['readmission_risk'] == 'High').mean() * 100:.1f}%",
            df['medical_history'].mode()[0] if not df['medical_history'].mode().empty else 'N/A',
            f"{results['accuracy']:.4f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = reports_dir / "healthcare_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary report saved to: {summary_path}\n")
    
    print("=== Analysis Pipeline Complete ===")
    print("Next steps:")
    print("1. Review the generated notebooks in the notebooks/ directory")
    print("2. Check the processed data in data/processed/")
    print("3. Examine the trained model in models/")
    print("4. View the summary report in reports/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
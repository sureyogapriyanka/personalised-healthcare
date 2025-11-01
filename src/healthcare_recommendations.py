#!/usr/bin/env python3
"""
Personalized Healthcare Recommendations System

This script provides a command-line interface for generating personalized 
healthcare recommendations based on patient data using a trained machine learning model.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import modules with relative imports
import data
import features
import models

def load_model(model_path="../models/healthcare_recommendation_model.pkl"):
    """
    Load the trained healthcare recommendation model.
    
    Parameters:
    model_path (str): Path to the saved model
    
    Returns:
    Trained model pipeline
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running the notebook.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def extract_vital_signs(vital_signs_str):
    """
    Extract systolic BP, diastolic BP, and heart rate from vital signs string.
    
    Parameters:
    vital_signs_str (str): Vital signs string in format "BP: 130/85, HR: 72"
    
    Returns:
    tuple: (systolic_bp, diastolic_bp, heart_rate)
    """
    try:
        # Example: "BP: 130/85, HR: 72"
        bp_part = vital_signs_str.split('BP: ')[1].split(',')[0]
        systolic, diastolic = map(int, bp_part.split('/'))
        heart_rate = int(vital_signs_str.split('HR: ')[1])
        return systolic, diastolic, heart_rate
    except:
        return np.nan, np.nan, np.nan

def generate_recommendation(patient_data, model, feature_columns):
    """
    Generate personalized healthcare recommendation for a patient.
    
    Parameters:
    patient_data (dict): Dictionary containing patient information
    model: Trained model pipeline
    feature_columns (list): List of feature column names used in training
    
    Returns:
    dict: Recommendation results
    """
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Apply the same preprocessing as training data
    patient_df.columns = patient_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Extract vital signs if present
    if 'vital_signs' in patient_df.columns:
        vital_data = patient_df['vital_signs'].apply(extract_vital_signs)
        patient_df['systolic_bp'] = [x[0] for x in vital_data]
        patient_df['diastolic_bp'] = [x[1] for x in vital_data]
        patient_df['heart_rate'] = [x[2] for x in vital_data]
    
    # Create age group
    if 'age' in patient_df.columns:
        patient_df['age_group'] = pd.cut(patient_df['age'], 
                                        bins=[0, 30, 45, 60, 100], 
                                        labels=['Young Adult', 'Middle Adult', 'Older Adult', 'Senior'])
    
    # Create chronic condition indicator
    chronic_conditions = ['diabetes', 'hypertension', 'heart disease', 'asthma', 'copd', 'kidney']
    if 'medical_history' in patient_df.columns:
        patient_df['has_chronic_condition'] = patient_df['medical_history'].str.lower().apply(
            lambda x: any(condition in str(x) for condition in chronic_conditions)
        ).astype(int)
    
    # Create family history indicator
    if 'family_history' in patient_df.columns:
        patient_df['family_history_indicator'] = (patient_df['family_history'] == 'Yes').astype(int)
    
    # Select the same features used in training
    patient_features = patient_df[feature_columns]
    
    # Make prediction
    prediction = model.predict(patient_features)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(patient_features)[0]
    confidence = np.max(probabilities)
    
    # Map predictions to recommendations
    recommendation_mapping = {
        'Excellent': 'Continue current treatment plan. Patient is responding very well.',
        'Good': 'Current treatment is effective. Consider minor adjustments if needed.',
        'Fair': 'Treatment response is moderate. Consider alternative approaches or additional interventions.',
        'Poor': 'Treatment response is suboptimal. Recommend immediate consultation and treatment plan revision.'
    }
    
    recommendation = recommendation_mapping.get(prediction, 'No specific recommendation available.')
    
    return {
        'predicted_response': prediction,
        'recommendation': recommendation,
        'confidence': confidence
    }

def main():
    """Main function to run the healthcare recommendation system."""
    print("=== Personalized Healthcare Recommendations System ===\n")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate personalized healthcare recommendations")
    parser.add_argument("--age", type=int, help="Patient age")
    parser.add_argument("--gender", choices=["Male", "Female", "Other"], help="Patient gender")
    parser.add_argument("--ethnicity", help="Patient ethnicity")
    parser.add_argument("--medical_history", help="Patient medical history")
    parser.add_argument("--family_history", choices=["Yes", "No"], help="Family medical history")
    parser.add_argument("--vital_signs", help="Vital signs (format: 'BP: 120/80, HR: 70')")
    parser.add_argument("--recovery_time", type=int, help="Recovery time in days")
    parser.add_argument("--model_path", default="../models/healthcare_recommendation_model.pkl", 
                        help="Path to the trained model")
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample usage:")
        print("python healthcare_recommendations.py --age 45 --gender Male --ethnicity Caucasian \\")
        print("  --medical_history \"Hypertension, Diabetes\" --family_history Yes \\")
        print("  --vital_signs \"BP: 130/85, HR: 72\" --recovery_time 14")
        return
    
    args = parser.parse_args()
    
    # Validate required arguments
    required_args = [args.age, args.gender, args.ethnicity, args.medical_history, 
                     args.family_history, args.vital_signs, args.recovery_time]
    
    if any(arg is None for arg in required_args):
        print("Error: All patient information fields are required.")
        parser.print_help()
        sys.exit(1)
    
    # Create patient data dictionary
    patient_data = {
        'age': args.age,
        'gender': args.gender,
        'ethnicity': args.ethnicity,
        'medical_history': args.medical_history,
        'family_history': args.family_history,
        'vital_signs': args.vital_signs,
        'recovery_time': args.recovery_time
    }
    
    # Load the trained model
    model = load_model(args.model_path)
    
    # Load feature information
    try:
        feature_info_path = Path(args.model_path).parent / "feature_info.pkl"
        feature_info = joblib.load(feature_info_path)
        feature_columns = feature_info['feature_columns']
    except FileNotFoundError:
        print("Error: Feature information file not found.")
        print("Please retrain the model to generate feature information.")
        sys.exit(1)
    
    # Generate recommendation
    print("Generating personalized healthcare recommendation...")
    result = generate_recommendation(patient_data, model, feature_columns)
    
    # Display results
    print("\n" + "="*50)
    print("PERSONALIZED HEALTHCARE RECOMMENDATION")
    print("="*50)
    print(f"Patient Age: {args.age}")
    print(f"Gender: {args.gender}")
    print(f"Ethnicity: {args.ethnicity}")
    print(f"Medical History: {args.medical_history}")
    print(f"Family History: {args.family_history}")
    print(f"Vital Signs: {args.vital_signs}")
    print(f"Recovery Time: {args.recovery_time} days")
    
    print("\n" + "-"*50)
    print(f"Predicted Treatment Response: {result['predicted_response']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Recommendation: {result['recommendation']}")
    print("-"*50)

if __name__ == "__main__":
    main()
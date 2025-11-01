"""
Unit tests for the data module in the personalized healthcare project.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data import load_data, clean_data, load_and_preprocess_data

def test_load_data():
    """Test that load_data function works correctly."""
    # Create a simple test CSV
    test_data = """Name,Age,Gender
John,25,Male
Jane,30,Female
Bob,35,Male"""
    
    # Write to temporary file
    test_file = "test_data.csv"
    with open(test_file, "w") as f:
        f.write(test_data)
    
    # Load data
    df = load_data(test_file)
    
    # Check that data was loaded correctly
    assert len(df) == 3
    assert list(df.columns) == ["Name", "Age", "Gender"]
    assert df["Age"].dtype == "int64"
    
    # Clean up
    os.remove(test_file)

def test_clean_data():
    """Test that clean_data function works correctly."""
    # Create test DataFrame
    df = pd.DataFrame({
        "Name": ["John", "Jane", "Bob"],
        "Age": [25, 30, 35],
        "Gender": ["Male", "Female", "Male"]
    })
    
    # Clean data
    cleaned_df = clean_data(df)
    
    # Check that column names were standardized (and patient_id was added at the beginning)
    expected_columns = ["patient_id", "name", "age", "gender"]
    assert list(cleaned_df.columns) == expected_columns
    
    # Check that patient_id was added and has correct values
    assert list(cleaned_df["patient_id"]) == [1, 2, 3]

def test_load_and_preprocess_data():
    """Test that load_and_preprocess_data function works correctly."""
    # Create a simple test CSV
    test_data = """Name,Age,Gender
John,25,Male
Jane,30,Female
Bob,35,Male"""
    
    # Write to temporary file
    test_file = "test_data.csv"
    with open(test_file, "w") as f:
        f.write(test_data)
    
    # Load and preprocess data
    df = load_and_preprocess_data(test_file)
    
    # Check that data was loaded and cleaned correctly
    assert len(df) == 3
    expected_columns = ["patient_id", "name", "age", "gender"]
    assert list(df.columns) == expected_columns
    
    # Clean up
    os.remove(test_file)

if __name__ == "__main__":
    pytest.main([__file__])
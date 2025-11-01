#!/usr/bin/env python3
"""
Virtual Environment Setup Script for Personalized Healthcare Project

This script creates and sets up a virtual environment with all required dependencies.
"""

import subprocess
import sys
import os
import venv

def create_virtual_environment(env_name="healthcare_env"):
    """Create a virtual environment"""
    try:
        print(f"Creating virtual environment '{env_name}'...")
        venv.create(env_name, with_pip=True)
        print(f"Virtual environment '{env_name}' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        return False

def install_requirements_in_venv(env_name="healthcare_env"):
    """Install requirements in the virtual environment"""
    try:
        # Determine the path to the virtual environment's Python executable
        if os.name == 'nt':  # Windows
            python_exe = os.path.join(env_name, "Scripts", "python.exe")
        else:  # Unix/Linux/Mac
            python_exe = os.path.join(env_name, "bin", "python")
        
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("Installing project dependencies...")
        subprocess.check_call([python_exe, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    except FileNotFoundError:
        print("Error: Could not find virtual environment Python executable.")
        return False

def main():
    env_name = "healthcare_env"
    
    print("=== Personalized Healthcare Project Setup ===")
    print("This script will create a virtual environment and install all dependencies.\n")
    
    # Create virtual environment
    if not create_virtual_environment(env_name):
        print("Failed to create virtual environment. Exiting.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements_in_venv(env_name):
        print("Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Provide activation instructions
    print("\n" + "="*50)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("To activate the virtual environment, run:")
    if os.name == 'nt':  # Windows
        print(f"   {env_name}\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print(f"   source {env_name}/bin/activate")
    print("\nOnce activated, you can run:")
    print("   jupyter notebook  # To start Jupyter")
    print("   python src/main.py  # To run the main script")
    print("\nTo deactivate the virtual environment, simply run:")
    print("   deactivate")

if __name__ == "__main__":
    main()
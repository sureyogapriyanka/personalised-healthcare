#!/usr/bin/env python3
"""
Setup script for the Personalized Healthcare project.
This script helps install all required dependencies.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages from requirements.txt"""
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("Error: requirements.txt not found!")
            return False
            
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All required packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False
    except FileNotFoundError:
        print("Error: pip not found. Please ensure Python is properly installed.")
        return False

def main():
    print("Setting up Personalized Healthcare project dependencies...")
    print("This may take a few minutes...")
    
    if install_requirements():
        print("\nSetup completed successfully!")
        print("You can now run the notebooks and scripts in this project.")
    else:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple launcher script for the Blog Maker application.
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Blog Maker application."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Launch the application
    print("Starting Blog Maker...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error launching application: {e}")

if __name__ == "__main__":
    main()

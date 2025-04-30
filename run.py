"""
Run script for the LLM Text Extraction System.
"""
import subprocess
import os
import sys
import webbrowser
from time import sleep
from dotenv import load_dotenv

def main():
    """Run the Streamlit application."""
    print("Starting LLM Text Extraction System...")
    
    # Load environment variables from .env file if it exists
    if os.path.exists(".env"):
        print("Loading environment variables from .env file")
        load_dotenv()
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed.")
    except ImportError:
        print("Streamlit is not installed. Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create data directories if they don't exist
    for directory in ["data/documents", "data/results", "data/prompts", "data/feedback"]:
        os.makedirs(directory, exist_ok=True)
    
    # Start the Streamlit application
    print("Launching application...")
    
    # Open browser after a short delay to ensure Streamlit server is running
    def open_browser():
        sleep(2)  # Wait for Streamlit to start
        webbrowser.open_new("http://localhost:8501")
    
    # Start browser in a separate thread
    import threading
    threading.Thread(target=open_browser).start()
    
    # Run Streamlit
    subprocess.call(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main() 
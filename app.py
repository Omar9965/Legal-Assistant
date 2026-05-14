"""
App Entry Point — Launches the Streamlit Legal AI chat interface.

Usage: python app.py
"""

import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "views", "streamlit_app.py")
    
    print("Starting Legal AI Assistant...")
    print(f"Streamlit app: {app_path}")
    print("Press Ctrl+C to stop.\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            app_path,
            "--server.headless", "true",
            "--theme.base", "dark"
        ], check=True)
    except KeyboardInterrupt:
        print("\nShutting down Legal AI Assistant.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Streamlit process exited with code {e.returncode}")

if __name__ == "__main__":
    main()

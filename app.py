"""
App Entry Point — Launches the Streamlit Legal AI chat interface.

Usage: python app.py
"""

import os
import sys
import subprocess
from controllers.graph import get_graph


def main():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views", "streamlit_app.py")
    
    print("⚖️  Starting Legal AI Assistant...")
    print(f"   Streamlit app: {app_path}")
    print("   Press Ctrl+C to stop.\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.headless", "true",
        "--theme.base", "dark",
    ])


if __name__ == "__main__":
    main()

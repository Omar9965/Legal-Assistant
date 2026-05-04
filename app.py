"""
App Entry Point — Launches the Streamlit Legal AI chat interface.

Usage: python app.py
"""

import os
import sys
import subprocess
import time
from controllers.graph import get_graph


def preload_dependencies():
    """Pre-compile graph and load models before Streamlit starts."""
    print("⏳ Loading AI models and compiling graph...")

    start = time.time()

    from utils.config import get_embedding_function, get_llm
    get_embedding_function()
    get_llm()

    get_graph()

    elapsed = time.time() - start
    print(f"✓ Dependencies loaded in {elapsed:.1f}s\n")


def main():
    preload_dependencies()

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

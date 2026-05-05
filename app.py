"""
App Entry Point — Launches the Streamlit Legal AI chat interface.

Usage: python app.py
"""

import os
import sys
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class StreamlitConfig:
    """Configuration for the Streamlit application."""
    app_path: str
    headless: bool = True
    theme_base: str = "dark"


class ApplicationRunner(ABC):
    """
    Abstract base class defining the contract for running an application.
    Follows Dependency Inversion Principle (DIP) and Open/Closed Principle (OCP).
    """
    @abstractmethod
    def run(self) -> None:
        pass


class StreamlitRunner(ApplicationRunner):
    """
    Responsible solely for launching and managing a Streamlit process.
    Follows Single Responsibility Principle (SRP).
    """
    def __init__(self, config: StreamlitConfig):
        self.config = config

    def _build_command(self) -> List[str]:
        return [
            sys.executable, "-m", "streamlit", "run",
            self.config.app_path,
            "--server.headless", str(self.config.headless).lower(),
            "--theme.base", self.config.theme_base,
        ]

    def run(self) -> None:
        print("⚖️  Starting Legal AI Assistant...")
        print(f"   Streamlit app: {self.config.app_path}")
        print("   Press Ctrl+C to stop.\n")
        
        try:
            subprocess.run(self._build_command(), check=True)
        except KeyboardInterrupt:
            print("\nShutting down Legal AI Assistant.")
        except subprocess.CalledProcessError as e:
            print(f"\nError: Streamlit process exited with code {e.returncode}")


class LegalAIAppFacade:
    """
    Facade pattern to simplify the initialization and execution of the application.
    """
    def __init__(self, runner: ApplicationRunner):
        self._runner = runner
        
    def start(self) -> None:
        # Additional startup orchestration could go here
        self._runner.run()


def main():
    # Construct paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "views", "streamlit_app.py")
    
    # Initialize dependencies
    config = StreamlitConfig(app_path=app_path)
    runner = StreamlitRunner(config)
    
    # Run application via Facade
    app = LegalAIAppFacade(runner)
    app.start()


if __name__ == "__main__":
    main()

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Legal AI workflow.
    Enforces a standard executable interface for each controller.
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the agent's primary responsibility.
        Should be implemented by all inheriting agents.
        """
        pass

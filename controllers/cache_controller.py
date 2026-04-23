"""
Cache Controller — Handles semantic cache lookup and hit/miss decision.
"""

from models.semantic_cache import lookup
from controllers.base_agent import BaseAgent


class CacheAgent(BaseAgent):
    """
    Agent responsible for semantic cache lookup.
    """
    
    def execute(self, query: str) -> dict:
        """
        Check the semantic cache for a previously answered similar query.
        
        Args:
            query: The user's input query.
        
        Returns:
            Dict with keys:
                - cache_hit (bool): Whether a sufficiently similar cached query was found.
                - cached_answer (str | None): The cached answer if hit.
                - similarity_score (float): The similarity score of the best match.
        """
        cached_answer, score = lookup(query)

        if cached_answer is not None:
            return {
                "cache_hit": True,
                "cached_answer": cached_answer,
                "similarity_score": score,
            }
        
        return {
            "cache_hit": False,
            "cached_answer": None,
            "similarity_score": score,
        }

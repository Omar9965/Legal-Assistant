"""
Cache Controller — Handles multi-level semantic cache lookup and hit/miss decision.

Levels:
1. Exact Query Cache (in-memory, O(1))
2. Semantic Cache (ChromaDB vector similarity)
3. Article Lookup Cache (in-memory)
"""

from controllers.base_agent import BaseAgent
from models.multi_level_cache import cache_lookup


class CacheAgent(BaseAgent):
    """
    Agent responsible for multi-level semantic cache lookup.
    """
    
    def execute(self, query: str) -> dict:
        """
        Check the multi-level cache for a previously answered similar query.
        
        Args:
            query: The user's input query.
        
        Returns:
            Dict with keys:
                - cache_hit (bool): Whether a cached query was found.
                - cached_answer (str | None): The cached answer if hit.
                - similarity_score (float): The similarity score of the best match.
                - cache_level (str): Which cache level hit (exact, semantic, article, none).
        """
        cached_answer, score, cache_level = cache_lookup(query)

        if cached_answer is not None:
            return {
                "cache_hit": True,
                "cached_answer": cached_answer,
                "similarity_score": score,
                "cache_level": cache_level,
            }
        
        return {
            "cache_hit": False,
            "cached_answer": None,
            "similarity_score": score,
            "cache_level": cache_level,
        }

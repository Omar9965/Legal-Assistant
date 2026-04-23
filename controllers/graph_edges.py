from controllers.graph_state import AgentState

def is_legal_check(state: AgentState) -> str:
    """Route based on whether the query is legal."""
    return "legal" if state.get("is_legal", True) else "not_legal"

def cache_hit_check(state: AgentState) -> str:
    """Route based on cache hit/miss."""
    return "cache_hit" if state.get("cache_hit", False) else "cache_miss"

def retrieval_quality_check(state: AgentState) -> str:
    """Check if retrieval quality needs retry."""
    quality = state.get("last_retrieval_quality", "success")
    attempts = state.get("retrieval_attempts", 1)
    
    # Retry if insufficient and hasn't exceeded max attempts
    if quality == "insufficient" and attempts < 2:
        return "retry"
    elif quality == "empty" and attempts < 2:
        return "retry"
    return "proceed"

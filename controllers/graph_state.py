from typing import TypedDict, Optional

class AgentState(TypedDict):
    """State object that flows through the graph."""
    # Input
    query: str
    conversation_history: list  # [{"role": ..., "content": ...}]
    
    # Router output
    language: str           # "ar" or "en"
    is_legal: bool
    router_confidence: float  # 0.0-1.0 confidence score
    router_reason: str
    
    # Cache output
    cache_hit: bool
    cached_answer: Optional[str]
    similarity_score: float
    
    # Researcher output & iteration tracking
    retrieved_docs: list
    search_metadata: dict
    retrieval_attempts: int  # Number of retrieval attempts (max 2)
    last_retrieval_quality: str  # "success", "insufficient", "empty"
    
    # Feedback loop
    reformulated_query: Optional[str]  # Query reformulated for retry
    
    # Final output
    response: str
    
    # Debug trace
    debug_trace: list       # List of step descriptions

"""
Graph — LangGraph StateGraph orchestrating the full Legal AI workflow.

Workflow:  START → Router → Cache Check → (hit?) → Researcher → Scribe → END
                                ↓ (not legal)        ↓ (cache hit)
                              Decline             Return Cached

No LangGraph checkpointer: conversation memory is owned by the UI, which
passes conversation_history with every request. The full-state re-init per
invocation made the checkpointer write-only dead weight.
"""

from langgraph.graph import StateGraph, END

from controllers.graph_state import AgentState
from controllers.graph_nodes import (
    router_node, decline_node, cache_node,
    researcher_node, reformulate_node, scribe_node
)
from controllers.graph_edges import (
    is_legal_check, cache_hit_check, retrieval_quality_check
)
from models import cache


# ── Build the Graph ──────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph StateGraph for the Legal AI workflow.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("decline", decline_node)
    workflow.add_node("cache_check", cache_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("reformulate", reformulate_node)
    workflow.add_node("scribe", scribe_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Router → conditional: legal or not
    workflow.add_conditional_edges(
        "router",
        is_legal_check,
        {
            "legal": "cache_check",
            "not_legal": "decline",
        },
    )

    # Decline → END
    workflow.add_edge("decline", END)

    # Cache → conditional: hit or miss
    workflow.add_conditional_edges(
        "cache_check",
        cache_hit_check,
        {
            "cache_hit": END,
            "cache_miss": "researcher",
        },
    )

    # Researcher → conditional: check quality for retry
    workflow.add_conditional_edges(
        "researcher",
        retrieval_quality_check,
        {
            "retry": "reformulate",
            "proceed": "scribe",
        },
    )

    # Reformulate → researcher (loop back)
    workflow.add_edge("reformulate", "researcher")

    # Researcher → Scribe → END (after retry logic)
    workflow.add_edge("scribe", END)

    return workflow.compile()

# ── Convenience function ─────────────────────────────────────────────────────

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, conversation_history: list = None, thread_id: str = "default") -> dict:
    """
    Run a query through the full Legal AI pipeline.

    Args:
        query: User's input query.
        conversation_history: Past messages for context.
        thread_id: Identifier used in debug traces (cache is global).

    Returns:
        Dict with: response, debug_trace, language, cache_hit, is_legal.
    """
    graph = get_graph()

    # Only the input keys are seeded; every other state field is filled by
    # the nodes themselves (each node read uses .get() with sane defaults).
    initial_state = {
        "query": query,
        "conversation_history": conversation_history or [],
        "debug_trace": [],
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(initial_state, config)

    response = final_state.get("response", "")

    # Cache policy lives here, at the pipeline level — write the fresh answer
    # for legal queries that had retrieved documents (never declines/greetings).
    if (
        response
        and final_state.get("is_legal", True)
        and not final_state.get("cache_hit", False)
        and final_state.get("retrieved_docs")
    ):
        try:
            cache.put(query, response)
        except Exception as e:
            print(f"[Pipeline] Warning: failed to cache response: {e}")

    return {
        "response": response,
        "debug_trace": final_state.get("debug_trace", []),
        "language": final_state.get("language", ""),
        "cache_hit": final_state.get("cache_hit", False),
        "is_legal": final_state.get("is_legal", True),
    }

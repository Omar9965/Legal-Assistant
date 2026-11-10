from controllers.graph_state import AgentState
from controllers.router import RouterAgent
from controllers.researcher import ResearcherAgent
from controllers.scribe import ScribeAgent
from models.query_expansion import fallback_expansions
from models import cache
from utils.config import CACHE_ROUTE_CONFIDENCE

router_agent = RouterAgent()
cache_agent = cache
researcher_agent = ResearcherAgent()
scribe_agent = ScribeAgent()

def router_node(state: AgentState) -> dict:
    """Step 1: Route the query — detect language and legal relevance."""
    result = router_agent.execute(state["query"])

    trace_entry = f"[Router] Language: {result['language']} | Legal: {result['is_legal']} | Confidence: {result['confidence']:.2f} | Reason: {result['reason']}"
    debug_trace = state.get("debug_trace", []) + [trace_entry]

    return {
        "language": result["language"],
        "is_legal": result["is_legal"],
        "is_greeting": result["is_greeting"],
        "router_confidence": result["confidence"],
        "debug_trace": debug_trace,
    }


def decline_node(state: AgentState) -> dict:
    """Handle non-legal queries with a polite decline or direct greeting.

    The router already classified the query; we consume its decision instead
    of re-running greeting detection here.
    """
    language = state.get("language", "ar")

    if state.get("is_greeting", False):
        if language == "ar":
            message = (
                "أهلاً بك! أنا مستشار قانوني متخصص في القانون المدني المصري. "
                "يسعدني مساعدتك في أي استفسار قانوني. كيف يمكنني خدمتك؟"
            )
        else:
            message = (
                "Hello! I'm a legal assistant specializing in the Egyptian Civil Code. "
                "Happy to help you with any legal inquiries. How can I assist you today?"
            )
        trace_entry = "[Decline] Greeting detected. Returning welcome message."
    else:
        if language == "ar":
            message = (
                "عذراً، أنا مساعد قانوني متخصص في القانون المدني المصري. "
                "لا أستطيع المساعدة في الأسئلة غير القانونية. "
                "يرجى طرح سؤال متعلق بالقانون المدني المصري."
            )
        else:
            message = (
                "I apologize, but I am a legal assistant specializing in the Egyptian Civil Code. "
                "I cannot assist with non-legal queries. "
                "Please ask a question related to Egyptian civil law."
            )
        trace_entry = "[Decline] Query classified as non-legal. Returning decline message."

    debug_trace = state.get("debug_trace", []) + [trace_entry]

    return {
        "response": message,
        "debug_trace": debug_trace,
    }


def cache_node(state: AgentState) -> dict:
    """Step 2: Check the response cache for similar past queries.

    Only check cache if the router's legal-classification confidence is high
    enough (CACHE_ROUTE_CONFIDENCE), otherwise skip to research.
    """
    router_confidence = state.get("router_confidence", 1.0)

    # Skip cache for uncertain queries
    if router_confidence < CACHE_ROUTE_CONFIDENCE:
        trace_entry = f"[Cache] SKIP (low confidence) | Router confidence: {router_confidence:.2f}"
        debug_trace = state.get("debug_trace", []) + [trace_entry]

        return {
            "cache_hit": False,
            "debug_trace": debug_trace,
        }

    # Normal cache check
    cached_answer, similarity = cache_agent.get(state["query"])
    cache_hit = cached_answer is not None

    hit_status = "HIT" if cache_hit else "MISS"
    trace_entry = f"[Cache] {hit_status} | Similarity: {similarity:.4f}"
    debug_trace = state.get("debug_trace", []) + [trace_entry]

    update = {
        "cache_hit": cache_hit,
        "debug_trace": debug_trace,
    }

    if cache_hit:
        update["response"] = cached_answer

    return update


def researcher_node(state: AgentState) -> dict:
    """Step 3: Retrieve relevant legal documents.

    Handles iterative retry with reformulated queries.
    """
    attempts = state.get("retrieval_attempts", 0)
    reformulated = state.get("reformulated_query")

    result = researcher_agent.execute(
        state["query"],
        state.get("language", "ar"),
        query_refinement=reformulated
    )

    num_results = len(result["retrieved_docs"])
    strategy = result["search_metadata"].get("strategy", "unknown")

    # Determine quality. An exact article lookup returning 1 document is a
    # full success (the article was found), not an "insufficient" retrieval.
    if num_results == 0:
        quality = "empty"
    elif num_results < 2 and strategy != "article_lookup":
        quality = "insufficient"
    else:
        quality = "success"

    new_attempts = attempts + 1

    trace_entry = f"[Retriever] Attempt {new_attempts} | Strategy: {strategy} | Docs: {num_results} | Quality: {quality}"

    if result["retrieved_docs"]:
        first_doc = result["retrieved_docs"][0]
        article = first_doc.metadata.get("article_number", "N/A")
        trace_entry += f" | Top article: {article}"

    debug_trace = state.get("debug_trace", []) + [trace_entry]

    return {
        "retrieved_docs": result["retrieved_docs"],
        "search_metadata": result["search_metadata"],
        "retrieval_attempts": new_attempts,
        "last_retrieval_quality": quality,
        "debug_trace": debug_trace,
    }


def scribe_node(state: AgentState) -> dict:
    """Step 4: Generate the final legal response."""
    retrieved_docs = state.get("retrieved_docs", [])
    search_metadata = state.get("search_metadata", {})

    num_docs = len(retrieved_docs)
    max_relevance = search_metadata.get("max_relevance", 1.0 if retrieved_docs else 0.0)

    answer = scribe_agent.execute(
        query=state["query"],
        retrieved_docs=retrieved_docs,
        conversation_history=state.get("conversation_history", []),
        language=state.get("language", "ar"),
        num_docs=num_docs,
        max_relevance=max_relevance,
    )

    trace_entry = f"[Synthesis] Generated response ({len(answer)} chars) in {state.get('language', 'ar')} | Docs: {num_docs} | Relevance: {max_relevance:.2f}"
    debug_trace = state.get("debug_trace", []) + [trace_entry]

    return {
        "response": answer,
        "debug_trace": debug_trace,
    }

def reformulate_node(state: AgentState) -> dict:
    """Reformulate the search query using fast local expansion (no LLM call).

    Uses synonym substitution and keyword expansion instead of calling the LLM,
    which saves 5-15 seconds on retry.
    """
    query = state.get("query", "")
    language = state.get("language", "ar")

    # Fast local expansions (no LLM call)
    reformulations = fallback_expansions(query, language)

    # Pick the best (first) reformulation
    best_reformulation = reformulations[0] if reformulations else query

    trace_entry = f"[Reformulate] Original: '{query}' -> Selected: '{best_reformulation}'"
    debug_trace = state.get("debug_trace", []) + [trace_entry]

    return {
        "reformulated_query": best_reformulation,
        "debug_trace": debug_trace,
    }

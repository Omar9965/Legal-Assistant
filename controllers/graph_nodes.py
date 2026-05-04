from controllers.graph_state import AgentState
from controllers.router import RouterAgent
from controllers.cache_controller import CacheAgent
from controllers.researcher import ResearcherAgent
from controllers.scribe import ScribeAgent

router_agent = RouterAgent()
cache_agent = CacheAgent()
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
        "router_confidence": result["confidence"],
        "router_reason": result["reason"],
        "debug_trace": debug_trace,
    }


def decline_node(state: AgentState) -> dict:
    """Handle non-legal queries with a polite decline or direct greeting."""
    language = state.get("language", "ar")
    query = state.get("query", "")

    ar_greetings = {"مرحبا", "أهلا", "السلام عليكم", "السلام عليكم ورحمة الله", "اهلا", "هاي", "هلا", "صباح الخير", "مساء الخير", "مساء الورد"}
    en_greetings = {"hello", "hi", "hey", "hi there", "hello there", "good morning", "good evening", "good afternoon", "greetings"}

    greeting_found = False
    if language == "ar":
        greeting_found = any(g in query for g in ar_greetings)
    else:
        greeting_found = any(g.lower() in query.lower() for g in en_greetings)

    if greeting_found:
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
    """Step 2: Check the semantic cache for similar past queries.
    
    Only check cache if router confidence is high (>0.8), otherwise skip to research.
    """
    router_confidence = state.get("router_confidence", 1.0)
    
    # Skip cache for uncertain queries
    if router_confidence < 0.8:
        hit_status = "SKIP (low confidence)"
        trace_entry = f"[Cache] {hit_status} | Router confidence: {router_confidence:.2f}"
        debug_trace = state.get("debug_trace", []) + [trace_entry]
        
        return {
            "cache_hit": False,
            "cached_answer": None,
            "similarity_score": 0.0,
            "debug_trace": debug_trace,
        }
    
    # Normal cache check
    result = cache_agent.execute(state["query"])
    
    hit_status = "HIT" if result["cache_hit"] else "MISS"
    trace_entry = f"[Cache] {hit_status} | Similarity: {result['similarity_score']:.4f}"
    debug_trace = state.get("debug_trace", []) + [trace_entry]
    
    update = {
        "cache_hit": result["cache_hit"],
        "cached_answer": result["cached_answer"],
        "similarity_score": result["similarity_score"],
        "debug_trace": debug_trace,
    }
    
    if result["cache_hit"]:
        update["response"] = result["cached_answer"]
    
    return update


def researcher_node(state: AgentState) -> dict:
    """Step 3: Retrieve relevant legal documents.
    
    Handles iterative retry with reformulated queries.
    """
    attempts = state.get("retrieval_attempts", 0)
    reformulated = state.get("reformulated_query")
    
    # Use reformulated query if available, otherwise original
    query_to_search = reformulated if reformulated else state["query"]
    
    result = researcher_agent.execute(
        state["query"], 
        state.get("language", "ar"),
        query_refinement=reformulated
    )
    
    # Determine quality
    num_results = len(result["retrieved_docs"])
    if num_results == 0:
        quality = "empty"
    elif num_results < 2:
        quality = "insufficient"
    else:
        quality = "success"
    
    # Increment attempts
    new_attempts = attempts + 1
    
    strategy = result["search_metadata"].get("strategy", "unknown")
    trace_entry = f"[Retriever] Attempt {new_attempts} | Strategy: {strategy} | Docs: {num_results} | Quality: {quality}"
    
    # Add snippet of what was found
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
    has_retrieved_docs = len(retrieved_docs) > 0

    num_docs = len(retrieved_docs)
    max_relevance = search_metadata.get("max_relevance", 1.0 if retrieved_docs else 0.0)

    answer = scribe_agent.execute(
        query=state["query"],
        retrieved_docs=retrieved_docs,
        conversation_history=state.get("conversation_history", []),
        language=state.get("language", "ar"),
        cache_result=has_retrieved_docs,
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
    """Reformulate the search query based on prior poor results."""
    query = state.get("query", "")
    language = state.get("language", "ar")
    
    # Get the LLM to reformulate
    from langchain_core.prompts import ChatPromptTemplate
    from utils.config import get_llm
    
    reformulate_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a legal query reformulation expert.
The previous retrieval attempt returned insufficient or empty results.
Reformulate this query to improve semantic search results.

Guidelines:
- Make the query more generic/broader if too specific
- Add relevant legal keywords in Arabic
- Consider alternative phrasings for the same legal concept
- If user mentioned article number, keep it

Return ONLY the reformulated query as a single sentence.
Do NOT add explanations or quotes."""),
        ("human", f"Original query: {query}\nLanguage: {language}")
    ])
    
    try:
        llm = get_llm()
        chain = reformulate_prompt | llm
        response = chain.invoke({"query": query, "lang": language})
        reformulated = response.content.strip()
    except Exception as e:
        print(f"[Reformulate] Error: {e}")
        reformulated = query
    
    trace_entry = f"[Reformulate] '{query}' → '{reformulated}'"
    debug_trace = state.get("debug_trace", []) + [trace_entry]
    
    return {
        "reformulated_query": reformulated,
        "debug_trace": debug_trace,
    }

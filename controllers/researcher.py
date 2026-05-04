"""
Researcher — Performs semantic vector retrieval across legal document collections.

Uses LLM structured output to extract optimal search query and article numbers
from user queries. Falls back to regex-based extraction when LLM fails.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from models.vector_store import search
from models.document_processor import extract_article_number
from utils.config import TOP_K, LEGAL_AR_COLLECTION, get_llm
from controllers.base_agent import BaseAgent


class LegalSearchSchema(BaseModel):
    """Schema for extracting search parameters from a legal query."""
    search_query: str = Field(description="The semantic search query in Arabic. It should be concise and optimized for vector search.")
    article_number: Optional[str] = Field(None, description="Specific article number if mentioned by the user (e.g., '147').")


RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Egyptian legal researcher.
Your goal is to convert the user's query into the best semantic search query for a vector database.

- Translate the intent into a clean, semantic Arabic `search_query`.
- If the user explicitly asks about a specific 'مادة' (article), extract the number into `article_number`.
- Do NOT use any category filters — search purely by semantic meaning.

Respond strictly according to the instructed schema."""),
    ("human", "{query}")
])


class ResearcherAgent(BaseAgent):
    """
    Agent responsible for finding the most relevant articles and passages 
    for a given query using LLM structured output / tool calling for parameter extraction.
    """
    
    def execute(self, query: str, language: str = "ar", query_refinement: str = None) -> dict:
        """
        Retrieve relevant legal documents for a given query by using the LLM
        to determine the optimal search parameters.
        
        Args:
            query: The user's legal query.
            language: Detected language ("ar" or "en").
            query_refinement: Optional reformulated query for retry (overrides extraction).
        
        Returns:
            Dict with:
                - retrieved_docs: List of Document objects.
                - search_metadata: Info about what was searched and how.
        """
        # Use refined query if provided
        effective_query = query_refinement if query_refinement else query
        
        # ── Step 1: Regex pre-extraction (always reliable) ───────────────
        regex_article = extract_article_number(query)

        # ── Step 2: Try LLM structured extraction ───────────────────────
        llm = get_llm()
        
        try:
            # Bind the schema as a tool so the LLM returns structured search instructions
            llm_with_tools = llm.with_structured_output(LegalSearchSchema)
            
            chain = RESEARCHER_PROMPT | llm_with_tools
            
            # We use structured output to get the exact arguments autonomously
            search_params: LegalSearchSchema = chain.invoke({"query": query})
            
            extracted_query = search_params.search_query
            article_ref = search_params.article_number
            
            strategy = "llm_extracted"
        except Exception as e:
            print(f"[ResearcherAgent] LLM extraction failed, falling back to basic query: {e}")
            extracted_query = query
            article_ref = None
            strategy = "fallback_semantic"

        # ── Step 3: Override with regex result if LLM missed the article ─
        if regex_article and not article_ref:
            article_ref = regex_article
            strategy = "regex_article_override"
            print(f"[ResearcherAgent] Regex found article {regex_article} (LLM missed it)")

        filter_dict = {"article_number": article_ref} if article_ref else None

        # Execute the search against the legal collection
        if article_ref:
            results = search(extracted_query, LEGAL_AR_COLLECTION, top_k=TOP_K, filter_dict=filter_dict)
            strategy += "_article_lookup"
        else:
            results = search(extracted_query, LEGAL_AR_COLLECTION, top_k=TOP_K)
            strategy += "_semantic"

        return {
            "retrieved_docs": results,
            "search_metadata": {
                "strategy": strategy,
                "extracted_query": extracted_query,
                "article_number": article_ref,
                "num_results": len(results),
                "max_relevance": 1.0 if results else 0.0,
            },
        }
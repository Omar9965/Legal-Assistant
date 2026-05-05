"""
Researcher — Performs advanced semantic vector retrieval across legal document collections.

Enhanced features:
- Hybrid search (vector + keyword)
- Multi-query expansion
- Metadata-based filtering and scoring
- Re-ranking of results
- Multiple search strategies
"""

import re
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from models.vector_store import search, hybrid_search
from models.query_expansion import expand_query
from models.metadata_filtering import (
    MetadataFilter, RelevanceScorer, create_query_filters, FilterCriteria, FilterType
)
from models.document_processor import extract_article_number
from utils.config import TOP_K, LEGAL_AR_COLLECTION, get_llm
from controllers.base_agent import BaseAgent


class LegalSearchSchema(BaseModel):
    """Schema for extracting search parameters from a legal query."""
    search_query: str = Field(description="The semantic search query in Arabic. It should be concise and optimized for vector search.")
    article_number: Optional[str] = Field(None, description="Specific article number if mentioned by the user (e.g., '147').")
    category: Optional[str] = Field(None, description="Detected legal category from the query.")
    strategy: str = Field(description="Search strategy to use: 'semantic', 'hybrid', 'article_lookup', 'expanded_search'.")


RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Egyptian legal researcher.
Your goal is to convert the user's query into the best search strategy for a vector database.

- Extract the core semantic meaning into a clean `search_query`
- If the user explicitly asks about a specific 'مادة' (article), extract the number into `article_number`
- Detect the legal category (contracts, obligations, property, inheritance, evidence, persons)
- Choose the best search strategy based on the query type:
  * 'semantic': General legal queries using vector similarity
  * 'hybrid': Complex queries needing both vector and keyword search
  * 'article_lookup': Specific article number queries
  * 'expanded_search': Broader legal concept queries

Respond strictly according to the instructed schema."""),
    ("human", "{query}")
])


class ResearcherAgent(BaseAgent):
    """
    Enhanced researcher agent with advanced retrieval strategies.
    """
    
    def __init__(self):
        self.metadata_filter = MetadataFilter()
        self.relevance_scorer = RelevanceScorer()
    
    def execute(self, query: str, language: str = "ar", query_refinement: str = None) -> dict:
        """
        Retrieve relevant legal documents using enhanced search strategies.
        
        Args:
            query: The user's legal query.
            language: Detected language ("ar" or "en").
            query_refinement: Optional reformulated query for retry.
        
        Returns:
            Dict with retrieved documents and search metadata.
        """
        # Use refined query if provided
        effective_query = query_refinement if query_refinement else query
        
        # Extract search parameters
        search_params = self._extract_search_parameters(query, language)
        
        # Create filters based on query analysis
        filters = create_query_filters(effective_query, language)
        
        # Execute search based on strategy
        results = self._execute_search(
            search_params=search_params,
            filters=filters,
            query=effective_query,
            language=language
        )
        
        # Score and rank results
        scored_results = self.relevance_scorer.score_documents(
            results, effective_query, language
        )
        
        # Extract top results
        top_results = [doc for doc, score in scored_results[:TOP_K]]
        
        return {
            "retrieved_docs": top_results,
            "search_metadata": {
                "strategy": search_params.strategy,
                "extracted_query": search_params.search_query,
                "article_number": search_params.article_number,
                "category": search_params.category,
                "num_results": len(top_results),
                "max_relevance": scored_results[0][1] if scored_results else 0.0,
                "filters_applied": [f.filter_type.value for f in filters],
            },
        }
    
    def _extract_search_parameters(self, query: str, language: str = "ar") -> LegalSearchSchema:
        """Extract search parameters using LLM structured output."""
        
        llm = get_llm()
        
        try:
            # Bind the schema as a tool
            llm_with_tools = llm.with_structured_output(LegalSearchSchema)
            
            chain = RESEARCHER_PROMPT | llm_with_tools
            search_params: LegalSearchSchema = chain.invoke({"query": query})
            
            # Override article number with regex extraction (more reliable)
            regex_article = extract_article_number(query)
            if regex_article and not search_params.article_number:
                search_params.article_number = regex_article
                search_params.strategy = "article_lookup"
            
            # Determine strategy based on query characteristics
            if search_params.article_number:
                search_params.strategy = "article_lookup"
            elif self._is_complex_query(query, language):
                search_params.strategy = "hybrid"
            elif self._is_expansion_needed(query, language):
                search_params.strategy = "expanded_search"
            else:
                search_params.strategy = "semantic"
            
            return search_params
            
        except Exception as e:
            print(f"[ResearcherAgent] LLM extraction failed: {e}")
            return self._fallback_search_params(query, language)
    
    def _fallback_search_params(self, query: str, language: str = "ar") -> LegalSearchSchema:
        """Fallback search parameter extraction."""
        
        regex_article = extract_article_number(query)
        
        if regex_article:
            strategy = "article_lookup"
        else:
            strategy = "semantic"
        
        return LegalSearchSchema(
            search_query=query,
            article_number=regex_article,
            category=None,
            strategy=strategy
        )
    
    def _is_complex_query(self, query: str, language: str = "ar") -> bool:
        """Determine if query requires hybrid search.
        
        Uses word-boundary matching for short conjunctions like Arabic 'و'
        to avoid false positives when they appear inside longer words.
        Multi-word indicators use simple substring matching.
        """
        complex_indicators = {
            "ar": ["مع", "إضافة إلى", "بالإضافة إلى", "فضلا عن", "بالنسبة"],
            "en": ["in addition to", "plus", "as well as", "regarding"],
        }
        
        # Word-boundary patterns for short conjunctions that are too common as substrings
        boundary_indicators = {
            "ar": [r"(?:^|\s)و(?:\s|$)"],  # standalone "و" only
            "en": [r"\band\b", r"\bwith\b"],
        }
        
        indicators = complex_indicators.get(language, [])
        if any(indicator in query for indicator in indicators):
            return True
        
        patterns = boundary_indicators.get(language, [])
        return any(re.search(pat, query) for pat in patterns)
    
    def _is_expansion_needed(self, query: str, language: str = "ar") -> bool:
        """Determine if query benefits from expansion.
        
        Only triggers for very short queries (< 3 words) that also contain
        overly-general terms. This avoids expanding nearly every legal query.
        """
        word_count = len(query.split())
        
        # Only very short queries benefit from expansion
        if word_count >= 4:
            return False
        
        # Short query AND contains only general terms → expand
        if word_count < 3:
            general_terms = {
                "ar": ["قانون", "حق", "واجب", "مسؤولية"],
                "en": ["law", "right", "duty", "liability"],
            }
            terms = general_terms.get(language, [])
            return any(term in query.lower() for term in terms)
        
        return False
    
    def _execute_search(
        self, 
        search_params: LegalSearchSchema, 
        filters: List[FilterCriteria],
        query: str,
        language: str = "ar"
    ) -> List:
        """Execute search based on the chosen strategy."""
        
        strategy = search_params.strategy
        
        if strategy == "article_lookup" and search_params.article_number:
            return self._article_lookup_search(search_params, filters)
        elif strategy == "hybrid":
            return self._hybrid_search(search_params, filters, query)
        elif strategy == "expanded_search":
            return self._expanded_search(search_params, filters, query, language)
        else:
            return self._semantic_search(search_params, filters)
    
    def _article_lookup_search(self, search_params: LegalSearchSchema, filters: List[FilterCriteria]) -> List:
        """Search for specific article number."""
        
        # Create strict article filter (use FilterType enum, not string)
        article_filter = FilterCriteria(
            filter_type=FilterType.ARTICLE_RANGE,
            value=(int(search_params.article_number), int(search_params.article_number)),
            weight=1.0,
            required=True
        )
        
        all_filters = filters + [article_filter]
        
        # Use hybrid search for article lookup
        results = hybrid_search(
            query=search_params.search_query,
            collection_name=LEGAL_AR_COLLECTION,
            top_k=TOP_K * 2,  # Get more results to ensure we find the article
            filter_dict={},
            vector_weight=0.6,
            keyword_weight=0.4
        )
        
        # Apply additional filtering
        filtered_results = self.metadata_filter.apply_filters(results, all_filters)
        
        return filtered_results
    
    def _hybrid_search(self, search_params: LegalSearchSchema, filters: List[FilterCriteria], query: str) -> List:
        """Execute hybrid search (vector + keyword)."""
        
        # Use hybrid search with balanced weights
        results = hybrid_search(
            query=search_params.search_query,
            collection_name=LEGAL_AR_COLLECTION,
            top_k=TOP_K * 2,
            filter_dict={},
            vector_weight=0.7,
            keyword_weight=0.3
        )
        
        # Apply filters
        filtered_results = self.metadata_filter.apply_filters(results, filters)
        
        return filtered_results
    
    def _expanded_search(self, search_params: LegalSearchSchema, filters: List[FilterCriteria], query: str, language: str) -> List:
        """Execute search with query expansion."""
        
        # Generate expanded queries
        expanded_queries = expand_query(query, language, max_variations=2)
        
        all_results = []
        
        # Search each expanded query
        for expanded_query in expanded_queries:
            results = search(
                query=expanded_query,
                collection_name=LEGAL_AR_COLLECTION,
                top_k=TOP_K,
                filter_dict={}
            )
            all_results.extend(results)
        
        # Remove duplicates (based on content hash)
        unique_results = self._remove_duplicates(all_results)
        
        # Apply filters
        filtered_results = self.metadata_filter.apply_filters(unique_results, filters)
        
        return filtered_results
    
    def _semantic_search(self, search_params: LegalSearchSchema, filters: List[FilterCriteria]) -> List:
        """Execute standard semantic search."""
        
        results = search(
            query=search_params.search_query,
            collection_name=LEGAL_AR_COLLECTION,
            top_k=TOP_K,
            filter_dict={}
        )
        
        # Apply filters
        filtered_results = self.metadata_filter.apply_filters(results, filters)
        
        return filtered_results
    
    def _remove_duplicates(self, documents: List) -> List:
        """Remove duplicate documents based on content similarity."""
        
        # Simple deduplication based on content hash
        seen_hashes = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
"""
Researcher — Performs semantic vector retrieval across legal document collections.

Features:
- Multi-query expansion for broad legal concepts
- Single-pass relevance scoring (no separate filter/sort stages)
- Fast regex-based search parameter extraction (no LLM call)
"""

from typing import List
from models.vector_store import search
from models.query_expansion import expand_query
from models.metadata_filtering import RelevanceScorer
from models.document_processor import extract_article_number
from utils.config import TOP_K, LEGAL_AR_COLLECTION


class ResearcherAgent:
    """
    Researcher agent with fast retrieval — uses regex for parameter
    extraction instead of an LLM call. Ranking happens in exactly one
    scoring pass (RelevanceScorer).
    """

    def __init__(self):
        self.relevance_scorer = RelevanceScorer()

    def execute(self, query: str, language: str = "ar", query_refinement: str = None) -> dict:
        """
        Retrieve relevant legal documents.

        Args:
            query: The user's legal query.
            language: Detected language ("ar" or "en").
            query_refinement: Optional reformulated query for retry.

        Returns:
            Dict with retrieved documents and search metadata.
        """
        effective_query = query_refinement if query_refinement else query

        # Fast regex-based parameter extraction (no LLM call)
        article_number = extract_article_number(query)

        if article_number:
            strategy = "article_lookup"
        elif self._is_expansion_needed(query, language):
            strategy = "expanded_search"
        else:
            strategy = "semantic"

        # Execute search
        if strategy == "article_lookup" and article_number:
            results = self._article_lookup_search(effective_query, article_number)
        elif strategy == "expanded_search":
            results = self._expanded_search(effective_query, language)
        else:
            results = self._semantic_search(effective_query)

        # Single-pass scoring and ranking
        scored_results = self.relevance_scorer.score_documents(
            results, effective_query, language
        )

        top_results = [doc for doc, score in scored_results[:TOP_K]]

        return {
            "retrieved_docs": top_results,
            "search_metadata": {
                "strategy": strategy,
                "extracted_query": effective_query,
                "article_number": article_number,
                "num_results": len(top_results),
                "max_relevance": scored_results[0][1] if scored_results else 0.0,
            },
        }

    def _is_expansion_needed(self, query: str, language: str = "ar") -> bool:
        """Determine if query benefits from expansion.

        Only triggers for very short queries (< 3 words) that also contain
        overly-general terms. This avoids expanding nearly every legal query.
        """
        word_count = len(query.split())

        if word_count >= 4:
            return False

        if word_count < 3:
            general_terms = {
                "ar": ["قانون", "حق", "واجب", "مسؤولية"],
                "en": ["law", "right", "duty", "liability"],
            }
            terms = general_terms.get(language, [])
            return any(term in query.lower() for term in terms)

        return False

    def _article_lookup_search(self, query: str, article_number: str) -> List:
        """Search for a specific article number using ChromaDB's native filter.

        Strategy:
        1. Direct metadata filter via ChromaDB `where` clause (fast, reliable).
        2. If no results, fall back to standard semantic search.
        """
        # Primary: use ChromaDB's native where filter for exact article match
        results = search(
            query=query,
            collection_name=LEGAL_AR_COLLECTION,
            top_k=TOP_K * 2,
            filter_dict={"article_number": article_number},
        )

        if results:
            return results

        # Fallback: broader semantic search without metadata filter
        return search(
            query=query,
            collection_name=LEGAL_AR_COLLECTION,
            top_k=TOP_K,
            filter_dict={},
        )

    def _expanded_search(self, query: str, language: str) -> List:
        """Execute search with query expansion."""
        expanded_queries = expand_query(query, language, max_variations=2)

        all_results = []
        for expanded_query in expanded_queries:
            results = search(
                query=expanded_query,
                collection_name=LEGAL_AR_COLLECTION,
                top_k=TOP_K,
                filter_dict={}
            )
            all_results.extend(results)

        unique_results = self._remove_duplicates(all_results)
        return unique_results

    def _semantic_search(self, query: str) -> List:
        """Execute standard semantic search."""
        return search(
            query=query,
            collection_name=LEGAL_AR_COLLECTION,
            top_k=TOP_K,
            filter_dict={}
        )

    @staticmethod
    def _remove_duplicates(documents: List) -> List:
        """Remove duplicate documents based on content."""
        seen_hashes = set()
        unique_docs = []

        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

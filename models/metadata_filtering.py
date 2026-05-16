"""
Metadata Filtering and Scoring — Advanced filtering and scoring for search results.

Supports:
- Category-based filtering
- Article number range filtering
- Page-based filtering
- Relevance scoring
- Quality assessment
"""

from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
import re
from dataclasses import dataclass
from enum import Enum
from models.document_processor import extract_article_number
from utils.constants import LEGAL_CATEGORIES


class FilterType(Enum):
    """Types of filters that can be applied."""
    CATEGORY = "category"
    ARTICLE_RANGE = "article_range"
    PAGE_RANGE = "page_range"
    LANGUAGE = "language"
    SOURCE = "source"


@dataclass
class FilterCriteria:
    """Criteria for filtering search results."""
    filter_type: FilterType
    value: any
    weight: float = 1.0
    required: bool = False


class MetadataFilter:
    """Advanced metadata filtering for legal documents."""
    
    def __init__(self):
        self.category_keywords = LEGAL_CATEGORIES
    
    def apply_filters(
        self, 
        documents: List[Document], 
        filters: List[FilterCriteria]
    ) -> List[Document]:
        """Apply multiple filters to documents."""
        
        filtered_docs = documents
        
        for filter_criteria in filters:
            if filter_criteria.required:
                filtered_docs = self._apply_required_filter(filtered_docs, filter_criteria)
            else:
                filtered_docs = self._apply_optional_filter(filtered_docs, filter_criteria)
        
        return filtered_docs
    
    def _apply_required_filter(self, documents: List[Document], criteria: FilterCriteria) -> List[Document]:
        """Apply a required filter (documents must match)."""
        if criteria.filter_type == FilterType.CATEGORY:
            return [doc for doc in documents if self._matches_category(doc, criteria.value)]
        elif criteria.filter_type == FilterType.ARTICLE_RANGE:
            return [doc for doc in documents if self._matches_article_range(doc, criteria.value)]
        elif criteria.filter_type == FilterType.PAGE_RANGE:
            return [doc for doc in documents if self._matches_page_range(doc, criteria.value)]
        elif criteria.filter_type == FilterType.LANGUAGE:
            return [doc for doc in documents if doc.metadata.get("language") == criteria.value]
        elif criteria.filter_type == FilterType.SOURCE:
            return [doc for doc in documents if criteria.value in doc.metadata.get("source", "")]
        return documents
    
    def _apply_optional_filter(self, documents: List[Document], criteria: FilterCriteria) -> List[Document]:
        """Apply an optional filter — boosts matching docs but keeps all."""
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_filter_score(doc, criteria)
            # Keep ALL documents; non-matching get a baseline score of 0
            scored_docs.append((doc, score))
        
        # Sort by score (matching docs bubble to the top) and return documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]
    
    def _matches_category(self, doc: Document, category: str) -> bool:
        """Check if document matches the specified category."""
        doc_category = doc.metadata.get("category", "general")
        return doc_category == category
    
    def _matches_article_range(self, doc: Document, article_range: Tuple[int, int]) -> bool:
        """Check if document article number falls within the specified range."""
        article_num = doc.metadata.get("article_number")
        if not article_num:
            return False
        
        try:
            article_int = int(article_num)
            start, end = article_range
            return start <= article_int <= end
        except ValueError:
            return False
    
    def _matches_page_range(self, doc: Document, page_range: Tuple[int, int]) -> bool:
        """Check if document page number falls within the specified range."""
        page_num = doc.metadata.get("page_number")
        if not page_num:
            return False
        
        try:
            page_int = int(page_num)
            start, end = page_range
            return start <= page_int <= end
        except ValueError:
            return False
    
    def _calculate_filter_score(self, doc: Document, criteria: FilterCriteria) -> float:
        """Calculate a score for how well a document matches a filter."""
        
        if criteria.filter_type == FilterType.CATEGORY:
            return self._calculate_category_score(doc, criteria.value, criteria.weight)
        elif criteria.filter_type == FilterType.ARTICLE_RANGE:
            return self._calculate_article_range_score(doc, criteria.value, criteria.weight)
        elif criteria.filter_type == FilterType.PAGE_RANGE:
            return self._calculate_page_range_score(doc, criteria.value, criteria.weight)
        elif criteria.filter_type == FilterType.LANGUAGE:
            return self._calculate_language_score(doc, criteria.value, criteria.weight)
        elif criteria.filter_type == FilterType.SOURCE:
            return self._calculate_source_score(doc, criteria.value, criteria.weight)
        
        return 0.0
    
    def _calculate_category_score(self, doc: Document, category: str, weight: float) -> float:
        """Calculate category matching score."""
        doc_category = doc.metadata.get("category", "general")
        if doc_category == category:
            return 1.0 * weight
        return 0.0
    
    def _calculate_article_range_score(self, doc: Document, article_range: Tuple[int, int], weight: float) -> float:
        """Calculate article range score based on proximity."""
        article_num = doc.metadata.get("article_number")
        if not article_num:
            return 0.0
        
        try:
            article_int = int(article_num)
            start, end = article_range
            
            # Calculate how close the article is to the center of the range
            center = (start + end) / 2
            distance = abs(article_int - center)
            max_distance = (end - start) / 2
            
            if max_distance == 0:
                return 1.0 * weight if article_int == start else 0.0
            
            # Score decreases with distance from center
            score = max(0.0, 1.0 - (distance / max_distance))
            return score * weight
            
        except ValueError:
            return 0.0
    
    def _calculate_page_range_score(self, doc: Document, page_range: Tuple[int, int], weight: float) -> float:
        """Calculate page range score."""
        page_num = doc.metadata.get("page_number")
        if not page_num:
            return 0.0
        
        try:
            page_int = int(page_num)
            start, end = page_range
            
            if start <= page_int <= end:
                return 1.0 * weight
            return 0.0
            
        except ValueError:
            return 0.0
    
    def _calculate_language_score(self, doc: Document, language: str, weight: float) -> float:
        """Calculate language matching score."""
        doc_language = doc.metadata.get("language", "ar")
        return 1.0 * weight if doc_language == language else 0.0
    
    def _calculate_source_score(self, doc: Document, source: str, weight: float) -> float:
        """Calculate source matching score."""
        doc_sources = doc.metadata.get("source", [])
        if isinstance(doc_sources, str):
            doc_sources = [doc_sources]
        
        for src in doc_sources:
            if source.lower() in src.lower():
                return 1.0 * weight
        
        return 0.0


class RelevanceScorer:
    """Score search results based on multiple relevance factors."""
    
    def score_documents(
        self, 
        documents: List[Document], 
        query: str,
        language: str = "ar"
    ) -> List[Tuple[Document, float]]:
        """Score documents based on multiple relevance factors."""
        
        scored_docs = []
        
        for doc in documents:
            # Calculate multiple scores
            semantic_score = self._calculate_semantic_score(doc, query, language)
            category_score = self._calculate_category_score(doc, query, language)
            
            # Combine scores: embedding similarity is the primary signal,
            # category alignment is a secondary metadata bonus.
            total_score = (
                semantic_score * 0.7 +
                category_score * 0.3
            )
            
            scored_docs.append((doc, total_score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
    
    def _calculate_semantic_score(self, doc: Document, query: str, language: str) -> float:
        """Use the embedding-based relevance score computed during vector search.

        The score is attached to doc.metadata['relevance_score'] by
        vector_store.search() using the Muffakir embedding model.
        Falls back to 0.5 if the score is missing (e.g., manually created docs).
        """
        return doc.metadata.get("relevance_score", 0.5)
    
    def _calculate_category_score(self, doc: Document, query: str, language: str) -> float:
        """Score based on whether the document's legal category matches the query's topic.

        Uses detect_category_from_query (keyword heuristic) for the query side,
        then checks alignment with the document's pre-assigned category metadata.
        Returns 1.0 for a match, 0.3 for general/unknown categories (benefit of
        the doubt), and 0.0 for a definite mismatch.
        """
        doc_category = doc.metadata.get("category", "general")
        query_category = detect_category_from_query(query, language)
        
        if not query_category:
            # No category detected in query — don't penalize any document
            return 0.5
        
        if doc_category == query_category:
            return 1.0
        
        if doc_category == "general":
            # Generic docs might still be relevant
            return 0.3
        
        return 0.0




def create_query_filters(query: str, language: str = "ar") -> List[FilterCriteria]:
    """Create filter criteria based on query analysis.
    """
    
    filters = []
    
    # Extract article number for filtering
    article_num = extract_article_number(query)
    if article_num:
        # Create article range around the mentioned article
        article_int = int(article_num)
        range_start = max(1, article_int - 2)
        range_end = article_int + 2
        filters.append(FilterCriteria(
            filter_type=FilterType.ARTICLE_RANGE,
            value=(range_start, range_end),
            weight=0.8,
            required=False
        ))
    
    # Extract category from query
    category = detect_category_from_query(query, language)
    if category:
        filters.append(FilterCriteria(
            filter_type=FilterType.CATEGORY,
            value=category,
            weight=0.6,
            required=False
        ))
    
    return filters


def detect_category_from_query(query: str, language: str = "ar") -> Optional[str]:
    """Detect legal category from query."""
    categories = LEGAL_CATEGORIES
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in query.lower():
                return category
    
    return None
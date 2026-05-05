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
        # Legal category definitions
        self.category_keywords = {
            "contracts": ["عقد", "عقود", "تعاقد", "إيجاب", "قبول", "اتفاقية"],
            "obligations": ["التزام", "التزامات", "مسؤولية", "تعويض", "دين"],
            "property": ["ملكية", "حيازة", "عقار", "ارتفاق", "أرض", "مبنى"],
            "inheritance": ["ميراث", "إرث", "وصية", "تركة", "وارث"],
            "evidence": ["إثبات", "بينة", "شهادة", "دليل", "برهان"],
            "persons": ["أهلية", "شخصية", "ولاية", "وصاية", "قاصر"],
            "family": ["زواج", "طلاق", "نفقة", "حضانة", "نسب"],
            "procedural": ["دعوى", "محكمة", "قضاء", "تنفيذ", "حجز"],
        }
    
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
            position_score = self._calculate_position_score(doc)
            recency_score = self._calculate_recency_score(doc)
            
            # Combine scores with weights
            total_score = (
                semantic_score * 0.4 +
                category_score * 0.3 +
                position_score * 0.2 +
                recency_score * 0.1
            )
            
            scored_docs.append((doc, total_score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
    
    def _calculate_semantic_score(self, doc: Document, query: str, language: str) -> float:
        """Calculate semantic relevance score."""
        # This is a simplified version - in practice, you'd use embeddings
        doc_text = doc.page_content.lower()
        query_words = query.lower().split()
        
        # Count query word matches
        matches = sum(1 for word in query_words if word in doc_text)
        
        # Normalize by query length
        if len(query_words) == 0:
            return 0.0
        
        return min(1.0, matches / len(query_words))
    
    def _calculate_category_score(self, doc: Document, query: str, language: str) -> float:
        """Calculate category relevance score."""
        doc_category = doc.metadata.get("category", "general")
        
        # Check if query contains category keywords
        category_keywords = {
            "contracts": ["عقد", "عقود", "تعاقد"],
            "obligations": ["التزام", "مسؤولية", "تعويض"],
            "property": ["ملكية", "عقار", "حيازة"],
            "inheritance": ["ميراث", "إرث", "وصية"],
            "evidence": ["إثبات", "بينة", "شهادة"],
            "persons": ["أهلية", "شخصية", "ولاية"],
        }
        
        for category, keywords in category_keywords.items():
            if category == doc_category:
                for keyword in keywords:
                    if keyword in query.lower():
                        return 1.0
        
        return 0.0
    
    def _calculate_position_score(self, doc: Document) -> float:
        """Calculate position-based score (earlier chunks get higher scores)."""
        chunk_index = doc.metadata.get("chunk_index", 0)
        sub_chunk_index = doc.metadata.get("sub_chunk_index", 0)
        
        # Earlier positions get higher scores
        position_score = 1.0 / (1 + chunk_index + sub_chunk_index * 0.1)
        return position_score
    
    def _calculate_recency_score(self, doc: Document) -> float:
        """Calculate recency-based score (if page number is available)."""
        page_num = doc.metadata.get("page_number")
        if not page_num:
            return 0.5  # Default score for documents without page info
        
        try:
            page_int = int(page_num)
            # Assume higher page numbers are newer (this may need adjustment)
            return min(1.0, page_int / 1000)  # Normalize to 0-1
        except ValueError:
            return 0.5


def create_query_filters(query: str, language: str = "ar") -> List[FilterCriteria]:
    """Create filter criteria based on query analysis.
    
    Note: Language filter is intentionally omitted because the corpus is
    entirely Arabic. Filtering by the user's query language would exclude
    all documents when the user writes in English.
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
    categories = {
        "contracts": ["عقد", "عقود", "تعاقد", "إيجاب", "قبول", "اتفاقية"],
        "obligations": ["التزام", "التزامات", "مسؤولية", "تعويض", "دين"],
        "property": ["ملكية", "حيازة", "عقار", "ارتفاق", "أرض", "مبنى"],
        "inheritance": ["ميراث", "إرث", "وصية", "تركة", "وارث"],
        "evidence": ["إثبات", "بينة", "شهادة", "دليل", "برهان"],
        "persons": ["أهلية", "شخصية", "ولاية", "وصاية", "قاصر"],
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in query.lower():
                return category
    
    return None
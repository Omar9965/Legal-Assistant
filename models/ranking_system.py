"""
Re-ranking System — Advanced re-ranking of search results using multiple scoring strategies.

Supports:
- Cross-encoder re-ranking
- Multi-factor scoring
- Context-aware ranking
- Quality assessment
- Result diversification
"""

from typing import List, Tuple, Dict, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dataclasses import dataclass
from enum import Enum


class RankingStrategy(Enum):
    """Different ranking strategies available."""
    CROSS_ENCODER = "cross_encoder"
    MULTI_FACTOR = "multi_factor"
    CONTEXT_AWARE = "context_aware"
    DIVERSITY = "diversity"


@dataclass
class RankingConfig:
    """Configuration for ranking strategy."""
    strategy: RankingStrategy
    weights: Dict[str, float]
    max_results: int
    diversity_threshold: float = 0.3
    context_window: int = 3


class ReRanker:
    """Advanced re-ranking system for search results."""
    
    def __init__(self):
        self.ranking_configs = {
            RankingStrategy.CROSS_ENCODER: RankingConfig(
                strategy=RankingStrategy.CROSS_ENCODER,
                weights={"semantic": 0.5, "relevance": 0.3, "quality": 0.2},
                max_results=10
            ),
            RankingStrategy.MULTI_FACTOR: RankingConfig(
                strategy=RankingStrategy.MULTI_FACTOR,
                weights={"semantic": 0.4, "position": 0.2, "category": 0.2, "recency": 0.2},
                max_results=15
            ),
            RankingStrategy.CONTEXT_AWARE: RankingConfig(
                strategy=RankingStrategy.CONTEXT_AWARE,
                weights={"relevance": 0.5, "context": 0.3, "coherence": 0.2},
                max_results=12,
                context_window=5
            ),
            RankingStrategy.DIVERSITY: RankingConfig(
                strategy=RankingStrategy.DIVERSITY,
                weights={"relevance": 0.6, "diversity": 0.4},
                max_results=10,
                diversity_threshold=0.3
            )
        }
    
    def rerank_results(
        self,
        results: List[Document],
        query: str,
        strategy: RankingStrategy = RankingStrategy.MULTI_FACTOR,
        metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank search results using the specified strategy.
        
        Args:
            results: List of documents to re-rank.
            query: Original search query.
            strategy: Ranking strategy to use.
            metadata: Additional metadata for ranking.
        
        Returns:
            List of (document, score) tuples sorted by rank.
        """
        config = self.ranking_configs[strategy]
        
        if strategy == RankingStrategy.CROSS_ENCODER:
            return self._cross_encoder_ranking(results, query, config)
        elif strategy == RankingStrategy.MULTI_FACTOR:
            return self._multi_factor_ranking(results, query, config, metadata)
        elif strategy == RankingStrategy.CONTEXT_AWARE:
            return self._context_aware_ranking(results, query, config)
        elif strategy == RankingStrategy.DIVERSITY:
            return self._diversity_ranking(results, query, config)
        else:
            return self._multi_factor_ranking(results, query, config, metadata)
    
    def _cross_encoder_ranking(
        self, 
        results: List[Document], 
        query: str, 
        config: RankingConfig
    ) -> List[Tuple[Document, float]]:
        """
        Cross-encoder based ranking (simulated since we don't have actual cross-encoder).
        In production, you'd use a model like 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        
        # Simulate cross-encoder scoring
        scored_results = []
        
        for doc in results:
            # Calculate multiple factors
            semantic_score = self._calculate_semantic_similarity(doc, query)
            relevance_score = self._calculate_relevance_score(doc, query)
            quality_score = self._calculate_quality_score(doc)
            
            # Combine scores with weights
            total_score = (
                semantic_score * config.weights["semantic"] +
                relevance_score * config.weights["relevance"] +
                quality_score * config.weights["quality"]
            )
            
            scored_results.append((doc, total_score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply max_results limit
        return scored_results[:config.max_results]
    
    def _multi_factor_ranking(
        self, 
        results: List[Document], 
        query: str, 
        config: RankingConfig,
        metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """Multi-factor ranking using multiple signals."""
        
        scored_results = []
        
        for doc in results:
            # Calculate multiple factors
            semantic_score = self._calculate_semantic_similarity(doc, query)
            position_score = self._calculate_position_score(doc)
            category_score = self._calculate_category_score(doc, query)
            recency_score = self._calculate_recency_score(doc)
            
            # Combine scores with weights
            total_score = (
                semantic_score * config.weights["semantic"] +
                position_score * config.weights["position"] +
                category_score * config.weights["category"] +
                recency_score * config.weights["recency"]
            )
            
            scored_results.append((doc, total_score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply max_results limit
        return scored_results[:config.max_results]
    
    def _context_aware_ranking(
        self, 
        results: List[Document], 
        query: str, 
        config: RankingConfig
    ) -> List[Tuple[Document, float]]:
        """Context-aware ranking considering document coherence and context."""
        
        scored_results = []
        
        for i, doc in enumerate(results):
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(doc, query)
            
            # Calculate context coherence with neighboring documents
            context_score = self._calculate_context_coherence(
                results, i, config.context_window
            )
            
            # Calculate overall quality
            quality_score = self._calculate_quality_score(doc)
            
            # Combine scores
            total_score = (
                relevance_score * config.weights["relevance"] +
                context_score * config.weights["context"] +
                quality_score * config.weights["coherence"]
            )
            
            scored_results.append((doc, total_score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results[:config.max_results]
    
    def _diversity_ranking(
        self, 
        results: List[Document], 
        query: str, 
        config: RankingConfig
    ) -> List[Tuple[Document, float]]:
        """Diversity-aware ranking to ensure result variety."""
        
        # First rank by relevance
        relevance_scored = []
        for doc in results:
            relevance_score = self._calculate_relevance_score(doc, query)
            relevance_scored.append((doc, relevance_score))
        
        relevance_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity-based re-ranking
        diverse_results = []
        seen_categories = set()
        
        for doc, relevance_score in relevance_scored:
            doc_category = doc.metadata.get("category", "general")
            
            # Calculate diversity bonus for new categories
            diversity_bonus = 0.0
            if doc_category not in seen_categories:
                diversity_bonus = config.diversity_threshold
                seen_categories.add(doc_category)
            
            # Combine relevance and diversity
            total_score = relevance_score + diversity_bonus
            diverse_results.append((doc, total_score))
        
        # Sort by combined score
        diverse_results.sort(key=lambda x: x[1], reverse=True)
        
        return diverse_results[:config.max_results]
    
    def _calculate_semantic_similarity(self, doc: Document, query: str) -> float:
        """Calculate semantic similarity between document and query."""
        # Simple implementation - in production, use actual embeddings
        doc_text = doc.page_content.lower()
        query_words = query.lower().split()
        
        # Count matching words
        matches = sum(1 for word in query_words if word in doc_text)
        
        # Normalize by query length
        if len(query_words) == 0:
            return 0.0
        
        return min(1.0, matches / len(query_words))
    
    def _calculate_relevance_score(self, doc: Document, query: str) -> float:
        """Calculate relevance score based on query-document match."""
        # Check for exact matches of key terms
        doc_text = doc.page_content.lower()
        query_lower = query.lower()
        
        # Check for article number matches
        if doc.metadata.get("article_number"):
            article_num = doc.metadata["article_number"]
            if f"مادة {article_num}" in query_lower or f"article {article_num}" in query_lower:
                return 1.0
        
        # Check for category matches
        doc_category = doc.metadata.get("category", "general")
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
                    if keyword in query_lower:
                        return 0.9
        
        # General word matching
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in doc_text)
        return min(1.0, matches / max(1, len(query_words)))
    
    def _calculate_position_score(self, doc: Document) -> float:
        """Calculate position-based score (earlier chunks get higher scores)."""
        chunk_index = doc.metadata.get("chunk_index", 0)
        sub_chunk_index = doc.metadata.get("sub_chunk_index", 0)
        
        # Earlier positions get higher scores
        position_score = 1.0 / (1 + chunk_index + sub_chunk_index * 0.1)
        return position_score
    
    def _calculate_category_score(self, doc: Document, query: str) -> float:
        """Calculate category matching score."""
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
    
    def _calculate_recency_score(self, doc: Document) -> float:
        """Calculate recency-based score (if page number is available)."""
        page_num = doc.metadata.get("page_number")
        if not page_num:
            return 0.5  # Default score for documents without page info
        
        try:
            page_int = int(page_num)
            # Assume higher page numbers are newer
            return min(1.0, page_int / 1000)
        except ValueError:
            return 0.5
    
    def _calculate_quality_score(self, doc: Document) -> float:
        """Calculate document quality score."""
        score = 1.0
        
        # Penalize documents with very short content
        if len(doc.page_content) < 50:
            score *= 0.5
        
        # Reward documents with rich metadata
        metadata = doc.metadata
        if metadata:
            if metadata.get("article_number"):
                score *= 1.1
            if metadata.get("category"):
                score *= 1.05
            if metadata.get("page_number"):
                score *= 1.02
        
        return min(1.0, score)
    
    def _calculate_context_coherence(
        self, 
        results: List[Document], 
        index: int, 
        window: int
    ) -> float:
        """Calculate context coherence with neighboring documents."""
        
        # Get neighboring documents
        start_idx = max(0, index - window)
        end_idx = min(len(results), index + window + 1)
        
        neighbors = results[start_idx:end_idx]
        
        if len(neighbors) <= 1:
            return 1.0  # No neighbors to compare with
        
        # Calculate coherence with neighbors
        coherence_scores = []
        
        for neighbor in neighbors:
            if neighbor != results[index]:
                # Simple coherence check based on shared metadata
                current_meta = results[index].metadata
                neighbor_meta = neighbor.metadata
                
                # Check for shared categories
                if current_meta.get("category") == neighbor_meta.get("category"):
                    coherence_scores.append(1.0)
                else:
                    coherence_scores.append(0.5)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0


class QualityAssessment:
    """Assess the quality of search results."""
    
    def assess_results(
        self, 
        results: List[Document], 
        query: str
    ) -> Dict[str, float]:
        """
        Assess the quality of search results.
        
        Returns:
            Dictionary with quality metrics.
        """
        
        if not results:
            return {
                "coverage": 0.0,
                "relevance": 0.0,
                "diversity": 0.0,
                "completeness": 0.0,
                "overall": 0.0
            }
        
        # Calculate individual metrics
        coverage = self._calculate_coverage(results, query)
        relevance = self._calculate_relevance(results, query)
        diversity = self._calculate_diversity(results)
        completeness = self._calculate_completeness(results, query)
        
        # Calculate overall score
        overall = (
            coverage * 0.3 +
            relevance * 0.4 +
            diversity * 0.2 +
            completeness * 0.1
        )
        
        return {
            "coverage": coverage,
            "relevance": relevance,
            "diversity": diversity,
            "completeness": completeness,
            "overall": overall
        }
    
    def _calculate_coverage(self, results: List[Document], query: str) -> float:
        """Calculate how well results cover the query."""
        query_words = query.lower().split()
        
        covered_words = set()
        for doc in results:
            doc_text = doc.page_content.lower()
            for word in query_words:
                if word in doc_text:
                    covered_words.add(word)
        
        return len(covered_words) / len(query_words) if query_words else 0.0
    
    def _calculate_relevance(self, results: List[Document], query: str) -> float:
        """Calculate average relevance of results."""
        total_relevance = 0.0
        
        for doc in results:
            relevance = self._calculate_single_relevance(doc, query)
            total_relevance += relevance
        
        return total_relevance / len(results) if results else 0.0
    
    def _calculate_single_relevance(self, doc: Document, query: str) -> float:
        """Calculate relevance of a single document."""
        doc_text = doc.page_content.lower()
        query_lower = query.lower()
        
        # Check for exact matches
        if doc.metadata.get("article_number"):
            article_num = doc.metadata["article_number"]
            if f"مادة {article_num}" in query_lower or f"article {article_num}" in query_lower:
                return 1.0
        
        # Count matching words
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in doc_text)
        
        return matches / len(query_words) if query_words else 0.0
    
    def _calculate_diversity(self, results: List[Document]) -> float:
        """Calculate diversity of results (different categories, sources)."""
        if not results:
            return 0.0
        
        categories = set()
        sources = set()
        
        for doc in results:
            categories.add(doc.metadata.get("category", "general"))
            sources.add(doc.metadata.get("source", "unknown"))
        
        # Calculate diversity score
        total_categories = len(categories)
        total_sources = len(sources)
        max_possible = max(1, len(results))
        
        category_diversity = total_categories / max_possible
        source_diversity = min(1.0, total_sources / max_possible)
        
        return (category_diversity + source_diversity) / 2
    
    def _calculate_completeness(self, results: List[Document], query: str) -> float:
        """Calculate completeness of results (adequate coverage)."""
        # Simple completeness check based on result count
        if len(results) >= 5:
            return 1.0
        elif len(results) >= 3:
            return 0.8
        elif len(results) >= 1:
            return 0.5
        else:
            return 0.0
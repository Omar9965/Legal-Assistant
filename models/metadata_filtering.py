"""
Relevance Scoring — single-pass scoring for legal retrieval results.

The embedding similarity computed during vector search and stored in
doc.metadata['relevance_score'] is the primary signal. Alignment between the
document's pre-assigned legal category and the query's detected category adds
a secondary bonus. There is deliberately no separate filter/sort stage:
scoring happens exactly once, in one pass.
"""

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from document_processor import _detect_category







class RelevanceScorer:
    """Score search results: 0.7 * semantic similarity + 0.3 * category match."""

    def score_documents(
        self,
        documents: List[Document],
        query: str,
        language: str = "ar",
    ) -> List[Tuple[Document, float]]:
        """Score and sort documents by relevance. Returns [(doc, score), ...]."""
        query_category = _detect_category(query, language)

        scored_docs = []
        for doc in documents:
            semantic_score = doc.metadata.get("relevance_score", 0.5)
            category_score = self._category_score(doc, query_category)
            total_score = semantic_score * 0.7 + category_score * 0.3
            scored_docs.append((doc, total_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    @staticmethod
    def _category_score(doc: Document, query_category: Optional[str]) -> float:
        """1.0 on category match, 0.3 for generic docs, 0.5 neutral, 0.0 mismatch."""
        doc_category = doc.metadata.get("category", "general")

        if not query_category:
            # No category detected in query — don't penalize any document
            return 0.5

        if doc_category == query_category:
            return 1.0

        if doc_category == "general":
            # Generic docs might still be relevant
            return 0.3

        return 0.0

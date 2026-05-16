"""
Query Expansion — Generate query variations for better semantic search coverage.

Uses fast local synonym/keyword expansion (no LLM calls).
"""

from typing import List
from models.document_processor import extract_article_number


def expand_query(
    query: str, 
    language: str = "ar",
    max_variations: int = 3,
    **kwargs,
) -> List[str]:
    """
    Generate query variations for improved semantic search.
    Uses fast local expansion (no LLM calls).
    
    Args:
        query: Original user query.
        language: Query language ("ar" or "en").
        max_variations: Maximum number of query variations to generate.
    
    Returns:
        List of expanded query variations (original included).
    """
    expanded_queries = [query]
    
    expansions = _fallback_expansions(query, language)
    
    for expansion in expansions:
        if expansion not in expanded_queries:
            expanded_queries.append(expansion)
    
    return expanded_queries[:max_variations + 1]


def _fallback_expansions(query: str, language: str) -> List[str]:
    """Fast query expansion using predefined synonym patterns."""
    
    arabic_synonyms = {
        "عقد": ["تعاقد", "اتفاقية", "ميثاق"],
        "التزام": ["مسؤولية", "واجب", "دَين"],
        "ملكية": ["تملّك", "احتياز", "حقوق"],
        "ميراث": ["إرث", "ترك", "تركة"],
        "إثبات": ["إقامة", "تأكيد", "برهنة"],
        "تعويض": ["مُقَابِل", "تعويضات", "أَجْرَة"],
    }
    
    english_synonyms = {
        "contract": ["agreement", "deal", "arrangement"],
        "obligation": ["duty", "responsibility", "liability"],
        "property": ["ownership", "possession", "assets"],
        "inheritance": ["succession", "heritage", "legacy"],
        "evidence": ["proof", "testimony", "demonstration"],
        "compensation": ["damages", "remedy", "redress"],
    }
    
    expansions = []
    
    if language == "ar":
        for term, synonyms in arabic_synonyms.items():
            if term in query:
                for synonym in synonyms:
                    expansion = query.replace(term, synonym)
                    if expansion != query:
                        expansions.append(expansion)
    else:
        for term, synonyms in english_synonyms.items():
            if term in query.lower():
                for synonym in synonyms:
                    expansion = query.lower().replace(term, synonym)
                    if expansion != query.lower():
                        expansions.append(expansion)
    
    if "مادة" in query or "article" in query.lower():
        if language == "ar":
            expansions.extend([
                f"تفاصيل {query}",
                f"شرح {query}",
                f"أحكام {query}",
            ])
        else:
            expansions.extend([
                f"details of {query}",
                f"explain {query}",
                f"provisions of {query}",
            ])
    else:
        if language == "ar":
            expansions.extend([
                f"قانون {query}",
                f"المادة المتعلقة ب {query}",
                f"ما هي أحكام {query}",
            ])
        else:
            expansions.extend([
                f"law about {query}",
                f"article related to {query}",
                f"what are the provisions of {query}",
            ])
    
    return expansions[:3]

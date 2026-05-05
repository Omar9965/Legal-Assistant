"""
Query Expansion — Generate multiple query variations for better semantic search coverage.

Supports:
- Legal domain-specific expansion
- Synonym generation
- Query decomposition
- Cross-lingual expansion
"""

from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import get_llm
from models.document_processor import extract_article_number


def expand_query(
    query: str, 
    language: str = "ar",
    max_variations: int = 3,
    include_synonyms: bool = True,
    include_decomposition: bool = True
) -> List[str]:
    """
    Generate multiple query variations for improved semantic search.
    
    Args:
        query: Original user query.
        language: Query language ("ar" or "en").
        max_variations: Maximum number of query variations to generate.
        include_synonyms: Whether to include legal synonyms.
        include_decomposition: Whether to decompose complex queries.
    
    Returns:
        List of expanded query variations.
    """
    
    # Always include the original query
    expanded_queries = [query]
    
    # Extract article number if present
    article_number = extract_article_number(query)
    
    # Generate LLM-based expansions
    if article_number:
        # If article number is present, keep it focused
        expansions = _generate_article_focused_expansions(query, article_number, language)
    else:
        # Generate comprehensive expansions
        expansions = _generate_comprehensive_expansions(query, language, max_variations)
    
    # Add expansions and remove duplicates
    for expansion in expansions:
        if expansion not in expanded_queries:
            expanded_queries.append(expansion)
    
    # Limit to max_variations + 1 (original)
    return expanded_queries[:max_variations + 1]


def _generate_article_focused_expansions(query: str, article_number: str, language: str) -> List[str]:
    """Generate expansions focused on specific article queries."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an Egyptian legal expert. The user is asking about a specific article.
        Generate 3 variations of the query that capture different aspects of asking about this article.
        Focus on:
        - Different phrasings of the same request
        - Related legal concepts
        - Alternative ways to ask about the same legal principle
        
        Return only the queries, one per line, without numbering or explanations."""),
        ("human", f"Original query: {query}\nArticle number: {article_number}\nLanguage: {language}")
    ])
    
    try:
        llm = get_llm()
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({
            "query": query, 
            "article_number": article_number,
            "language": language
        })
        
        # Parse response into individual queries
        variations = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return variations[:3]  # Limit to 3 variations
        
    except Exception as e:
        print(f"[QueryExpansion] LLM expansion failed: {e}")
        return _fallback_expansions(query, language)


def _generate_comprehensive_expansions(query: str, language: str, max_variations: int) -> List[str]:
    """Generate comprehensive query expansions including synonyms and decomposition."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Egyptian legal researcher. The user has asked a legal question.
        Generate {max_variations} variations of the query that capture different semantic aspects.
        
        For each variation, consider:
        - Synonyms and legal terminology
        - Broader and narrower terms
        - Related legal concepts
        - Alternative phrasings
        - Different angles on the same legal issue
        
        Return only the queries, one per line, without numbering or explanations."""),
        ("human", f"Original query: {query}\nLanguage: {language}")
    ])
    
    try:
        llm = get_llm()
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "language": language,
            "max_variations": max_variations
        })
        
        # Parse response into individual queries
        variations = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return variations[:max_variations]
        
    except Exception as e:
        print(f"[QueryExpansion] LLM expansion failed: {e}")
        return _fallback_expansions(query, language)


def _fallback_expansions(query: str, language: str) -> List[str]:
    """Fallback query expansion using predefined patterns."""
    
    # Legal terminology mapping for Arabic
    arabic_synonyms = {
        "عقد": ["تعاقد", "اتفاقية", "ميثاق", "تعهّد"],
        "التزام": ["مسؤولية", "واجب", "دَين"],
        "ملكية": ["تملّك", "احتياز", "حقوق"],
        "ميراث": ["إرث", "ترك", "تركة"],
        "إثبات": ["إقامة", "تأكيد", "برهنة"],
        "تعويض": ["مُقَابِل", "تعويضات", "أَجْرَة"],
    }
    
    # English synonyms
    english_synonyms = {
        "contract": ["agreement", "deal", "arrangement"],
        "obligation": ["duty", "responsibility", "liability"],
        "property": ["ownership", "possession", "assets"],
        "inheritance": ["succession", "heritage", "legacy"],
        "evidence": ["proof", "testimony", "demonstration"],
        "compensation": ["damages", "remedy", "redress"],
    }
    
    expansions = []
    
    # Add synonyms based on language
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
    
    # Add broader/narrower variations
    if "مادة" in query or "article" in query.lower():
        # Article-focused queries
        expansions.extend([
            f"تفاصيل {query}",
            f"شرح {query}",
            f"أحكام {query}",
        ])
    else:
        # General legal queries
        expansions.extend([
            f"قانون {query}",
            f"المادة المتعلقة ب {query}",
            f"ما هي أحكام {query}",
        ])
    
    return expansions[:3]  # Limit to 3 fallback expansions


def expand_with_legal_terms(query: str, language: str = "ar") -> str:
    """Expand query with relevant legal terminology."""
    
    legal_terms = {
        "ar": ["قانوني", "مدني", "عقاري", "تجاري", "جنائي", "إداري"],
        "en": ["legal", "civil", "property", "commercial", "criminal", "administrative"],
    }
    
    terms = legal_terms.get(language, [])
    
    # Add relevant legal terms to the query
    for term in terms:
        if term not in query.lower():
            # Create a version with legal terms
            expanded = f"{query} {term}"
            return expanded
    
    return query


def decompose_complex_query(query: str, language: str = "ar") -> List[str]:
    """Decompose complex queries into simpler components."""
    
    # Check if query is complex (contains multiple legal concepts)
    complex_indicators = {
        "ar": ["و", "مع", "إضافة إلى", "بالإضافة إلى", "فضلا عن"],
        "en": ["and", "with", "in addition to", "plus", "as well as"],
    }
    
    indicators = complex_indicators.get(language, [])
    
    decomposed = []
    
    for indicator in indicators:
        if indicator in query:
            # Split query at the indicator
            parts = query.split(indicator)
            for part in parts:
                part = part.strip()
                if part and len(part) > 5:  # Only add meaningful parts
                    decomposed.append(part)
    
    # Add original query as well
    decomposed.append(query)
    
    return decomposed
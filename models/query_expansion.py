"""
Query Expansion — Generate query variations for better semantic search coverage.

Strategies:
    1. LLM-based expansion (primary) — semantically rich reformulations via OpenRouter
    2. Static synonym expansion (fallback) — fast, no API call, used in retry loops
"""

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from models.document_processor import extract_article_number

logger = logging.getLogger(__name__)

# ── LLM Expansion Prompt ─────────────────────────────────────────────────────

_EXPANSION_PROMPT_AR = ChatPromptTemplate.from_messages([
    ("system", """أنت مساعد متخصص في توسيع استعلامات البحث القانوني في القانون المدني المصري.

مهمتك: أعد صياغة الاستعلام التالي بطرق مختلفة لتحسين نتائج البحث.

القواعد:
1. حافظ على المعنى القانوني الأصلي.
2. استخدم مصطلحات قانونية بديلة ومترادفات.
3. اكتب باللغة العربية فقط.
4. أنتج {max_variations} صياغات بديلة فقط.
5. اكتب كل صياغة في سطر منفصل.
6. لا تكتب أرقاماً أو شروحات، فقط الصياغات البديلة."""),
    ("human", "{query}"),
])

_EXPANSION_PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system", """You are a legal search query expansion assistant for the Egyptian Civil Code.

Your task: rephrase the following query in different ways to improve search results.

Rules:
1. Preserve the original legal intent.
2. Use alternative legal terminology and synonyms.
3. Write in English only.
4. Produce exactly {max_variations} alternative phrasings.
5. Write each phrasing on a separate line.
6. No numbering, no explanations — only the alternative queries."""),
    ("human", "{query}"),
])


# ── LLM-based Expansion ─────────────────────────────────────────────────────

def _llm_expand(query: str, language: str = "ar", max_variations: int = 3) -> List[str]:
    """
    Generate query variations using the LLM for semantically rich reformulations.

    The LLM understands Arabic morphology, legal synonyms, and contextual
    meaning far better than static synonym tables.

    Args:
        query: Original user query.
        language: Query language ("ar" or "en").
        max_variations: Number of variations to generate.

    Returns:
        List of expanded queries (may be empty if LLM fails).
    """
    from utils.config import get_llm

    try:
        llm = get_llm()
        prompt = _EXPANSION_PROMPT_AR if language == "ar" else _EXPANSION_PROMPT_EN
        chain = prompt | llm

        response = chain.invoke({
            "query": query,
            "max_variations": max_variations,
        })

        # Parse LLM output: one query per line
        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]

        # Filter out empty, too-short, or duplicate-of-original results
        expansions = [
            line for line in lines
            if len(line) > 5 and line != query
        ]

        logger.info(f"[QueryExpansion] LLM generated {len(expansions)} variations")
        return expansions[:max_variations]

    except Exception as e:
        logger.warning(f"[QueryExpansion] LLM expansion failed: {e}")
        return []


# ── Public API ───────────────────────────────────────────────────────────────

def expand_query(
    query: str,
    language: str = "ar",
    max_variations: int = 3,
    **kwargs,
) -> List[str]:
    """
    Generate query variations for improved semantic search.

    Primary: LLM-based expansion (semantically rich, handles Arabic morphology).
    Fallback: Static synonym expansion (if LLM fails or is unavailable).

    Args:
        query: Original user query.
        language: Query language ("ar" or "en").
        max_variations: Maximum number of additional query variations.

    Returns:
        List of query variations (original query always included first).
    """
    expanded_queries = [query]

    # Primary: LLM-based expansion
    llm_expansions = _llm_expand(query, language, max_variations)

    if llm_expansions:
        for exp in llm_expansions:
            if exp not in expanded_queries:
                expanded_queries.append(exp)
    else:
        # Fallback: static synonym expansion
        for exp in _fallback_expansions(query, language):
            if exp not in expanded_queries:
                expanded_queries.append(exp)

    return expanded_queries[:max_variations + 1]


# ── Static Fallback (used by reformulate_node for fast retry) ────────────────

def _fallback_expansions(query: str, language: str) -> List[str]:
    """Fast query expansion using predefined synonym patterns.

    Intentionally avoids LLM calls — used in the retry loop
    (reformulate_node) where speed is critical (~0ms vs ~5-15s).
    """

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

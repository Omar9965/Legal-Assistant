"""
Semantic Cache — Stores past (query → answer) pairs as vectors for fast retrieval.

Uses a separate ChromaDB collection to avoid polluting the legal document collections.
"""

import logging
import uuid
import chromadb
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
from utils.config import (
    CHROMA_DB_PATH,
    SEMANTIC_CACHE_COLLECTION,
    SIMILARITY_THRESHOLD,
    get_embedding_function,
)
from langchain_core.documents import Document
from models.document_processor import extract_article_number

_cache_collection_cache: Chroma | None = None


def _enrich_query(query: str) -> str:
    """
    Append extracted article number as a tag so embeddings for different
    article numbers diverge more in vector space.
    """
    article_num = extract_article_number(query)
    if article_num:
        return f"{query} [article:{article_num}]"
    return query


def _get_cache_collection() -> Chroma:
    """Return the semantic_cache Chroma collection (cached)."""
    global _cache_collection_cache
    if _cache_collection_cache is None:
        _cache_collection_cache = Chroma(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            embedding_function=get_embedding_function(),
            persist_directory=CHROMA_DB_PATH,
        )
    return _cache_collection_cache


def lookup(query: str) -> tuple[str | None, float]:
    """
    Check the semantic cache for a similar past query.

    Args:
        query: The user's input query.

    Returns:
        Tuple of (cached_answer, similarity_score).
        If no match above threshold, returns (None, 0.0).
    """
    collection = _get_cache_collection()

    try:
        results = collection.similarity_search_with_score(_enrich_query(query), k=1)
    except Exception as e:
        logger.warning(f"[SemanticCache] Embedding lookup failed: {e}")
        return None, 0.0

    if not results:
        return None, 0.0

    doc, dist = results[0]
    dist = max(0.0, dist)
    similarity = max(0.0, 1.0 - (dist / 2.0))

    if similarity >= SIMILARITY_THRESHOLD:
        # reject if article numbers differ
        incoming_article = extract_article_number(query)
        cached_article   = extract_article_number(doc.metadata.get("query", ""))
        if incoming_article and cached_article and incoming_article != cached_article:
            return None, 0.0

        cached_answer = doc.metadata.get("answer", "")
        return cached_answer, similarity

    return None, similarity


def store(query: str, answer: str, metadata: dict = None) -> None:
    """
    Store a query-answer pair in the semantic cache.

    The enriched query is stored as document content for a better embedding,
    and the original query + answer are preserved in metadata for retrieval
    and the article number guard.

    Args:
        query: The user's original query.
        answer: The generated answer to cache.
        metadata: Additional metadata (e.g., strategy used, confidence).
    """
    collection = _get_cache_collection()

    # Build metadata
    doc_metadata = {
        "answer": answer,
        "query": query,             
    }
    
   
    if metadata:
        doc_metadata["strategy"] = metadata.get("strategy", "unknown")
        doc_metadata["router_confidence"] = metadata.get("router_confidence", 0.5)
        doc_metadata["retrieval_attempts"] = metadata.get("retrieval_attempts", 1)

    doc = Document(
        page_content=_enrich_query(query),  # Fix 3: richer embedding
        metadata=doc_metadata,
    )
    collection.add_documents([doc], ids=[str(uuid.uuid4())])


def clear_cache() -> None:
    """Clear all entries in the semantic cache."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(SEMANTIC_CACHE_COLLECTION)
        print("[SemanticCache] Cache cleared.")
    except Exception:
        print("[SemanticCache] Cache was already empty.")
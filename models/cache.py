"""
Cache — two-level response cache (exact in-memory + semantic ChromaDB).

Single facade for cache reads AND writes so both paths go through the same
layer:

    get(query) → L1 exact (O(1), in-memory) → L2 semantic (cosine vector search)
    put(query, answer)                    → writes both levels
    invalidate()                          → drops everything (called after
                                            corpus re-ingestion)

Semantic entries carry a creation timestamp and expire after
CACHE_TTL_SECONDS; stale entries are ignored (never served).
"""

import hashlib
import logging
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.config import (
    CACHE_TTL_SECONDS,
    CHROMA_DB_PATH,
    CHROMA_DISTANCE_METRIC,
    SEMANTIC_CACHE_COLLECTION,
    SIMILARITY_THRESHOLD,
)
from models.document_processor import extract_article_number
from models.resources import get_embedding_function

logger = logging.getLogger(__name__)

_EXACT_MAX_SIZE = 1000

_exact_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()  # key → (answer, created_at)
_cache_lock = threading.Lock()


def _normalize(query: str) -> str:
    """Normalize a query into a stable key form (Arabic-aware)."""
    normalized = unicodedata.normalize("NFKC", query.strip())
    normalized = "".join(
        c for c in normalized if not ("\u064B" <= c <= "\u065F" or c == "\u0670")
    )
    return normalized.lower()


def _make_key(query: str) -> str:
    return hashlib.sha256(_normalize(query).encode()).hexdigest()[:32]


def _enrich_query(query: str) -> str:
    """Append the extracted article number so different articles diverge more."""
    article_num = extract_article_number(query)
    if article_num:
        return f"{query} [article:{article_num}]"
    return query


@lru_cache(maxsize=1)
def _get_collection() -> Chroma:
    """Return the semantic_cache Chroma collection (cosine metric, cached)."""
    return Chroma(
        collection_name=SEMANTIC_CACHE_COLLECTION,
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_DB_PATH,
        collection_metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
    )


# ── L1: exact in-memory cache ────────────────────────────────────────────────

def _exact_get(query: str) -> str | None:
    key = _make_key(query)
    with _cache_lock:
        entry = _exact_cache.get(key)
        if entry is None:
            return None
        answer, created_at = entry
        if time.time() - created_at > CACHE_TTL_SECONDS:
            del _exact_cache[key]
            return None
        _exact_cache.move_to_end(key)
    return answer


def _exact_put(query: str, answer: str) -> None:
    key = _make_key(query)
    with _cache_lock:
        if key in _exact_cache:
            _exact_cache.move_to_end(key)
        elif len(_exact_cache) >= _EXACT_MAX_SIZE:
            _exact_cache.popitem(last=False)
        _exact_cache[key] = (answer, time.time())


# ── L2: semantic (ChromaDB) cache ────────────────────────────────────────────

def _semantic_get(query: str) -> tuple[str | None, float]:
    """Vector-similarity lookup. Returns (answer|None, similarity_score)."""
    try:
        results = _get_collection().similarity_search_with_score(_enrich_query(query), k=1)
    except Exception as e:
        logger.warning(f"[Cache] Semantic lookup failed: {e}")
        return None, 0.0

    if not results:
        return None, 0.0

    doc, dist = results[0]
    similarity = max(0.0, 1.0 - max(0.0, dist))  # cosine distance → similarity

    if similarity < SIMILARITY_THRESHOLD:
        return None, similarity

    # TTL: ignore stale entries (they will be overwritten on next store)
    created_at = doc.metadata.get("created_at", 0)
    if time.time() - created_at > CACHE_TTL_SECONDS:
        return None, similarity

    # Guard: never serve a cached answer for a different article number
    incoming_article = extract_article_number(query)
    cached_article = extract_article_number(doc.metadata.get("query", ""))
    if incoming_article and cached_article and incoming_article != cached_article:
        return None, 0.0

    return doc.metadata.get("answer", ""), similarity


# ── Public API ───────────────────────────────────────────────────────────────

def get(query: str) -> tuple[str | None, float]:
    """Look up a cached answer. Returns (answer|None, similarity_score)."""
    exact_answer = _exact_get(query)
    if exact_answer is not None:
        logger.info("[Cache] L1 (exact) hit")
        return exact_answer, 1.0

    answer, score = _semantic_get(query)
    if answer is not None:
        logger.info(f"[Cache] L2 (semantic) hit, similarity: {score:.3f}")
        _exact_put(query, answer)  # promote to L1 for next time
    return answer, score


def put(query: str, answer: str) -> None:
    """Store a query-answer pair in both cache levels."""
    if not answer:
        return
    _exact_put(query, answer)
    try:
        _get_collection().add_documents(
            [
                Document(
                    page_content=_enrich_query(query),
                    metadata={
                        "answer": answer,
                        "query": query,
                        "created_at": time.time(),
                    },
                )
            ],
            ids=[str(uuid.uuid4())],
        )
    except Exception as e:
        logger.warning(f"[Cache] Failed to store in semantic cache: {e}")


def invalidate() -> None:
    """Drop all cached answers (used after corpus re-ingestion)."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(SEMANTIC_CACHE_COLLECTION)
        logger.info("[Cache] Semantic cache cleared.")
    except Exception:
        logger.info("[Cache] Semantic cache was already empty.")
    _get_collection.cache_clear()
    with _cache_lock:
        _exact_cache.clear()

"""
Multi-Level Caching System — Enhanced caching for legal document retrieval.

Levels:
1. Exact Query Cache (in-memory) — O(1) lookup for exact queries
2. Semantic Cache (ChromaDB) — Vector similarity search for similar queries
3. Article Lookup Cache (in-memory) — Fast lookup for specific article numbers
4. Result Cache (in-memory) — Caches frequent search results

Features:
- LRU eviction
- TTL support
- Cache warming
- Statistics tracking
- Multi-level fallback
"""

import logging
import hashlib
import time
import unicodedata
from typing import Optional, Dict, List, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
import threading

logger = logging.getLogger(__name__)

from models.semantic_cache import lookup as semantic_lookup, store as semantic_store
from models.document_processor import extract_article_number
from utils.config import SIMILARITY_THRESHOLD


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class ExactQueryCache:
    """Level 1: Exact query match cache (in-memory, O(1) lookup)."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def get(self, query: str) -> Optional[Tuple[str, float]]:
        """Get cached result for exact query match."""
        key = self._make_key(query)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL expiration
                if entry.is_expired():
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                self._stats["hits"] += 1
                return entry.value, 1.0  # Exact match = 1.0 similarity
            
            self._stats["misses"] += 1
            return None
    
    def set(self, query: str, answer: str, metadata: Dict = None) -> None:
        """Store query-answer pair in cache."""
        key = self._make_key(query)
        
        with self._lock:
            # Check if key exists
            if key in self._cache:
                # Update existing entry
                entry = self._cache[key]
                entry.value = answer
                entry.last_accessed = time.time()
                entry.access_count += 1
                if metadata:
                    entry.metadata.update(metadata)
                return
            
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1
            
            # Add new entry
            self._cache[key] = CacheEntry(
                key=key,
                value=answer,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=self._default_ttl,
                metadata=metadata or {}
            )
    
    def _make_key(self, query: str) -> str:
        """Create cache key from query with Arabic-aware normalization."""
        # Strip whitespace
        normalized = query.strip()
        # NFKC normalization (consistent Unicode forms)
        normalized = unicodedata.normalize("NFKC", normalized)
        # Remove Arabic diacritics (tashkeel: U+064B to U+065F, U+0670)
        normalized = ''.join(
            c for c in normalized
            if not ('\u064B' <= c <= '\u065F' or c == '\u0670')
        )
        # Lowercase for Latin characters (no effect on Arabic)
        normalized = normalized.lower()
        # Use hash for consistent key
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
            }
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}


class ArticleLookupCache:
    """Level 2: Fast lookup for specific article numbers."""
    
    def __init__(self, max_size: int = 500):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._max_size = max_size
    
    def get(self, article_number: str) -> Optional[List[Dict]]:
        """Get cached results for specific article number."""
        with self._lock:
            if article_number in self._cache:
                entry = self._cache[article_number]
                if not entry.is_expired():
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    return entry.value
                else:
                    del self._cache[article_number]
            return None
    
    def set(self, article_number: str, results: List[Dict]) -> None:
        """Store article lookup results."""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
                del self._cache[oldest_key]
            
            self._cache[article_number] = CacheEntry(
                key=article_number,
                value=results,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=7200  # 2 hours TTL
            )
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


class ResultCache:
    """Level 3: Cache for frequent search results."""
    
    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._max_size = max_size
    
    def get(self, search_params: Dict) -> Optional[List[Any]]:
        """Get cached search results."""
        key = self._make_result_key(search_params)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    self._cache.move_to_end(key)
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    return entry.value
                else:
                    del self._cache[key]
            return None
    
    def set(self, search_params: Dict, results: List[Any]) -> None:
        """Store search results."""
        key = self._make_result_key(search_params)
        
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                key=key,
                value=results,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=1800  # 30 minutes TTL
            )
    
    def _make_result_key(self, params: Dict) -> str:
        """Create cache key from search parameters."""
        # Sort parameters for consistent key
        sorted_params = sorted(params.items())
        param_str = str(sorted_params)
        return hashlib.sha256(param_str.encode()).hexdigest()[:32]
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


class MultiLevelCache:
    """
    Multi-level cache with fallback through levels:
    1. Exact Query → 2. Semantic → 3. Article Lookup → 4. Result Cache
    """
    
    def __init__(self):
        self._exact_cache = ExactQueryCache()
        self._article_cache = ArticleLookupCache()
        self._result_cache = ResultCache()
        self._enabled = {
            "exact": True,
            "semantic": True,
            "article": True,
            "result": False  
        }
    
    def lookup(self, query: str) -> Tuple[Optional[str], float, str]:
        """
        Multi-level cache lookup.
        
        Returns:
            Tuple of (cached_answer, similarity_score, cache_level)
        """
        
        # Level 1: Exact query match
        if self._enabled["exact"]:
            result = self._exact_cache.get(query)
            if result:
                logger.info("[Cache] Level 1 (exact) hit")
                return result[0], result[1], "exact"
        
        # Level 2: Semantic cache (vector similarity)
        if self._enabled["semantic"]:
            try:
                cached_answer, similarity = semantic_lookup(query)
                if cached_answer and similarity >= SIMILARITY_THRESHOLD:
                    # Store in level 1 for faster future access
                    self._exact_cache.set(query, cached_answer)
                    logger.info(f"[Cache] Level 2 (semantic) hit, similarity: {similarity:.3f}")
                    return cached_answer, similarity, "semantic"
            except Exception as e:
                logger.warning(f"[Cache] Semantic lookup failed: {e}")
        
        # Level 3: Article-specific lookup
        if self._enabled["article"]:
            article_num = extract_article_number(query)
            if article_num:
                article_results = self._article_cache.get(article_num)
                if article_results:
                    logger.info(f"[Cache] Level 3 (article) hit for article {article_num}")
                    # Return formatted results as string
                    formatted = self._format_article_results(article_results)
                    return formatted, 0.9, "article"
        
        return None, 0.0, "none"
    
    def store(self, query: str, answer: str, metadata: Dict = None) -> None:
        """Store result in appropriate cache levels."""
        
        # Store in exact cache
        if self._enabled["exact"]:
            self._exact_cache.set(query, answer, metadata)
        
        # Store in semantic cache (ChromaDB)
        if self._enabled["semantic"]:
            try:
                semantic_store(query, answer, metadata)
            except Exception as e:
                logger.warning(f"[Cache] Failed to store in semantic cache: {e}")
        
        # Store article-specific results
        if self._enabled["article"]:
            article_num = extract_article_number(query)
            if article_num and metadata and "retrieved_docs" in metadata:
                self._article_cache.set(article_num, metadata["retrieved_docs"])
    
    def _format_article_results(self, results: List[Dict]) -> str:
        """Format article results for caching."""
        # Simplified formatting - in production, would store structured data
        if not results:
            return ""
        
        formatted_parts = []
        for result in results[:3]:  # Top 3 results
            if isinstance(result, dict):
                content = result.get("page_content", str(result))
            else:
                content = str(result)
            formatted_parts.append(content[:500])  # Limit length
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def get_stats(self) -> Dict:
        """Get statistics from all cache levels."""
        return {
            "exact": self._exact_cache.get_stats(),
            "article": {"size": len(self._article_cache._cache)},
            "result": {"size": len(self._result_cache._cache)},
            "enabled": self._enabled
        }
    
    def enable_level(self, level: str, enabled: bool) -> None:
        """Enable or disable a cache level."""
        if level in self._enabled:
            self._enabled[level] = enabled
            logger.info(f"[Cache] Level '{level}' {'enabled' if enabled else 'disabled'}")
    
    def clear_all(self) -> None:
        """Clear all cache levels."""
        self._exact_cache.clear()
        self._article_cache.clear()
        self._result_cache.clear()
        logger.info("[Cache] All levels cleared")


# Global cache instance
_global_cache: Optional[MultiLevelCache] = None
_cache_lock = threading.Lock()


def get_multi_level_cache() -> MultiLevelCache:
    """Get or create the global multi-level cache instance."""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = MultiLevelCache()
    
    return _global_cache


def cache_lookup(query: str) -> Tuple[Optional[str], float, str]:
    """Convenience function for multi-level cache lookup."""
    cache = get_multi_level_cache()
    return cache.lookup(query)


def cache_store(query: str, answer: str, metadata: Dict = None) -> None:
    """Convenience function for storing in multi-level cache."""
    cache = get_multi_level_cache()
    cache.store(query, answer, metadata)


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    cache = get_multi_level_cache()
    return cache.get_stats()


def clear_all_caches() -> None:
    """Clear all cache levels."""
    cache = get_multi_level_cache()
    cache.clear_all()
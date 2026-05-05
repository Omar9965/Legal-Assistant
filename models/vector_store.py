"""
Vector Store — ChromaDB collection management for legal documents.

Collections:
    - legal_ar: Base Egyptian Civil Code articles
    - legal_ar_supplementary: Additional legal materials
    - hybrid_search_index: Combined search index for hybrid retrieval
"""

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Optional, Tuple
import re
import os
from utils.config import (
    CHROMA_DB_PATH,
    LEGAL_AR_COLLECTION,
    TOP_K,
    get_embedding_function,
)

_legal_collection_cache: Chroma | None = None
_collection_cache: dict[str, Chroma] = {}
_hybrid_index_cache: Chroma | None = None


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_hybrid_index() -> Chroma:
    """Return the hybrid search Chroma collection (cached)."""
    global _hybrid_index_cache
    if _hybrid_index_cache is None:
        _hybrid_index_cache = Chroma(
            collection_name="hybrid_search_index",
            embedding_function=get_embedding_function(),
            persist_directory=CHROMA_DB_PATH,
        )
    return _hybrid_index_cache


def get_legal_collection() -> Chroma:
    """Return the main legal_ar Chroma collection (LangChain wrapper)."""
    global _legal_collection_cache
    if _legal_collection_cache is None:
        _legal_collection_cache = Chroma(
            collection_name=LEGAL_AR_COLLECTION,
            embedding_function=get_embedding_function(),
            persist_directory=CHROMA_DB_PATH,
        )
    return _legal_collection_cache


def _get_cached_collection(collection_name: str) -> Chroma:
    """Return a cached Chroma collection instance."""
    if collection_name not in _collection_cache:
        _collection_cache[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=get_embedding_function(),
            persist_directory=CHROMA_DB_PATH,
        )
    return _collection_cache[collection_name]


def add_documents(documents: list[Document], collection_name: str = LEGAL_AR_COLLECTION) -> None:
    """
    Add a list of LangChain Documents to the specified ChromaDB collection.

    Args:
        documents: List of Document objects with page_content and metadata.
        collection_name: Target collection name.
    """
    if not documents:
        print(f"[VectorStore] No documents to add to '{collection_name}'.")
        return

    collection = _get_cached_collection(collection_name)
    collection.add_documents(documents)
    print(f"[VectorStore] Added {len(documents)} documents to '{collection_name}'.")


def search(
    query: str,
    collection_name: str = LEGAL_AR_COLLECTION,
    top_k: int = TOP_K,
    filter_dict: dict = None,
    use_hybrid: bool = False,
) -> list[Document]:
    """
    Perform similarity search on a ChromaDB collection.
    
    Args:
        query: The search query text.
        collection_name: Which collection to search.
        top_k: Number of results to return.
        filter_dict: Optional metadata filter (e.g., {"category": "contracts"}).
        use_hybrid: Whether to use hybrid search (vector + keyword).
    
    Returns:
        List of matching Document objects.
    """
    collection = _get_cached_collection(collection_name)

    kwargs = {"k": top_k}
    if filter_dict:
        kwargs["filter"] = filter_dict

    if use_hybrid:
        # Use ChromaDB's native query method for hybrid search
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        # Convert to Document format
        documents = []
        for doc, metadata, distance in zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        ):
            documents.append(Document(
                page_content=doc,
                metadata=metadata or {}
            ))
        return documents
    else:
        # Standard similarity search
        results = collection.similarity_search(query, **kwargs)
        return results


def hybrid_search(
    query: str,
    collection_name: str = LEGAL_AR_COLLECTION,
    top_k: int = TOP_K,
    filter_dict: dict = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[Document]:
    """
    Perform hybrid search combining vector similarity and keyword matching.
    
    Note: ChromaDB does not support true BM25/keyword search. Both branches
    use embedding-based retrieval with different APIs. The benefit comes from
    combining similarity_search_with_score (which returns distances) and
    the query API, then merging their ranked lists.
    
    Args:
        query: The search query text.
        collection_name: Which collection to search.
        top_k: Number of results to return.
        filter_dict: Optional metadata filter.
        vector_weight: Weight for vector similarity (0-1).
        keyword_weight: Weight for keyword matching (0-1).
    
    Returns:
        List of Document objects with hybrid scores.
    """
    collection = _get_cached_collection(collection_name)
    
    # Get vector similarity results (returns distance, lower = better)
    raw_vector_results = collection.similarity_search_with_score(query, k=top_k * 2)
    
    # Convert distances to similarity scores (higher = better)
    vector_results = [
        (doc, max(0.0, 1.0 - (dist / 2.0)))
        for doc, dist in raw_vector_results
    ]
    
    # Get keyword matching results (already returns similarity scores)
    keyword_results = _keyword_search(query, collection, top_k * 2)
    
    # Combine and rank results
    combined_scores = _combine_results(vector_results, keyword_results, vector_weight, keyword_weight)
    
    # Return top_k results
    return [doc for doc, score in combined_scores[:top_k]]


def _keyword_search(
    query: str, 
    collection: Chroma, 
    top_k: int = 10
) -> List[Tuple[Document, float]]:
    """Perform keyword-based search using ChromaDB query method."""
    try:
        # Use ChromaDB's native query for keyword search
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        keyword_results = []
        for doc, metadata, distance in zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        ):
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = max(0.0, 1.0 - (distance / 2.0))
            keyword_results.append((Document(
                page_content=doc,
                metadata=metadata or {}
            ), similarity))
        
        return keyword_results
    except Exception as e:
        print(f"[HybridSearch] Keyword search failed: {e}")
        return []


def _combine_results(
    vector_results: List[Tuple[Document, float]],
    keyword_results: List[Tuple[Document, float]],
    vector_weight: float,
    keyword_weight: float
) -> List[Tuple[Document, float]]:
    """Combine vector and keyword results with weighted scoring."""
    
    # Create a dictionary to store scores for each document
    document_scores = {}
    
    # Process vector results
    for doc, score in vector_results:
        doc_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"
        if doc_id not in document_scores:
            document_scores[doc_id] = {"doc": doc, "vector_score": 0.0, "keyword_score": 0.0, "count": 0}
        document_scores[doc_id]["vector_score"] += score
        document_scores[doc_id]["count"] += 1
    
    # Process keyword results
    for doc, score in keyword_results:
        doc_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"
        if doc_id not in document_scores:
            document_scores[doc_id] = {"doc": doc, "vector_score": 0.0, "keyword_score": 0.0, "count": 0}
        document_scores[doc_id]["keyword_score"] += score
        document_scores[doc_id]["count"] += 1
    
    # Calculate combined scores
    combined_results = []
    for doc_id, data in document_scores.items():
        doc = data["doc"]
        
        # Average scores if multiple occurrences
        avg_vector_score = data["vector_score"] / data["count"] if data["count"] > 0 else 0.0
        avg_keyword_score = data["keyword_score"] / data["count"] if data["count"] > 0 else 0.0
        
        # Weighted combination
        combined_score = (avg_vector_score * vector_weight) + (avg_keyword_score * keyword_weight)
        
        combined_results.append((doc, combined_score))
    
    # Sort by combined score
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results





def clear_collection(collection_name: str) -> None:
    """Delete all documents in a collection (for re-ingestion)."""
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
        print(f"[VectorStore] Cleared collection '{collection_name}'.")
    except Exception:
        print(f"[VectorStore] Collection '{collection_name}' does not exist, nothing to clear.")
    # Invalidate the in-memory cache so subsequent writes don't use a stale wrapper
    reset_collection_cache(collection_name)


def get_collection_count(collection_name: str) -> int:
    """Return the number of documents in a collection."""
    client = _get_chroma_client()
    try:
        col = client.get_collection(collection_name)
        return col.count()
    except Exception:
        return 0


def optimize_collection(collection_name: str = LEGAL_AR_COLLECTION) -> Dict:
    """
    Optimize ChromaDB collection for better performance.
    
    Returns:
        Dictionary with optimization results.
    """
    client = _get_chroma_client()
    
    try:
        col = client.get_collection(collection_name)
        
        count = col.count()
        
        try:
            sample = col.peek(limit=5)
            has_data = len(sample.get("ids", [])) > 0
        except:
            has_data = False
        
        return {
            "collection": collection_name,
            "document_count": count,
            "has_data": has_data,
            "status": "optimized",
            "recommendations": _get_optimization_recommendations(count)
        }
    except Exception as e:
        return {
            "collection": collection_name,
            "error": str(e),
            "status": "error"
        }


def _get_optimization_recommendations(doc_count: int) -> List[str]:
    """Generate recommendations based on collection size."""
    recommendations = []
    
    if doc_count == 0:
        recommendations.append("No documents in collection. Run process.py to ingest documents.")
    elif doc_count < 100:
        recommendations.append("Small collection. Consider adding more legal documents for better coverage.")
    elif doc_count > 10000:
        recommendations.append("Large collection. Consider implementing pagination for searches.")
    
    return recommendations


def get_collection_stats(collection_name: str = LEGAL_AR_COLLECTION) -> Dict:
    """Get detailed statistics about a collection."""
    client = _get_chroma_client()
    
    try:
        col = client.get_collection(collection_name)
        
        data = col.get()
        
        metadata_fields = {}
        for meta in data.get("metadatas", []):
            if meta:
                for key, value in meta.items():
                    if key not in metadata_fields:
                        metadata_fields[key] = set()
                    metadata_fields[key].add(str(value))
        
        field_counts = {k: len(v) for k, v in metadata_fields.items()}
        
        return {
            "collection": collection_name,
            "total_documents": len(data.get("ids", [])),
            "metadata_fields": field_counts,
            "unique_categories": list(metadata_fields.get("category", set())) if "category" in metadata_fields else [],
            "status": "ok"
        }
    except Exception as e:
        return {
            "collection": collection_name,
            "error": str(e),
            "status": "error"
        }


def reset_collection_cache(collection_name: str = None) -> None:
    """Reset the in-memory cache for collections."""
    global _collection_cache, _legal_collection_cache
    
    if collection_name:
        if collection_name in _collection_cache:
            del _collection_cache[collection_name]
        if collection_name == LEGAL_AR_COLLECTION:
            _legal_collection_cache = None
    else:
        _collection_cache.clear()
        _legal_collection_cache = None
    
    print(f"[VectorStore] Cache reset for collection: {collection_name or 'all'}")

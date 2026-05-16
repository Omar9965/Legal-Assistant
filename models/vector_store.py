"""
Vector Store — ChromaDB collection management for legal documents.

Collections:
    - legal_ar: Base Egyptian Civil Code articles
"""

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config import (
    CHROMA_DB_PATH,
    LEGAL_AR_COLLECTION,
    TOP_K,
    get_embedding_function,
)

_collection_cache: dict[str, Chroma] = {}


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


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
) -> list[Document]:
    """
    Perform similarity search on a ChromaDB collection.
    
    Args:
        query: The search query text.
        collection_name: Which collection to search.
        top_k: Number of results to return.
        filter_dict: Optional metadata filter (e.g., {"category": "contracts"}).
    
    Returns:
        List of matching Document objects.
    """
    collection = _get_cached_collection(collection_name)

    kwargs = {"k": top_k}
    if filter_dict:
        kwargs["filter"] = filter_dict

    results = collection.similarity_search(query, **kwargs)
    return results


def delete_documents_by_source(source: str, collection_name: str = LEGAL_AR_COLLECTION) -> None:
    """
    Delete all documents with the given source filename from a collection.

    Used for incremental re-ingestion: removes only one PDF's chunks
    instead of clearing the entire collection.

    Args:
        source: The source filename to match (e.g., "law-131-1948.pdf").
        collection_name: Target collection name.
    """
    client = _get_chroma_client()
    try:
        col = client.get_collection(collection_name)
        col.delete(where={"source": source})
        print(f"[VectorStore] Deleted documents with source='{source}' from '{collection_name}'.")
    except Exception as e:
        print(f"[VectorStore] Could not delete by source '{source}': {e}")
    # Invalidate cache so subsequent writes use a fresh wrapper
    reset_collection_cache(collection_name)


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


def reset_collection_cache(collection_name: str = None) -> None:
    """Reset the in-memory cache for collections."""
    global _collection_cache
    
    if collection_name:
        if collection_name in _collection_cache:
            del _collection_cache[collection_name]
    else:
        _collection_cache.clear()
    
    print(f"[VectorStore] Cache reset for collection: {collection_name or 'all'}")

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


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_legal_collection() -> Chroma:
    """Return the main legal_ar Chroma collection (LangChain wrapper)."""
    return Chroma(
        collection_name=LEGAL_AR_COLLECTION,
        embedding_function=get_embedding_function(),
persist_directory=CHROMA_DB_PATH,
    )


def add_documents(documents: list[Document], collection_name: str = LEGAL_AR_COLLECTION) -> None:
    """
    Add a list of LangChain Documents to the specified ChromaDB collection.

    The Chroma collection (and its embedding model) is instantiated once.
    No internal batching is performed — the caller controls batch size.

    Args:
        documents: List of Document objects with page_content and metadata.
        collection_name: Target collection name.
    """
    if not documents:
        print(f"[VectorStore] No documents to add to '{collection_name}'.")
        return

    collection = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_DB_PATH,
    )

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
    collection = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_DB_PATH,
    )

    kwargs = {"k": top_k}
    if filter_dict:
        kwargs["filter"] = filter_dict

    results = collection.similarity_search(query, **kwargs)
    return results





def clear_collection(collection_name: str) -> None:
    """Delete all documents in a collection (for re-ingestion)."""
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
        print(f"[VectorStore] Cleared collection '{collection_name}'.")
    except Exception:
        print(f"[VectorStore] Collection '{collection_name}' does not exist, nothing to clear.")


def get_collection_count(collection_name: str) -> int:
    """Return the number of documents in a collection."""
    client = _get_chroma_client()
    try:
        col = client.get_collection(collection_name)
        return col.count()
    except Exception:
        return 0

"""
Resource factories — cached singletons for the embedding model and the LLM.

Uses functools.lru_cache so instances are shared across the whole process
(including Streamlit reruns) without coupling the config layer to any UI
framework. Heavy imports are deferred into the factory bodies so that mere
config access (e.g. the ingestion script) does not pull in model libraries.
"""

from functools import lru_cache

from utils.config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    API_KEY,
)


@lru_cache(maxsize=1)
def get_embedding_function():
    """Return the HuggingFace embedding model (multilingual, L2-normalized)."""
    if not EMBEDDING_MODEL_NAME:
        raise RuntimeError(
            "EMBEDDING_MODEL is not set. Define it in your .env file "
            "(see .env.example)."
        )
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_llm():
    """Return the LLM instance (any OpenAI-compatible endpoint)."""
    if not API_KEY:
        raise RuntimeError(
            "API_KEY is not set. Define it in your .env file "
            "(see .env.example)."
        )
    if not LLM_BASE_URL:
        raise RuntimeError(
            "LLM_BASE_URL is not set. Define it in your .env file "
            "(see .env.example)."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0.4,
        api_key=API_KEY,
        base_url=LLM_BASE_URL,
    )

"""
Configuration module — loads environment variables and initializes shared resources.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── API Keys & Model Names ────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_raw_chroma = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "data", "chroma_db"))
CHROMA_DB_PATH = _raw_chroma if os.path.isabs(_raw_chroma) else os.path.join(BASE_DIR, _raw_chroma)

_raw_checkpoint = os.getenv("CHECKPOINT_DB_PATH", os.path.join(BASE_DIR, "data", "checkpoints.db"))
CHECKPOINT_DB_PATH = _raw_checkpoint if os.path.isabs(_raw_checkpoint) else os.path.join(BASE_DIR, _raw_checkpoint)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PDF_DIR = os.path.join(BASE_DIR, "PDF")

# ── Retrieval Settings ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── Collection Names ──────────────────────────────────────────────────────────
LEGAL_AR_COLLECTION = "legal_ar"
SEMANTIC_CACHE_COLLECTION = "semantic_cache"

# ── Ensure data directories exist ─────────────────────────────────────────────
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_embedding_function():
    """Return HuggingFace embedding (multilingual)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    """Return OpenRouter LLM instance."""
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=0.4,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    return llm

    
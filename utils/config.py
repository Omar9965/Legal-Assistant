"""
Configuration module — pure environment parsing and path setup.

Framework-agnostic on purpose: importing this module must stay cheap and
side-effect free (no Streamlit, no model imports). Cached model factories
live in models/resources.py.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── API Keys & Model Names ────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(raw: str) -> str:
    """Resolve a possibly-relative path against the project root."""
    return raw if os.path.isabs(raw) else os.path.join(BASE_DIR, raw)


CHROMA_DB_PATH = _resolve_path(os.getenv("CHROMA_DB_PATH", os.path.join("data", "chroma_db")))
PROCESSED_DIR = _resolve_path(os.getenv("PROCESSED_DIR", os.path.join("data", "processed")))
PDF_DIR = _resolve_path(os.getenv("PDF_DIR", "PDF"))

# ── Retrieval Settings ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── Cache Settings ────────────────────────────────────────────────────────────
# Confidence level (0-1 keyword-count based) below which the router result is
# considered too uncertain to trust the response cache. Distinct from
# SIMILARITY_THRESHOLD, which is a cosine similarity (0-1) for cache matching.
CACHE_ROUTE_CONFIDENCE = float(os.getenv("CACHE_ROUTE_CONFIDENCE", "0.70"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 3600)))

# ── Collection Names & Metric ─────────────────────────────────────────────────
LEGAL_AR_COLLECTION = "legal_ar"
SEMANTIC_CACHE_COLLECTION = "semantic_cache"
# Distance metric for all Chroma collections. Embeddings are L2-normalized, so
# cosine distance = 1 - cosine_similarity and similarity = 1 - distance.
CHROMA_DISTANCE_METRIC = "cosine"

# ── Ensure data directories exist ─────────────────────────────────────────────
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


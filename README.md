# ⚖️ Legal AI Assistant: Egyptian Civil Code

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/LangGraph-Agentic-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![VectorStore](https://img.shields.io/badge/VectorStore-ChromaDB-blue)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20Compatible-blue)

An advanced, agentic AI legal assistant specialized in the **Egyptian Civil Code**. Built using the **LangGraph framework**, this project employs a highly-optimized, multi-agent architecture to classify queries, perform deep semantic research, cache responses, and generate structured bilingual answers.

The system is designed with a strict **MVC (Model-View-Controller)** structure, making it highly modular and extensible. The agents adhere to an Object-Oriented paradigm following SOLID principles, ensuring scalability and ease of maintenance.

---

## ✨ Key Features & Capabilities

- **Agentic Workflow Engine**: Orchestrated by **LangGraph**, the system acts as a state machine that dynamically coordinates multiple specialized agents. The graph is lazily compiled and cached to ensure optimal execution speeds.
- **Ultra-Fast Heuristic Routing**: Employs an ultra-fast, regex-based keyword matcher and language detector to route queries in sub-millisecond time, bypassing expensive LLM classification calls.
- **Bilingual Interface**: Seamlessly detects whether the user is speaking Arabic or English and operates fluently in both, adapting both search mechanisms and synthesis language dynamically.
- **RAG & Vector Retrieval**: Employs **ChromaDB** with advanced sentence transformers to securely search across base legal collections and user-uploaded legal PDFs.
- **Optimized CPU Embeddings**: Uses HuggingFace's `mohamed2811/Muffakir_Embedding_V2` model, selected specifically for its superior performance on multilingual and Arabic legal text, running entirely locally on CPU.
- **Two-Level Response Caching**: Exact-match in-memory cache (O(1)) plus a semantic vector cache (`semantic_cache` collection, cosine metric). Cached answers expire after `CACHE_TTL_SECONDS` and are automatically invalidated whenever the corpus is re-ingested.
- **Conversational Memory**: Per-conversation history is kept in the UI and recent messages are passed to the synthesizer, so the assistant remembers context and answers follow-up questions accurately.
- **Interactive Streamlit UI**: Provides a clean, responsive, and dark-themed chat interface. Heavy models and embeddings are preloaded on startup using Streamlit's caching mechanisms.
- **Robust Ingestion Pipeline**: A dedicated `process.py` module to parse, clean, chunk (preserving RTL text properties), and index large Arabic legal PDFs. Uses MD5 hashing to skip unchanged files for rapid processing.

---

## ⚡ Performance & Latency Optimizations

This application has been meticulously engineered to minimize latency and provide sub-second retrieval times:

1. **Heuristic-First Pattern**: Replaced multiple LLM calls in the routing and parameter extraction phases with fast regex heuristics and local synonym dictionaries. LLM calls are limited to the final answer synthesis (`ScribeAgent`) and optional query expansion for very short queries.
2. **Native DB Filtering**: Article-specific lookups leverage ChromaDB's native `where` metadata filters for `O(1)` direct lookups instead of doing expensive post-filtering on vector similarity results.
3. **Fast Fallback Query Expansion**: When language-model expansion is unavailable — and always inside the retry loop — a local dictionary of Arabic/English legal synonyms expands and refines queries instantly (no API call).
4. **Confidence-Gated Cache**: The LangGraph execution bypasses the response cache when the Router's legal-classification confidence is below `CACHE_ROUTE_CONFIDENCE`, preventing unnecessary database hits.
5. **Collection Caching**: ChromaDB collection instances (`legal_ar`, `semantic_cache`) are cached in memory as module-level singletons to prevent reloading the embedding model on every interaction.

---

## 🧠 System Architecture

### Multi-Agent Flow


The core application runs as a deterministic LangGraph state machine defined in `controllers/graph.py`:

1. **Router Node (`RouterAgent`)**: The entry point. Uses high-speed regex heuristics to classify the query's domain (legal vs. general greeting vs. out-of-domain) and detects the language.
2. **Cache Node (`CacheAgent`)**: Evaluates the Semantic Cache database. If the query's similarity to a past query exceeds `SIMILARITY_THRESHOLD` (e.g., `0.80`), it instantly returns the cached response and halts graph execution.
3. **Researcher Node (`ResearcherAgent`)**: If no cache hit, this agent takes over. 
   - Uses regex to extract article numbers or identify if the query is general.
   - Executes either a direct native `where` lookup (for specific articles) or an expanded semantic vector search.
   - **Retry Loop**: If the search returns 0 documents, a conditional edge triggers a **Reformulate Node** that instantly substitutes synonyms and retries the search. Maximum 2 attempts.
4. **Scribe Node (`ScribeAgent`)**: The synthesizer. It receives the retrieved legal documents, the conversation history, and the system prompts to draft a formal, accurate, and structured legal response via the configured LLM endpoint.

### Model-View-Controller (MVC) Structure

- **Controllers (`controllers/`)**: Business logic, Agent definitions (plain classes with typed `execute()`), and LangGraph orchestrator.
- **Models (`models/`)**: Data access layer (ChromaDB, Semantic Cache, Document Processing, cached resource factories).
- **Views (`views/`)**: Presentation layer (Streamlit). Conversation memory is owned by the UI: recent history is passed to the synthesizer with every request (no LangGraph checkpointer).

---

## 📁 Project Directory Structure

```text
Project/
├── app.py                     # Main application entry point -> routes to Streamlit
├── process.py                 # Document ingestion pipeline for parsing PDFs to ChromaDB
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys & Config)
├── PDF/                       # Directory for raw source PDF legal documents
├── data/                      # Local databases and temporary data
│   ├── chroma_db/             # ChromaDB persistence directory
│   └── processed/             # Extracted/chunked text + MD5 hashes from PDFs
├── models/                    # Data Layer
│   ├── document_processor.py  # PDF text extraction and processing orchestration
│   ├── metadata_filtering.py  # Single-pass relevance scoring
│   ├── cache.py               # Two-level response cache (exact + semantic)
│   ├── query_expansion.py     # LLM + local synonym query expansion
│   ├── resources.py           # Cached factories for the LLM and embedding model
│   └── vector_store.py        # ChromaDB setup, retrieval logic, and singleton caching
├── controllers/               # Business Logic and LLM Agents
│   ├── graph.py               # Orchestrates the agents into the final LangGraph
│   ├── graph_edges.py         # Conditional routing logic for graph state
│   ├── graph_nodes.py         # Node wrapper functions for LangGraph
│   ├── graph_state.py         # TypedDict definition of the graph state
│   ├── researcher.py          # Retrieval orchestration node
│   ├── router.py              # Heuristic Router implementation
│   └── scribe.py              # LLM Synthesis generation node
├── tests/                     # Pytest suite (router, expansion, ingestion helpers)
├── utils/                     # Configuration and helper utilities
│   ├── config.py              # Environment parsing and path setup
│   └── constants.py           # Shared keyword lists, greetings, categories
└── views/                     # Presentation Layer
    └── streamlit_app.py       # Streamlit Chat GUI
```

---

## ⚙️ Technical Stack

- **LLM Provider**: Any OpenAI-compatible endpoint via `langchain_openai.ChatOpenAI` (configured with `LLM_BASE_URL` / `LLM_MODEL`, e.g. Ollama).
- **Embedding Model**: `mohamed2811/Muffakir_Embedding_V2` (HuggingFace, Multilingual Arabic).
- **Vector Database**: [ChromaDB](https://www.trychroma.com/).
- **Orchestration**: [LangGraph](https://python.langchain.com/docs/langgraph) & LangChain Core.
- **Frontend**: [Streamlit](https://streamlit.io/).
- **PDF Processing**: `PyMuPDF` (fitz) with RTL/NFKC normalization for high-quality Arabic extraction.

---

## 🚀 Setup & Installation

### 1. Prerequisites
This project is configured for a local Python environment and has a ready-to-use virtual environment under `myenv/` in the repository root. You can also create your own virtual environment if needed.

```powershell
# Use the repo's bundled environment (Windows)
.\myenv\Scripts\Activate.ps1

# Or create a fresh venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The codebase targets **Python 3.10+** and is verified against the local environment in this repo (`myenv/`).

### 2. Install Dependencies
Install the project requirements from the root directory:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root. This repo expects the standard LLM, embedding, and retrieval settings used by `utils/config.py`.

```env
# Required: LLM provider (any OpenAI-compatible endpoint)
API_KEY=your_api_key_here
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1:8b

# Required: Embedding + retrieval settings
SIMILARITY_THRESHOLD=0.80
TOP_K=5
EMBEDDING_MODEL=mohamed2811/Muffakir_Embedding_V2
EMBEDDING_DEVICE=cpu

# Optional: cache tuning
CACHE_ROUTE_CONFIDENCE=0.70
CACHE_TTL_SECONDS=86400

# Optional: Hugging Face access token for gated models
HF_TOKEN=your_hf_token_here

# Optional: override data locations if needed
CHROMA_DB_PATH=data/chroma_db
PROCESSED_DIR=data/processed
PDF_DIR=PDF
```

### 4. Document Ingestion (Data Pipeline)
Place the legal source PDFs in the `PDF/` directory and run the ingestion script. The pipeline extracts text, normalizes Arabic content, chunks it, and stores embeddings in the local ChromaDB database.

```powershell
python process.py
```

This command is safe to re-run: it skips unchanged PDFs by checking the extracted text and hash, so the index is only rebuilt for changed documents.

> If you are upgrading from an older version of the vector store, delete `data/chroma_db` once and rerun `python process.py` so the embeddings are rebuilt consistently with the current cosine metric.

### 5. Running the Assistant
Start the Streamlit legal assistant from the project root:

```powershell
python app.py
```

You can also run the interface directly:

```powershell
python -m streamlit run views/streamlit_app.py
```

### 6. Repository Quick Start (This Repo)
This repository includes the full application layout for a legal RAG pipeline:

- `app.py` starts the Streamlit UI
- `process.py` ingests and updates the legal document index
- `controllers/` contains the LangGraph routing and agent orchestration
- `models/` contains the ChromaDB vector store, caching, and embedding logic
- `views/` contains the chat UI
- `data/` stores the vector database and processed legal text

The default workflow is:

1. Add or update PDFs in `PDF/`
2. Run `python process.py`
3. Launch `python app.py`
4. Ask legal questions in Arabic or English

---

## ⚠️ Disclaimer
**This is an AI-powered informative tool and does not constitute certified legal advice.** The answers generated are based on semantic retrieval from provided texts and probabilistic language models. Always consult a licensed attorney in the Arab Republic of Egypt for official legal rulings, contract drafting, and personal legal matters.

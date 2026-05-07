# ⚖️ Legal AI Assistant: Egyptian Civil Code

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/LangGraph-Agentic-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![VectorStore](https://img.shields.io/badge/VectorStore-ChromaDB-blue)

An advanced, agentic AI legal assistant specialized in the **Egyptian Civil Code**. Built using the LangGraph framework, this project employs a robust multi-agent architecture to route queries, perform deep semantic research, cache responses, and generate structured bilingual answers.

The system is designed with a strict MVC (Model-View-Controller) structure, making it highly modular and extensible. The agents adhere to an Object-Oriented paradigm following SOLID principles, ensuring scalability and ease of maintenance.

---

## ✨ Key Features

- **Agentic Workflow Engine:** Orchestrated by **LangGraph**, it acts as a state machine that dynamically coordinates multiple specialized agents. The graph is lazily compiled and cached to ensure optimal execution speeds.
- **Object-Oriented Agents:** Agents inherit from a central `BaseAgent` abstract class (`controllers/base_agent.py`) for strict interface adherence and single responsibility. Agents are initialized as singletons at the module level.
- **Bilingual Interface:** Detects the language of the query and operates fluently in both Arabic and English, allowing for seamless context translation.
- **RAG & Vector Retrieval:** Employs **ChromaDB** (`models/vector_store.py`) with advanced sentence transformers to securely search across base legal collections (stored in the `legal_ar` collection) and user-uploaded legal PDFs.
- **Optimized Embeddings:** Uses HuggingFace's `mohamed2811/Muffakir_Embedding_V2` model, specifically selected for its superior performance on multilingual Arabic text. Embeddings run completely locally.
- **Semantic Caching:** Identifies previously answered concepts and matches semantic similarity using a dedicated `semantic_cache` ChromaDB collection to bypass LLM calls, responding instantaneously and significantly reducing API overhead.
- **Conversational Memory:** Uses `langgraph-checkpoint-sqlite` to persist state across conversation threads (using `thread_id`), managing history and contextualizing follow-up questions over long, complex sessions.
- **Interactive UI:** Powered by **Streamlit**, providing a clean, responsive, and dark-themed Legal chat interface. Heavy models and embeddings are preloaded on startup using `@st.cache_resource`.
- **Robust Ingestion Pipeline:** A dedicated `process.py` module to parse, clean, chunk (preserving RTL properties), and index large Arabic legal PDFs into the vector store. Uses MD5 hashing to skip unchanged files for rapid processing.

---

## 🧠 Architecture Overview

### Model-View-Controller (MVC) Layout
- **Controllers (`controllers/`)**: Contains the business logic, agent definitions, and the LangGraph orchestrator.
- **Models (`models/`)**: Contains the data access layer, including Vector Database interactions, semantic cache, and SQLite checkpointer memory.
- **Views (`views/`)**: Presentation layer built using Streamlit.

### Multi-Agent Flow

![LangGraph Workflow Architecture](graph.png)

The core of the application relies on specialized LangChain/LangGraph agents, forming a state machine defined in `controllers/graph.py`:

1. **RouterAgent**: The entry point of the graph. It classifies the user's query intent (e.g., legal question, general greeting, or out-of-domain) and detects the requested language.
2. **CacheAgent**: Evaluates the Semantic Cache database for similar prior queries based on `SIMILARITY_THRESHOLD`. If a highly similar query exists, it bypasses the LLM and instantly returns the cached response. **Note:** Semantic cache evaluation is skipped if router confidence is < 0.8.
3. **ResearcherAgent**: If the query requires new research, this agent constructs optimized search terms, queries ChromaDB, and extracts the top `TOP_K` relevant legal articles, laws, and precedents. **Retry Loop:** Features a maximum of 2 retrieval attempts; if the first attempt fails to retrieve sufficiently relevant context, the query is reformulated.
4. **ScribeAgent**: The synthesizer. It takes the retrieved legal documents and drafts a formal, structured, and highly accurate response. It ensures the final output strictly matches the language and context requested by the user.

---

## 📁 Project Structure

The codebase is meticulously organized to adhere to the Model-View-Controller pattern:

```text
Project/
├── app.py                     # Main application entry point -> routes to Streamlit
├── process.py                 # Document ingestion pipeline for PDFs
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys & Config)
├── PDF/                       # Directory for raw source PDF legal documents
├── data/                      # Local databases and temporary data
│   ├── chroma_db/             # ChromaDB persistence directory
│   ├── checkpoints.db         # SQLite checkpointer for conversation memory
│   └── processed/             # Extracted/chunked text + MD5 hashes from PDFs
├── models/                    # Data Layer
│   ├── document_processor.py  # PDF text extraction and chunking logic
│   ├── vector_store.py        # ChromaDB setup, retrieval logic, and singleton caching
│   ├── semantic_cache.py      # Cache database integration
│   └── memory.py              # SQLite checkpointer initialization
├── controllers/               # Business Logic and LLM Agents
│   ├── base_agent.py          # Abstract BaseAgent class (OOP)
│   ├── router.py              # RouterAgent implementation
│   ├── cache_controller.py    # CacheAgent implementation
│   ├── researcher.py          # ResearcherAgent implementation
│   ├── scribe.py              # ScribeAgent implementation
│   └── graph.py               # Orchestrates the agents into a LangGraph state machine
├── utils/                     # Configuration and helper utilities
│   └── config.py              # Centralized configuration and environment loading
└── views/                     # Presentation Layer
    └── streamlit_app.py       # Streamlit Chat GUI
```

---

## ⚙️ Key Implementation Details

- **Dependencies & Models**:
  - **LLM**: OpenRouter via `langchain_openai.ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"`
  - **Embeddings**: HuggingFace `mohamed2811/Muffakir_Embedding_V2` (Forced to CPU via `model_kwargs={"device": "cpu"}`)
  - **Vector Store**: ChromaDB with singleton caching. Collection instances (`legal_ar`, `semantic_cache`) are cached at the module level to avoid reloading embeddings on every query.
- **Ingestion Optimization (`process.py`)**: Skips unchanged PDFs by comparing MD5 hashes and batches ChromaDB writes (500 documents per batch) to drastically improve ingestion speed on subsequent runs.
- **Agent Initialization**: Agents are initialized as singletons at the module level.
- **Performance Characteristics**:
  - The first query is typically slower due to lazy graph compilation.
  - Re-running `process.py` on unchanged PDFs is extremely fast as extraction and ChromaDB writes are entirely skipped.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed. It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create a Virtual Environment
python -m venv agents-env

# Activate (Windows)
.\agents-env\Scripts\activate

# Activate (Mac/Linux)
source agents-env/bin/activate
```

### 2. Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory. **There are no defaults for these variables — the app will crash if they are missing.**

```env
# Required: OpenRouter API Key for the LLM
OPEN_ROUTER_API=your_api_key_here

# Required: Core parameters
SIMILARITY_THRESHOLD=0.80
TOP_K=5
EMBEDDING_MODEL=mohamed2811/Muffakir_Embedding_V2
LLM_MODEL=openai/gpt-oss-20b:free

# Optional: HuggingFace Token (if using gated models)
HF_TOKEN=your_hf_token_here
```

### 4. Document Ingestion (Data Pipeline)
To populate the vector database with legal knowledge, place your source PDFs in the `PDF/` directory and run the ingestion script. This will parse the PDFs, clean the text, perform semantic chunking, and persist the embeddings.

```bash
python process.py
```
*Note: You can safely run this command multiple times. Unchanged PDFs will be skipped automatically.*

### 5. Running the Assistant
You can launch the complete pipeline and the Streamlit UI using the following commands:

**Option A (Using App Launcher):**
This acts as the standard entry point and spins up Streamlit locally.
```bash
python app.py
```

**Option B (Direct Streamlit Run):**
If you want to run Streamlit directly targeting the presentation layer:
```bash
python -m streamlit run views/streamlit_app.py
```

---

## ⚠️ Disclaimer
**This is an AI-powered informative tool and does not constitute certified legal advice.** The answers generated are based on semantic retrieval from provided texts and probabilistic language models. Always consult a licensed attorney in the Arab Republic of Egypt for official legal rulings, contract drafting, and personal legal matters.

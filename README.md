# ⚖️ Legal AI Assistant: Egyptian Civil Code

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/LangGraph-Agentic-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![VectorStore](https://img.shields.io/badge/VectorStore-ChromaDB-blue)

An advanced, agentic AI legal assistant specialized in the **Egyptian Civil Code**. Built using the LangGraph framework, this project employs a robust multi-agent architecture to route queries, perform deep semantic research, cache responses, and generate structured bilingual answers. 

The system is designed with an strict MVC (Model-View-Controller) structure, making it highly modular and extensible. The agents adhere to an Object-Oriented paradigm following SOLID principles, ensuring scalability and ease of maintenance.

---

## ✨ Key Features

- **Agentic Workflow Engine:** Orchestrated by **LangGraph**, it acts as a state machine that dynamically coordinates multiple specialized agents. The graph is lazily compiled and cached to ensure optimal execution speeds.
- **Object-Oriented Agents:** Agents inherit from a central `BaseAgent` abstract class (`controllers/base_agent.py`) for strict interface adherence and single responsibility. Agents are initialized as singletons at the module level.
- **Bilingual Interface:** Detects the language of the query and operates fluently in both Arabic and English, allowing for seamless context translation.
- **RAG & Vector Retrieval:** Employs **ChromaDB** (`models/vector_store.py`) with advanced sentence transformers to securely search across base legal collections (stored in the `legal_ar` collection) and user-uploaded legal PDFs.
- **Optimized Embeddings:** Uses HuggingFace's `mohamed2811/Muffakir_Embedding_V2` model, specifically selected for its superior performance on multilingual Arabic text.
- **Semantic Caching:** Identifies previously answered concepts and matches semantic similarity using a dedicated `semantic_cache` ChromaDB collection to bypass LLM calls, responding instantaneously and significantly reducing API overhead.
- **Conversational Memory:** Uses `langgraph-checkpoint-sqlite` to persist state across conversation threads (using `thread_id`), managing history and contextualizing follow-up questions over long, complex sessions. The data is saved in `data/checkpoints.db`.
- **Interactive UI:** Powered by **Streamlit**, providing a clean, responsive, and dark-themed Legal chat interface. Heavy models and embeddings are preloaded on startup using `@st.cache_resource`.
- **Robust Ingestion Pipeline:** A dedicated `process.py` module to parse, clean, chunk (preserving RTL properties), and index large Arabic legal PDFs into the vector store. Processing artifacts are stored in `data/processed/`.

---

## 🧠 Multi-Agent Architecture

The core of the application relies on specialized LangChain/LangGraph agents, each handling a distinct part of the query lifecycle. The workflow forms a state machine defined in `controllers/graph.py`:

1. **RouterAgent (`router.py`)**: The entry point of the graph. It classifies the user's query intent (e.g., legal question, general greeting, or out-of-domain) and detects the requested language.
2. **CacheAgent (`cache_controller.py`)**: Evaluates the Semantic Cache database for similar prior queries based on `SIMILARITY_THRESHOLD`. If a highly similar query exists, it bypasses the LLM and instantly returns the cached response.
3. **ResearcherAgent (`researcher.py`)**: If the query requires new research, this agent takes over. It constructs optimized search terms, queries ChromaDB, and extracts the top `TOP_K` relevant legal articles, laws, and precedents.
4. **ScribeAgent (`scribe.py`)**: The synthesizer. It takes the retrieved legal documents and drafts a formal, structured, and highly accurate response. It ensures the final output strictly matches the language and context requested by the user.
5. **Graph Orchestrator (`graph.py`)**: Ties all agents together. The LangGraph state machine routes the flow of data across nodes, executing conditional edges based on agent assessments.

---

## 📁 Project Structure (MVC)

The codebase is meticulously organized adhering to the Model-View-Controller pattern:

```text
Project/
├── app.py                     # Main application entry point (Launcher)
├── process.py                 # Document ingestion pipeline for PDFs
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys & Config)
├── PDF/                       # Directory for raw source PDF legal documents
├── data/                      # Local databases and temporary data
│   ├── chroma_db/             # ChromaDB persistence directory
│   ├── checkpoints.db         # SQLite checkpointer for conversation memory
│   └── processed/             # Extracted/chunked text from ingested PDFs
├── models/                    # Data Layer
│   ├── document_processor.py  # PDF text extraction and chunking logic
│   ├── vector_store.py        # ChromaDB setup and retrieval logic
│   ├── semantic_cache.py      # Cache database integration
│   └── memory.py              # SQLite checkpointer initialization
├── controllers/               # Business Logic and LLM Agents
│   ├── base_agent.py          # Abstract BaseAgent class (OOP)
│   ├── router.py              # RouterAgent implementation
│   ├── cache_controller.py    # CacheAgent implementation
│   ├── researcher.py          # ResearcherAgent implementation
│   ├── scribe.py              # ScribeAgent implementation
│   └── graph.py               # Orchestrates the agents into a LangGraph state machine
└── views/                     # Presentation Layer
    └── streamlit_app.py       # Streamlit Chat GUI
```

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
Install all required libraries, including LangGraph, Streamlit, and ChromaDB:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory. The application relies on OpenRouter via `langchain_openai.ChatOpenAI` (base URL `https://openrouter.ai/api/v1`). You can customize the models and retrieval parameters:

```env
# Required: OpenRouter API Key for the LLM
OPEN_ROUTER_API=your_api_key_here

# Optional: Overrides for Models and Parameters
LLM_MODEL=openai/gpt-oss-20b:free
EMBEDDING_MODEL=mohamed2811/Muffakir_Embedding_V2
SIMILARITY_THRESHOLD=0.80
TOP_K=5

# Optional: HuggingFace Token (if needed for gated models)
HF_TOKEN=your_hf_token_here
```

### 4. Document Ingestion (Data Pipeline)
To populate the vector database with legal knowledge, place your source PDFs in the `PDF/` directory and run the ingestion script. This will parse the PDFs, clean the text, perform semantic chunking, and persist the embeddings in `data/chroma_db/`.

```bash
python process.py
```

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

## 🛠️ Technology Stack
- **Frameworks:** LangGraph, LangChain, Streamlit
- **LLM API Provider:** OpenRouter API (`langchain_openai.ChatOpenAI`)
- **Default LLM:** `openai/gpt-oss-20b:free`
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace `mohamed2811/Muffakir_Embedding_V2`
- **Document Processing:** PyMuPDF (`fitz`)
- **Persistence:** SQLite (for conversational memory and semantic cache)

---

## ⚠️ Disclaimer
**This is an AI-powered informative tool and does not constitute certified legal advice.** The answers generated are based on semantic retrieval from provided texts and probabilistic language models. Always consult a licensed attorney in the Arab Republic of Egypt for official legal rulings, contract drafting, and personal legal matters.

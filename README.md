# Enterprise RAG Assistant

A production-quality **Retrieval-Augmented Generation (RAG)** application that lets users query enterprise documents through a modern web interface. Built with Python, LangChain, SentenceTransformers, FAISS, FastAPI, and Streamlit.

---

## Overview

The system:

- **Loads** enterprise documents (PDF, Markdown, TXT) from a folder
- **Chunks** text with configurable size and overlap
- **Embeds** chunks using SentenceTransformers (`all-MiniLM-L6-v2`)
- **Stores** embeddings in a FAISS vector index
- **Retrieves** relevant context for each user query
- **Generates** answers using an open source LLM (Mistral or Llama)
- **Exposes** a FastAPI backend and a Streamlit chat UI

---

## Architecture

```
User → Web UI (Streamlit) → API (FastAPI) → RAG Pipeline → Retrieval → Vector DB (FAISS)
                                                                              ↓
User ← Web UI ← API ← Answer ← LLM (Mistral/Llama) ← Prompt + Retrieved Context
```

**Layers:**

| Layer        | Responsibility                          |
|-------------|------------------------------------------|
| **UI**      | Chat interface, theme, sources display   |
| **API**     | `POST /query`, request/response schema   |
| **RAG**     | Retrieve context, build prompt, call LLM |
| **Retrieval** | Embed query, similarity search, top-k  |
| **Vector store** | FAISS index, persist/reload           |
| **Ingestion**   | Load docs, chunk, embed, index         |

---

## Project Structure

```
enterprise_rag_assistant/
├── app/
│   ├── api.py              # FastAPI endpoints
│   └── rag_pipeline.py     # RAG orchestration
├── ingestion/
│   ├── document_loader.py  # PDF, MD, TXT loading
│   ├── chunker.py          # RecursiveCharacterTextSplitter
│   └── embedding_pipeline.py
├── vectorstore/
│   └── faiss_store.py      # FAISS index + docstore
├── models/
│   └── llm_model.py        # Swappable LLM (Mistral/Llama)
├── ui/
│   └── streamlit_app.py    # Streamlit chat UI
├── configs/
│   └── config.yaml         # Model, chunk, retrieval, paths
├── utils/
│   ├── logger.py
│   └── config_loader.py
├── data/
│   └── raw_docs/           # Place documents here
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and enter project

```bash
cd enterprise_rag_assistant
```

### 2. Create virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables

```bash
copy .env.example .env
```

Edit `.env` and set `HUGGINGFACE_HUB_TOKEN` if you use gated models (e.g. Llama).

### 5. Add documents

Put PDF, Markdown, or TXT files in `data/raw_docs/`.

### 6. Build the index (ingestion)

Run the ingestion pipeline to chunk, embed, and build the FAISS index:

```bash
python -m ingestion.index_builder
```

This creates:

- `data/faiss_index`
- `data/faiss_docstore.pkl`

### 7. Run the API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 8. Run the UI

```bash
streamlit run ui/streamlit_app.py
```

---

## Example Queries

- “What are the main conclusions of the report?”
- “Summarize the section on risk mitigation.”
- “Which deadlines are mentioned in the documents?”

---

## Configuration

Edit `configs/config.yaml` to change:

- **embedding.model** : SentenceTransformers model
- **chunking.chunk_size** / **chunk_overlap**
- **retrieval.top_k**
- **llm.model_name**: e.g. `mistralai/Mistral-7B-Instruct-v0.2` or `meta-llama/Llama-3-8B-Instruct`
- **vectorstore** / **paths**: index and raw docs locations

---

## Testing

```bash
pytest tests/ -v
```

Covers document loading, chunking, vector retrieval, and API response shape.

---

## License

Use and modify as needed for your organization.

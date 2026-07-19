# Local PDF RAG Agent

A production-oriented Retrieval-Augmented Generation (RAG) service. Upload PDFs, index their chunks in a local vector database, and ask questions through the browser UI or REST API. Answers are instructed to use retrieved context only.

## Architecture

```text
PDF -> PyMuPDF + RapidOCR -> RecursiveCharacterTextSplitter -> Vector Embeddings
    -> FAISS Index -> BM25 + Vector Hybrid Retriever -> Grounded Prompt -> OpenRouter LLM
```

`app/` keeps the responsibilities separate: `api/routes.py` owns HTTP behaviour, `rag/loader.py` reads PDFs, `splitter.py` chunks them, `embeddings.py` creates the embedding client, `vectorstore.py` persists FAISS, `retriever.py` retrieves chunks, and `chain.py` creates grounded or streamed answers. `app/static/index.html` is the glassmorphism chat interface.

## Key Recommendations

1. **Multi-Column & Table OCR Parsing**: Upgrade the document processing pipeline to support multi-column layout detection and complex table extraction for scientific papers.
2. **Persistent Cloud Vector Store**: Migrate local FAISS files to managed cloud vector databases (e.g. Qdrant, Pinecone) for multi-tenant scalability and low-latency index updates.
3. **Automated RAG Evaluation**: Integrate evaluation frameworks (such as RAGAS) to dynamically track precision, recall, and context faithfulness metrics.

## Install and run

Requires Python 3.12+ and an OpenRouter API key. Create a virtual environment, install dependencies, then copy `.env.example` to `.env` and add the key.

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the web app or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation. A sample PDF is available at `documents/sample.pdf`.

## Environment

```dotenv
OPENROUTER_API_KEY=your-openrouter-key-here
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTORSTORE_PATH=./vectorstore
DOCUMENTS_PATH=./documents
MAX_UPLOAD_MB=50
LOG_LEVEL=INFO
```

The service stores files and a separate FAISS index per session. Keep `vectorstore/` private: FAISS persistence uses Python pickle internally and must only be loaded from a trusted disk location.

## REST API

Create a session first, then include its `session_id` when ingesting and querying.

```bash
# Create a session
curl -X POST http://127.0.0.1:8000/api/sessions

# Ingest a PDF (substitute the returned session id)
curl -X POST "http://127.0.0.1:8000/ingest?session_id=SESSION_ID" \
  -F "file=@documents/sample.pdf"

# Query with sources in JSON
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID","question":"What does the document say?","top_k":4}'

# Stream plain-text answer chunks
curl -N -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID","question":"Summarize the document.","stream":true}'

curl http://127.0.0.1:8000/health
```

The JSON query response includes the chunks and metadata used as sources. The streaming variant appends a machine-readable `__SOURCES__:` marker for the bundled UI.

## Testing and containers

```bash
python -m pytest -q
docker compose up --build
```

Tests use fake embeddings and mock the model, so they do not make external network calls. The Docker configuration persists `documents/` and `vectorstore/` as mounted local directories.

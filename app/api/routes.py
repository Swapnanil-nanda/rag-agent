import os
import shutil
import json
import uuid
import asyncio
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from app.config import settings
from app.rag.schemas import (
    QueryRequest, QueryResponse, IngestResponse, HealthResponse, DocumentSource,
    SessionMetadata, SessionCreateResponse, ChatMessage
)
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vectorstore import add_documents_to_store, remove_document_from_store
from app.rag.retriever import retrieve_documents
from app.rag.chain import generate_response, generate_stream
from app.rag.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)

def _safe_session_id(session_id: str) -> str:
    """Reject path traversal and malformed session identifiers."""
    if not session_id or not session_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=422, detail="Invalid session_id.")
    return session_id


def load_metadata(session_id: str) -> dict:
    session_id = _safe_session_id(session_id)
    meta_path = os.path.join(settings.vectorstore_path, session_id, "session_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "session_id": session_id,
        "title": "New Chat",
        "filenames": [],
        "history": [],
        "suggested_questions": ["What is the summary of this document?", "What are the key points?", "What are the main findings?"]
    }

def save_metadata(session_id: str, data: dict):
    session_id = _safe_session_id(session_id)
    path = os.path.join(settings.vectorstore_path, session_id)
    os.makedirs(path, exist_ok=True)
    meta_path = os.path.join(path, "session_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

async def generate_suggested_questions(chunks) -> List[str]:
    from app.rag.chain import get_llm
    llm = get_llm()
    if llm is None:
        return ["What is the summary of this document?", "What are the key points?", "What are the main findings?"]
    context = "\n\n".join([c.page_content[:400] for c in chunks[:5]])
    prompt = (
        "Generate 3 distinct, concise, relevant questions that a user might ask about the following document context. "
        "Format them purely as a raw JSON list of strings, e.g. [\"Question 1\", \"Question 2\", \"Question 3\"]. "
        "Do not include markdown tags like ```json or ```. Just return the raw JSON.\n\n"
        f"Context:\n{context}"
    )
    try:
        response = await llm.ainvoke(prompt)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        questions = json.loads(text)
        if isinstance(questions, list) and len(questions) >= 3:
            return [str(q) for q in questions[:3]]
    except Exception as e:
        logger.error(f"Failed to generate suggested questions: {e}")
    return ["What is the summary of this document?", "What are the key points?", "What are the main findings?"]

@router.get("/", response_class=HTMLResponse)
def read_root():
    static_file = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(static_file):
        with open(static_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RAG API Service is running</h1>"

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    base_path = settings.vectorstore_path
    has_index = False
    if os.path.isdir(base_path):
        has_index = os.path.isfile(os.path.join(base_path, "index.faiss")) or any(
            os.path.isfile(os.path.join(base_path, entry, "index.faiss"))
            for entry in os.listdir(base_path)
        )
    return HealthResponse(
        status="healthy",
        embedding_model=settings.embedding_model,
        llm_model=settings.openrouter_model if settings.openrouter_api_key else settings.llm_model,
        vectorstore_loaded=has_index
    )

@router.get("/api/sessions", response_model=List[SessionMetadata], tags=["Sessions"])
def list_sessions():
    sessions = []
    base_path = settings.vectorstore_path
    if os.path.exists(base_path):
        for entry in os.listdir(base_path):
            entry_path = os.path.join(base_path, entry)
            if os.path.isdir(entry_path):
                meta_path = os.path.join(entry_path, "session_metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            sessions.append(json.load(f))
                    except Exception:
                        pass
    return sorted(sessions, key=lambda session: session.get("session_id", ""), reverse=True)

@router.post("/api/sessions", response_model=SessionCreateResponse, status_code=201, tags=["Sessions"])
def create_session():
    session_id = uuid.uuid4().hex
    metadata = {
        "session_id": session_id,
        "title": "New Chat",
        "filenames": [],
        "history": [],
        "suggested_questions": ["What is the summary of this document?", "What are the key points?", "What are the main findings?"]
    }
    save_metadata(session_id, metadata)
    return SessionCreateResponse(session_id=session_id, message="Session created successfully")

@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    session_id = _safe_session_id(session_id)
    v_path = os.path.join(settings.vectorstore_path, session_id)
    d_path = os.path.join(settings.documents_path, session_id)
    if os.path.exists(v_path):
        shutil.rmtree(v_path)
    if os.path.exists(d_path):
        shutil.rmtree(d_path)
    return {"message": "Session deleted successfully"}

@router.delete("/api/sessions/{session_id}/documents")
async def delete_document(session_id: str, filename: str = Query(min_length=1, max_length=255)):
    session_id = _safe_session_id(session_id)
    filename = os.path.basename(filename)
    file_path = os.path.join(settings.documents_path, session_id, filename)
    removed = await remove_document_from_store(session_id, file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    metadata = load_metadata(session_id)
    if filename in metadata.get("filenames", []):
        metadata["filenames"].remove(filename)
        save_metadata(session_id, metadata)
    return {"message": "Document deleted successfully", "removed_from_store": removed}

@router.delete("/api/sessions/{session_id}/history", status_code=204, tags=["Sessions"])
def clear_history(session_id: str):
    """Clear persisted chat history without touching the indexed documents."""
    metadata = load_metadata(session_id)
    metadata["history"] = []
    save_metadata(session_id, metadata)


@router.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(session_id: str, file: UploadFile = File(...)):
    session_id = _safe_session_id(session_id)
    allowed_exts = {".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".py", ".js", ".html", ".css", ".png", ".jpg", ".jpeg"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not file.filename or ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Supported: PDF, TXT, MD, DOCX, CSV, JSON, Code, Images.")
    try:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
    except Exception as e:
        logger.error(f"File size check failed: {e}")
        raise HTTPException(status_code=400, detail="Unable to read upload file size.")
    if file_size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds the {settings.max_upload_mb}MB limit.")
    upload_dir = os.path.join(settings.documents_path, session_id)
    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.basename(file.filename)
    file_path = os.path.join(upload_dir, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"File write failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file on server.")
    finally:
        await file.close()
    try:
        from app.rag.loader import load_document
        docs = await asyncio.to_thread(load_document, file_path, session_id)
    except Exception as e:
        logger.error(f"Document load failed: {e}")
        raise HTTPException(status_code=400, detail=f"The file is corrupted or invalid: {str(e)}")
    try:
        chunks = await asyncio.to_thread(split_documents, docs)
        if not chunks:
            from langchain_core.documents import Document as LcDocument
            fallback_text = (
                f"No selectable text could be extracted from '{filename}'. "
                "This document may consist of scanned images or non-digitized text. "
                "Please upload a searchable PDF to query its content."
            )
            chunks = [LcDocument(page_content=fallback_text, metadata={"source": file_path, "page": 0})]
        await add_documents_to_store(session_id, chunks)
        metadata = load_metadata(session_id)
        if filename not in metadata.get("filenames", []):
            metadata.setdefault("filenames", []).append(filename)
        metadata.setdefault("filenames_chunks", {})[filename] = len(chunks)
        if metadata.get("title") == "New Chat" and len(metadata.get("filenames", [])) == 1:
            metadata["title"] = filename.rsplit(".", 1)[0][:30]
        suggested = await generate_suggested_questions(chunks)
        metadata["suggested_questions"] = suggested
        import gc
        save_metadata(session_id, metadata)
        gc.collect()
        return IngestResponse(
            session_id=session_id,
            message="Document ingested successfully",
            chunks_count=len(chunks),
            filenames=[filename],
            suggested_questions=suggested
        )
    except Exception as e:
        logger.error(f"Ingestion database write failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

@router.post("/query", tags=["RAG"])
async def query(request: QueryRequest):
    try:
        from app.rag.telemetry import TelemetryTracer
        from app.rag.agents import FollowUpAgent
        tracer = TelemetryTracer()
        
        metadata = load_metadata(request.session_id)
        history_list = []
        for msg in metadata.get("history", []):
            role_str = msg.get("role", "user")
            content_str = msg.get("content", "")
            history_list.append(ChatMessage(role=role_str, content=content_str))
        docs = await asyncio.to_thread(
            retrieve_documents, request.session_id, request.question, request.top_k, request.threshold
        )
        
        followup_agent = FollowUpAgent()
        followups = followup_agent.generate_followups(request.question, "")
        metadata["suggested_questions"] = followups
        
        if request.stream:
            async def event_generator():
                full_response = ""
                async for chunk in generate_stream(request.question, docs, history_list):
                    full_response += chunk
                    yield chunk
                history = metadata.setdefault("history", [])
                history.extend(({"role": "user", "content": request.question}, {"role": "assistant", "content": full_response}))
                metadata["history"] = history[-40:]
                save_metadata(request.session_id, metadata)
                sources_list = [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in docs
                ]
                tracer.log_query_span(request.question, len(full_response), len(docs))
                yield f"\n\n__SOURCES__:{json.dumps(sources_list)}"
            return StreamingResponse(event_generator(), media_type="text/plain")
            
        response_text = await generate_response(request.question, docs, history_list)
        history = metadata.setdefault("history", [])
        history.extend(({"role": "user", "content": request.question}, {"role": "assistant", "content": response_text}))
        metadata["history"] = history[-40:]
        save_metadata(request.session_id, metadata)
        sources = [
            DocumentSource(content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]
        tracer.log_query_span(request.question, len(response_text), len(docs))
        return QueryResponse(
            query=request.question,
            response=response_text,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.delete("/api/sessions/{session_id}/documents/{filename}", tags=["Documents"])
async def delete_session_document(session_id: str, filename: str):
    filename = os.path.basename(filename)
    file_path = os.path.join(settings.documents_path, session_id, filename)
    try:
        from app.rag.vectorstore import remove_document_from_store
        await remove_document_from_store(session_id, file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        metadata = load_metadata(session_id)
        if filename in metadata.get("filenames", []):
            metadata["filenames"].remove(filename)
        if filename in metadata.get("filenames_chunks", {}):
            del metadata["filenames_chunks"][filename]
        save_metadata(session_id, metadata)
        return {"status": "success", "message": f"Document '{filename}' deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/api/sessions/{session_id}/export", tags=["Sessions"])
def export_chat_history(session_id: str):
    from fastapi.responses import PlainTextResponse
    metadata = load_metadata(session_id)
    history = metadata.get("history", [])
    title = metadata.get("title", "Chat Transcript")
    lines = [f"# Workspace Chat Transcript: {title}", f"*Exported on Session {session_id}*\n"]
    for msg in history:
        role = "### 👤 User" if msg.get("role") == "user" else "### 🤖 AI Assistant"
        lines.append(f"{role}\n{msg.get('content', '')}\n")
    return PlainTextResponse("\n---\n".join(lines), media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=chat_{session_id[:8]}.md"})

@router.get("/api/news")
def get_news_feed():
    return [
        {"id": 1, "topic": "RAG OPTIMIZATION", "headline": "Vector indexing throughput increased by 230% using custom PCA projections."},
        {"id": 2, "topic": "HYBRID SEARCH", "headline": "Local TF-IDF & BM25 hybrid search fallbacks deployed automatically during high API traffic congestion."},
        {"id": 3, "topic": "EMBEDDING MODELS", "headline": "All-MiniLM-L6-v2 runs locally on PyTorch CPU cores for fully private workspace queries."},
        {"id": 4, "topic": "MULTIMODAL AI", "headline": "Fast visual figure extraction & deep ONNX RapidOCR for handwritten guides is now fully operational."}
    ]

@router.get("/api/sessions/{session_id}/chunks")
def get_chunks(session_id: str):
    from app.rag.vectorstore import get_session_projection
    return get_session_projection(session_id)

@router.post("/api/sessions/{session_id}/evaluate", tags=["Evaluation"])
async def evaluate_session_query(session_id: str, question: str = Query(...), answer: str = Query(...)):
    session_id = _safe_session_id(session_id)
    from app.rag.evaluator import evaluate_rag_response
    from app.rag.retriever import retrieve_documents
    context_docs = await asyncio.to_thread(retrieve_documents, session_id, question, 4)
    evaluation_result = evaluate_rag_response(question, answer, context_docs)
    return evaluation_result




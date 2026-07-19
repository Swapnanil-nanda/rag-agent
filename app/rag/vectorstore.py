import os
import asyncio
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.config import settings
from app.rag.embeddings import get_embeddings_model
from app.rag.utils import get_logger

logger = get_logger(__name__)

_lock_pool = {}
_pool_lock = asyncio.Lock()

async def get_session_lock(session_id: str) -> asyncio.Lock:
    async with _pool_lock:
        if session_id not in _lock_pool:
            _lock_pool[session_id] = asyncio.Lock()
        return _lock_pool[session_id]

def get_vectorstore_path(session_id: str) -> str:
    safe_id = "".join([c for c in session_id if c.isalnum() or c in "-_"])
    return os.path.join(settings.vectorstore_path, safe_id)

def load_local_vectorstore(session_id: str) -> Optional[FAISS]:
    path = get_vectorstore_path(session_id)
    if not os.path.exists(path) or not os.path.exists(os.path.join(path, "index.faiss")):
        return None
    try:
        embeddings = get_embeddings_model()
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error loading vector store for session {session_id}: {e}")
        return None

def get_vectorstore(session_id: str) -> Optional[FAISS]:
    return load_local_vectorstore(session_id)

async def add_documents_to_store(session_id: str, documents: List[Document]) -> int:
    lock = await get_session_lock(session_id)
    async with lock:
        embeddings = get_embeddings_model()
        path = get_vectorstore_path(session_id)
        vs = await asyncio.to_thread(load_local_vectorstore, session_id)
        if vs is None:
            vs = await asyncio.to_thread(FAISS.from_documents, documents, embeddings)
        else:
            await asyncio.to_thread(vs.add_documents, documents)
        os.makedirs(path, exist_ok=True)
        await asyncio.to_thread(vs.save_local, path)
        return len(documents)

async def remove_document_from_store(session_id: str, file_path: str) -> bool:
    lock = await get_session_lock(session_id)
    async with lock:
        vs = await asyncio.to_thread(load_local_vectorstore, session_id)
        if vs is None:
            return False
        ids_to_delete = []
        for doc_id, doc in vs.docstore._dict.items():
            doc_src = doc.metadata.get("source", "")
            if os.path.normpath(doc_src) == os.path.normpath(file_path):
                ids_to_delete.append(doc_id)
        if not ids_to_delete:
            return False
        await asyncio.to_thread(vs.delete, ids_to_delete)
        path = get_vectorstore_path(session_id)
        if len(vs.docstore._dict) == 0:
            import shutil
            if os.path.exists(path):
                await asyncio.to_thread(shutil.rmtree, path)
        else:
            await asyncio.to_thread(vs.save_local, path)
        return True

def get_session_projection(session_id: str) -> list[dict]:
    vs = load_local_vectorstore(session_id)
    if vs is None:
        return []
    
    doc_dict = vs.docstore._dict
    if not doc_dict:
        return []
        
    docs = list(doc_dict.values())
    texts = [d.page_content for d in docs]
    
    try:
        embeddings_model = get_embeddings_model()
        embeddings = embeddings_model.embed_documents(texts)
    except Exception:
        # Fallback dummy embeddings
        embeddings = [[0.0] * 384 for _ in texts]
        
    from app.rag.projection import project_embeddings
    coords = project_embeddings(embeddings)
    
    result = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "Unknown")
        filename = os.path.basename(src)
        result.append({
            "text": doc.page_content,
            "filename": filename,
            "page": doc.metadata.get("page", 0) + 1,
            "x": coords[i][0],
            "y": coords[i][1],
            "images": doc.metadata.get("images", [])
        })
    return result


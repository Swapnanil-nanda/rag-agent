from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=20_000)

class SessionMetadata(BaseModel):
    session_id: str
    title: str
    filenames: List[str] = Field(default_factory=list)
    filenames_chunks: Optional[Dict[str, int]] = Field(default_factory=dict)
    history: List[ChatMessage] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)

class SessionCreateResponse(BaseModel):
    session_id: str
    message: str

class QueryRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(..., min_length=1, max_length=4_000)
    top_k: int = Field(default=4, ge=1, le=12)
    threshold: Optional[float] = Field(default=None, ge=0, le=1)
    stream: bool = False

class DocumentSource(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[DocumentSource]

class IngestResponse(BaseModel):
    session_id: str
    message: str
    chunks_count: int
    filenames: List[str]
    suggested_questions: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
    vectorstore_loaded: bool

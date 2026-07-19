"""Local embedding model factory, shared by all FAISS operations."""

from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

_embeddings_instance: HuggingFaceEmbeddings | None = None

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Load a local, CPU-friendly embedding model with no API key requirement."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance
    _embeddings_instance = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return _embeddings_instance

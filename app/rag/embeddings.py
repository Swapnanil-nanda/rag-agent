import os
import gc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

_embeddings_instance = None

def get_embeddings_model():
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance
    
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    _embeddings_instance = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )
    gc.collect()
    return _embeddings_instance

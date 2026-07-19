import numpy as np
from app.config import settings

class LightweightEmbeddings:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _text_to_vector(self, text: str) -> list[float]:
        if not isinstance(text, str):
            text = str(text or "")
        tokens = text.lower().split()
        if not tokens:
            return [0.0] * self.dim
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            idx = abs(hash(token)) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

_embeddings_instance = None

def get_embeddings_model():
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance
    if settings.openai_api_key and not settings.openai_api_key.startswith("mock"):
        try:
            from langchain_openai import OpenAIEmbeddings
            _embeddings_instance = OpenAIEmbeddings(api_key=settings.openai_api_key)
            return _embeddings_instance
        except Exception:
            pass
    _embeddings_instance = LightweightEmbeddings(dim=384)
    return _embeddings_instance

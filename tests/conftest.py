import pytest
import shutil
import os
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from langchain_core.embeddings import Embeddings

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    monkeypatch.setattr(settings, "vectorstore_path", "./test_vectorstore")
    monkeypatch.setattr(settings, "documents_path", "./test_documents")

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()

@pytest.fixture
def clean_store():
    test_dir = "./test_vectorstore"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists("./test_documents"):
        shutil.rmtree("./test_documents")
    import app.rag.vectorstore
    app.rag.vectorstore._vectorstore = None
    yield
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists("./test_documents"):
        shutil.rmtree("./test_documents")
    app.rag.vectorstore._vectorstore = None

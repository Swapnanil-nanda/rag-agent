import pytest
from unittest.mock import patch
from langchain_core.documents import Document

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@patch("app.api.routes.load_metadata", return_value={"session_id": "test", "title": "New Chat", "filenames": [], "history": []})
@patch("app.api.routes.save_metadata")
@patch("app.rag.loader.load_document", return_value=[Document(page_content="mock content")])
@patch("app.api.routes.add_documents_to_store", return_value=1)
def test_ingest(mock_add, mock_load, mock_save, mock_load_meta, client):
    file_data = {"file": ("test.txt", b"mock text document data", "text/plain")}
    response = client.post("/ingest?session_id=test_session", files=file_data)
    assert response.status_code == 200
    data = response.json()
    assert "ingested" in data["message"].lower()
    assert data["chunks_count"] > 0

@patch("app.api.routes.load_metadata", return_value={"session_id": "test", "title": "New Chat", "filenames": [], "history": []})
@patch("app.api.routes.save_metadata")
@patch("app.api.routes.retrieve_documents", return_value=[Document(page_content="target context", metadata={"source": "doc"})])
@patch("app.api.routes.generate_response", return_value="mocked answer")
def test_query_json(mock_gen, mock_ret, mock_save, mock_load_meta, client):
    payload = {"session_id": "test_session", "question": "what is the answer?", "stream": False}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "mocked answer"
    assert len(data["sources"]) == 1

async def fake_stream(question, docs, history=None):
    yield "word1 "
    yield "word2"

@patch("app.api.routes.load_metadata", return_value={"session_id": "test", "title": "New Chat", "filenames": [], "history": []})
@patch("app.api.routes.save_metadata")
@patch("app.api.routes.retrieve_documents", return_value=[Document(page_content="target context", metadata={"source": "doc"})])
@patch("app.api.routes.generate_stream", side_effect=fake_stream)
def test_query_stream(mock_gen, mock_ret, mock_save, mock_load_meta, client):
    payload = {"session_id": "test_session", "question": "what is the answer?", "stream": True}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    assert "word1" in response.text
    assert "word2" in response.text
    assert "__SOURCES__" in response.text

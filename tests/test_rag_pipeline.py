import pytest
from unittest.mock import patch
from tests.conftest import FakeEmbeddings
from langchain_core.documents import Document
from app.rag.splitter import split_documents
from app.rag.vectorstore import add_documents_to_store, get_vectorstore
from app.rag.retriever import retrieve_documents
from app.rag import retriever

def test_split_documents():
    docs = [Document(page_content="hello world. this is a test. another sentence here.")]
    chunks = split_documents(docs, chunk_size=20, chunk_overlap=0)
    assert len(chunks) > 0

@pytest.mark.asyncio
@patch("app.rag.vectorstore.get_embeddings_model", return_value=FakeEmbeddings())
@patch("app.rag.embeddings.get_embeddings_model", return_value=FakeEmbeddings())
async def test_vectorstore_and_retrieval(mock_embed1, mock_embed2, clean_store):
    docs = [Document(page_content="some financial data is here", metadata={"source": "test"})]
    count = await add_documents_to_store("test_session", docs)
    assert count == 1
    vs = get_vectorstore("test_session")
    assert vs is not None
    results = retrieve_documents("test_session", "financial", k=1)
    assert len(results) == 1
    assert results[0].page_content == "some financial data is here"


def test_summary_retrieval_balances_sources(monkeypatch):
    """Overview queries should represent each uploaded document when possible."""
    documents = {
        "a1": Document(page_content="First file page one", metadata={"source": "first.pdf", "page": 0}),
        "a2": Document(page_content="First file page two", metadata={"source": "first.pdf", "page": 1}),
        "b1": Document(page_content="Second file page one", metadata={"source": "second.pdf", "page": 0}),
    }
    class Store:
        docstore = type("DocStore", (), {"_dict": documents})()
    monkeypatch.setattr("app.rag.vectorstore.get_vectorstore", lambda _session: Store())
    result = retriever.retrieve_documents("session", "summary of these documents", k=2)
    assert {document.metadata["source"] for document in result} == {"first.pdf", "second.pdf"}

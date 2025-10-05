import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

from ms_graphrag_neo4j.api import app, get_ms_graph_rag

@pytest.fixture
def mock_rag_instance():
    """
    Creates a fresh MagicMock for MsGraphRAG for each test,
    ensuring test isolation.
    """
    mock = MagicMock()
    # search_chunks is a regular async function, so AsyncMock is correct.
    mock.search_chunks = AsyncMock()
    # astream_run is an async generator. We mock it with a regular MagicMock
    # that will return an async generator object when called.
    mock.astream_run = MagicMock()
    return mock

@pytest.fixture
def client(mock_rag_instance):
    """
    Provides a TestClient with the MsGraphRAG dependency overridden.
    This fixture ensures that all API calls made by the client
    are routed to the mock_rag_instance instead of a real
    MsGraphRAG object.
    """
    app.dependency_overrides[get_ms_graph_rag] = lambda: mock_rag_instance
    yield TestClient(app)
    # Clear the override after the test is done
    app.dependency_overrides.clear()

def test_get_document_count(client, mock_rag_instance):
    """
    Tests the /document_count endpoint to ensure it returns a successful response
    with the expected count from the mocked backend.
    """
    mock_rag_instance.query.return_value = [{"count": 123}]
    response = client.get("/document_count")
    assert response.status_code == 200
    assert response.json() == {"count": 123}
    mock_rag_instance.query.assert_called_once_with("MATCH (d:Document) RETURN count(d) AS count")

def test_get_authors_count(client, mock_rag_instance):
    """
    Tests the /authors_count endpoint to ensure it returns a successful response
    with the expected count from the mocked backend.
    """
    mock_rag_instance.query.return_value = [{"count": 45}]
    response = client.get("/authors_count")
    assert response.status_code == 200
    assert response.json() == {"count": 45}
    mock_rag_instance.query.assert_called_once_with("MATCH (a:Author) RETURN count(a) AS count")

def test_stream_rag_prompt(client, mock_rag_instance):
    """
    Tests the /prompt streaming endpoint to ensure it correctly handles
    the streaming response from the mocked RAG pipeline.
    """
    async def mock_stream_generator():
        yield "This "
        yield "is "
        yield "a "
        yield "streamed "
        yield "response."

    # The mocked astream_run function returns our async generator
    mock_rag_instance.astream_run.return_value = mock_stream_generator()
    response = client.post("/prompt", json={"prompt": "What is GraphRAG?"})
    assert response.status_code == 200
    assert response.text == "This is a streamed response."
    mock_rag_instance.astream_run.assert_called_once_with("What is GraphRAG?")

def test_get_gaps(client, mock_rag_instance):
    """
    Tests the /gaps endpoint to ensure it returns a list of documents
    with no supporting or contradicting claims.
    """
    mock_rag_instance.query.return_value = [
        {"document_title": "Document A"},
        {"document_title": "Document B"},
    ]
    response = client.get("/gaps")
    assert response.status_code == 200
    assert response.json() == [
        {"document_title": "Document A"},
        {"document_title": "Document B"},
    ]
    mock_rag_instance.query.assert_called_once_with(
        "MATCH (d:Document) WHERE NOT (d)-[:SUPPORTS|:CONTRADICTS]-() RETURN d.title AS document_title"
    )

def test_search_documents(client, mock_rag_instance):
    """
    Tests the /search endpoint to ensure it returns documents based on a query.
    """
    mock_rag_instance.search_chunks.return_value = [
        {"id": "chunk1"},
        {"id": "chunk2"},
    ]
    mock_rag_instance.query.return_value = [
        {"d": {"title": "Document Alpha"}},
        {"d": {"title": "Document Beta"}},
    ]

    response = client.post("/search", json={"query": "machine learning"})

    assert response.status_code == 200
    assert response.json() == [
        {"d": {"title": "Document Alpha"}},
        {"d": {"title": "Document Beta"}},
    ]
    mock_rag_instance.search_chunks.assert_called_once_with("machine learning", top_k=5)
    mock_rag_instance.query.assert_called_once_with(
        """
        UNWIND $chunk_ids AS c_id
        MATCH (c:__Chunk__ {id: c_id})-[:FROM_DOCUMENT]->(d:Document)
        RETURN DISTINCT d
        """,
        params={"chunk_ids": ["chunk1", "chunk2"]},
    )
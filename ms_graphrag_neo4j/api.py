import os
import asyncio
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from neo4j import GraphDatabase
from .ms_graphrag import MsGraphRAG
from pydantic import BaseModel

load_dotenv()

# --- Pydantic Models ---
class PromptRequest(BaseModel):
    """Request model for the prompt endpoint."""
    prompt: str

class SearchRequest(BaseModel):
    """Request model for the search endpoint."""
    query: str

# --- FastAPI App Instance ---
app = FastAPI()

# --- Dependency ---
def get_ms_graph_rag():
    """
    Dependency function that creates and yields an MsGraphRAG instance.
    This ensures that the Neo4j driver is properly managed and closed.
    """
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    try:
        ms_graph = MsGraphRAG(driver=driver, model='gpt-4o')
        yield ms_graph
    finally:
        driver.close()

# --- API Endpoints ---
# --- Helper Function for Document Formatting ---
def _get_formatted_document(ms_graph: MsGraphRAG, doc_id: int = None, title: str = None):
    """
    Internal helper to fetch and format a single document with its metadata,
    authors, chunks, and related documents.
    """
    if doc_id is None and title is None:
        return None

    # A single, efficient query to gather all document details
    query = """
    MATCH (d:Document)
    WHERE id(d) = $doc_id OR d.title = $title
    // Collect authors
    OPTIONAL MATCH (d)-[:AUTHORED_BY]->(a:Author)
    // Collect chunks
    OPTIONAL MATCH (d)<-[:FROM_DOCUMENT]-(c:__Chunk__)
    // Collect related documents
    OPTIONAL MATCH (d)-[r:SUPPORTS|CONTRADICTS]->(d2:Document)
    WITH d,
         collect(DISTINCT a.name) AS authors,
         collect(DISTINCT c {.*}) AS chunks,
         collect(DISTINCT {relationship: type(r), title: d2.title}) AS similar_docs
    RETURN d {.*} AS metadata, authors, chunks, similar_docs
    LIMIT 1
    """
    
    result = ms_graph.query(query, params={"doc_id": doc_id, "title": title})
    
    if not result:
        return None
        
    data = result[0]
    
    # Structure the final response
    formatted_response = {
        "metadata": {
            "title": data['metadata'].get('title'),
            "date": data['metadata'].get('date'), # Assumes 'date' is a property on the node
            "authors": data.get('authors', []),
            "similar_documents": data.get('similar_docs', [])
        },
        "chunks": data.get('chunks', [])
    }
    
    return formatted_response


@app.get("/document_count", summary="Get the total number of documents")
async def get_document_count(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns the total count of `Document` nodes in the graph."""
    result = ms_graph.query("MATCH (d:Document) RETURN count(d) AS count")
    return result[0] if result else {"count": 0}

@app.get("/authors_count", summary="Get the total number of authors")
async def get_authors_count(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns the total count of `Author` nodes in the graph."""
    result = ms_graph.query("MATCH (a:Author) RETURN count(a) AS count")
    return result[0] if result else {"count": 0}

@app.get("/document_by_id/{doc_id}", summary="Get a document by its ID")
async def get_document_by_id(doc_id: int, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns a single `Document` node by its Neo4j internal ID."""
    result = ms_graph.query("MATCH (d:Document) WHERE id(d) = $doc_id RETURN d", params={"doc_id": doc_id})
    return result[0] if result else {}

@app.get("/document_by_title/{title}", summary="Get a document by its title")
async def get_document_by_title(title: str, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns a single `Document` node by its title."""
    result = ms_graph.query("MATCH (d:Document {title: $title}) RETURN d", params={"title": title})
    return result[0] if result else {}

@app.get("/graph_sample", summary="Get a sample of the graph")
async def get_graph_sample(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns a small sample of the graph, including nodes and relationships."""
    return ms_graph.query("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")

@app.get("/contradicting_claims", summary="Get documents with contradicting claims")
async def get_contradicting_claims(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns pairs of documents that are linked by a `CONTRADICTS` relationship."""
    return ms_graph.query("MATCH (d1:Document)-[:CONTRADICTS]->(d2:Document) RETURN d1.title AS doc1, d2.title AS doc2")

@app.get("/supporting_claims", summary="Get documents with supporting claims")
async def get_supporting_claims(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns pairs of documents that are linked by a `SUPPORTS` relationship."""
    return ms_graph.query("MATCH (d1:Document)-[:SUPPORTS]->(d2:Document) RETURN d1.title AS doc1, d2.title AS doc2")

@app.get("/gaps", summary="Get documents with no claims")
async def get_gaps(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns documents that have no `SUPPORTS` or `CONTRADICTS` relationships."""
    return ms_graph.query("MATCH (d:Document) WHERE NOT (d)-[:SUPPORTS|:CONTRADICTS]-() RETURN d.title AS document_title")

@app.post("/search", summary="Search for documents by query")
async def search(request: SearchRequest, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Accepts a search query and returns a list of documents
    containing similar text chunks.
    """
    chunks = await ms_graph.search_chunks(request.query, top_k=5)
    if not chunks:
        return []

    chunk_ids = [c["id"] for c in chunks]

    documents = ms_graph.query(
        """
        UNWIND $chunk_ids AS c_id
        MATCH (c:__Chunk__ {id: c_id})-[:FROM_DOCUMENT]->(d:Document)
        RETURN DISTINCT d
        """,
        params={"chunk_ids": chunk_ids},
    )
    return documents@app.get("/document_count", summary="Get the total number of documents")
async def get_document_count(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns the total count of `Document` nodes in the graph."""
    result = ms_graph.query("MATCH (d:Document) RETURN count(d) AS count")
    return result[0] if result else {"count": 0}

@app.get("/authors_count", summary="Get the total number of authors")
async def get_authors_count(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns the total count of `Author` nodes in the graph."""
    result = ms_graph.query("MATCH (a:Author) RETURN count(a) AS count")
    return result[0] if result else {"count": 0}

@app.get("/document_by_id/{doc_id}", summary="Get a document by its ID")
async def get_document_by_id(doc_id: int, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Returns a single `Document` node by its Neo4j internal ID, formatted with
    its full metadata, chunks, and related documents.
    """
    document = _get_formatted_document(ms_graph, doc_id=doc_id)
    return document if document else {}

@app.get("/document_by_title/{title}", summary="Get a document by its title")
async def get_document_by_title(title: str, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Returns a single `Document` node by its title, formatted with
    its full metadata, chunks, and related documents.
    """
    document = _get_formatted_document(ms_graph, title=title)
    return document if document else {}

@app.get("/graph_sample", summary="Get a sample of the graph")
async def get_graph_sample(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """Returns a small sample of the graph, including nodes and relationships."""
    return ms_graph.query("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")

@app.get("/contradicting_claims", summary="Get documents with contradicting claims")
async def get_contradicting_claims(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Identifies pairs of documents with 'CONTRADICTS' relationships and returns
    their metadata.
    """
    results = ms_graph.query("""
        MATCH (d1:Document)-[:CONTRADICTS]->(d2:Document)
        RETURN d1 {.*} AS doc1, d2 {.*} AS doc2
    """)
    
    unique_docs = {}
    for record in results:
        unique_docs[record['doc1']['title']] = record['doc1']
        unique_docs[record['doc2']['title']] = record['doc2']
        
    return {
        "title": "Contradicting Claims Analysis",
        "description": "This report lists documents that have been identified as contradicting one another.",
        "documents": list(unique_docs.values())
    }

@app.get("/supporting_claims", summary="Get documents with supporting claims")
async def get_supporting_claims(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Identifies pairs of documents with 'SUPPORTS' relationships and returns
    their metadata.
    """
    results = ms_graph.query("""
        MATCH (d1:Document)-[:SUPPORTS]->(d2:Document)
        RETURN d1 {.*} AS doc1, d2 {.*} AS doc2
    """)
    
    unique_docs = {}
    for record in results:
        unique_docs[record['doc1']['title']] = record['doc1']
        unique_docs[record['doc2']['title']] = record['doc2']
        
    return {
        "title": "Supporting Claims Analysis",
        "description": "This report lists documents that have been identified as supporting one another.",
        "documents": list(unique_docs.values())
    }

@app.get("/gaps", summary="Get documents with no claims")
async def get_gaps(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Finds documents that are isolated from the claim analysis (have no 'SUPPORTS'
    or 'CONTRADICTS' relationships) and returns their metadata.
    """
    results = ms_graph.query("""
        MATCH (d:Document)
        WHERE NOT (d)-[:SUPPORTS|:CONTRADICTS]-() AND NOT ()-[:SUPPORTS|:CONTRADICTS]->(d)
        RETURN d {.*} AS document
    """)
    
    return {
        "title": "Knowledge Gap Analysis",
        "description": "This report lists documents that are not connected to others through supporting or contradicting claims, potentially indicating a knowledge gap.",
        "documents": [record['document'] for record in results]
    }

@app.post("/search", summary="Search for documents by query")
async def search(request: SearchRequest, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Accepts a search query, finds documents with similar text chunks,
    and returns a list of those documents in the detailed format.
    """
    chunks = await ms_graph.search_chunks(request.query, top_k=5)
    if not chunks:
        return []

    chunk_ids = [c["id"] for c in chunks]

    # Get the unique titles of the documents found
    doc_titles = ms_graph.query(
        """
        UNWIND $chunk_ids AS c_id
        MATCH (:__Chunk__ {id: c_id})-[:FROM_DOCUMENT]->(d:Document)
        RETURN DISTINCT d.title AS title
        """,
        params={"chunk_ids": chunk_ids},
    )

    # Use the helper to fetch and format each unique document
    formatted_documents = []
    for record in doc_titles:
        doc_data = _get_formatted_document(ms_graph, title=record['title'])
        if doc_data:
            formatted_documents.append(doc_data)
            
    return formatted_documents

@app.post("/prompt", summary="Submit a prompt for a streaming RAG response")
async def stream_rag_prompt(request: PromptRequest, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    """
    Accepts a prompt and streams a response from the GraphRAG pipeline.
    """
    return StreamingResponse(ms_graph.astream_run(request.prompt))
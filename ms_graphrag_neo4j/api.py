import os
import asyncio
import logging
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from neo4j import GraphDatabase
from .ms_graphrag import MsGraphRAG
from pydantic import BaseModel

# --- Load environment variables ---
load_dotenv()

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("GraphRAGAPI")

# --- Pydantic Models ---
class PromptRequest(BaseModel):
    prompt: str

class SearchRequest(BaseModel):
    query: str

# --- FastAPI App Instance ---
app = FastAPI(title="GraphRAG API", description="Neo4j Graph + RAG integration")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 GraphRAG API starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 GraphRAG API shutting down")

# --- Dependency ---
def get_ms_graph_rag():
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    try:
        logger.debug("Initializing MsGraphRAG with Neo4j driver")
        ms_graph = MsGraphRAG(driver=driver, model='gpt-4o')
        yield ms_graph
    finally:
        logger.debug("Closing Neo4j driver")
        driver.close()

# --- Helper Function ---
def _get_formatted_document(ms_graph: MsGraphRAG, doc_id: int = None, title: str = None):
    if doc_id is None and title is None:
        logger.warning("_get_formatted_document called without doc_id or title")
        return None

    logger.info(f"Fetching formatted document for "
                f"{'id=' + str(doc_id) if doc_id else 'title=' + title}")

    query = """
    MATCH (d:Document)
    WHERE id(d) = $doc_id OR d.title = $title
    OPTIONAL MATCH (d)-[:AUTHORED_BY]->(a:Author)
    OPTIONAL MATCH (d)<-[:FROM_DOCUMENT]-(c:__Chunk__)
    OPTIONAL MATCH (d)-[r:SUPPORTS|CONTRADICTS]->(d2:Document)
    WITH d,
         collect(DISTINCT a.name) AS authors,
         collect(DISTINCT c {.*}) AS chunks,
         collect(DISTINCT {relationship: type(r), title: d2.title}) AS similar_docs
    RETURN d {.*} AS metadata, authors, chunks, similar_docs
    LIMIT 1
    """
    
    try:
        result = ms_graph.query(query, params={"doc_id": doc_id, "title": title})
    except Exception as e:
        logger.exception(f"Error fetching formatted document: {e}")
        return None

    if not result:
        logger.info("No document found for given parameters")
        return None

    data = result[0]
    formatted_response = {
        "metadata": {
            "title": data['metadata'].get('title'),
            "date": data['metadata'].get('date'),
            "authors": data.get('authors', []),
            "similar_documents": data.get('similar_docs', [])
        },
        "chunks": data.get('chunks', [])
    }
    
    logger.debug(f"Formatted document fetched: {formatted_response['metadata']['title']}")
    return formatted_response

# --- API Endpoints ---
@app.get("/document_count")
async def get_document_count(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info("GET /document_count")
    result = ms_graph.query("MATCH (d:Document) RETURN count(d) AS count")
    return result[0] if result else {"count": 0}

@app.get("/authors_count")
async def get_authors_count(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info("GET /authors_count")
    result = ms_graph.query("MATCH (a:Author) RETURN count(a) AS count")
    return result[0] if result else {"count": 0}

@app.get("/document_by_id/{doc_id}")
async def get_document_by_id(doc_id: int, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info(f"GET /document_by_id/{doc_id}")
    return _get_formatted_document(ms_graph, doc_id=doc_id) or {}

@app.get("/document_by_title/{title}")
async def get_document_by_title(title: str, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info(f"GET /document_by_title/{title}")
    return _get_formatted_document(ms_graph, title=title) or {}

@app.get("/graph_sample")
async def get_graph_sample(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info("GET /graph_sample")
    return ms_graph.query("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")

@app.get("/contradicting_claims")
async def get_contradicting_claims(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info("GET /contradicting_claims")
    results = ms_graph.query("""
        MATCH (d1:Document)-[:CONTRADICTS]->(d2:Document)
        RETURN d1 {.*} AS doc1, d2 {.*} AS doc2
    """)
    logger.debug(f"Found {len(results)} contradicting pairs")
    unique_docs = {r['doc1']['title']: r['doc1'] for r in results}
    unique_docs.update({r['doc2']['title']: r['doc2'] for r in results})
    return {
        "title": "Contradicting Claims Analysis",
        "documents": list(unique_docs.values())
    }

@app.get("/supporting_claims")
async def get_supporting_claims(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info("GET /supporting_claims")
    results = ms_graph.query("""
        MATCH (d1:Document)-[:SUPPORTS]->(d2:Document)
        RETURN d1 {.*} AS doc1, d2 {.*} AS doc2
    """)
    logger.debug(f"Found {len(results)} supporting pairs")
    unique_docs = {r['doc1']['title']: r['doc1'] for r in results}
    unique_docs.update({r['doc2']['title']: r['doc2'] for r in results})
    return {
        "title": "Supporting Claims Analysis",
        "documents": list(unique_docs.values())
    }

@app.get("/gaps")
async def get_gaps(ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info("GET /gaps")
    results = ms_graph.query("""
        MATCH (d:Document)
        WHERE NOT (d)-[:SUPPORTS|:CONTRADICTS]-() AND NOT ()-[:SUPPORTS|:CONTRADICTS]->(d)
        RETURN d {.*} AS document
    """)
    logger.debug(f"Found {len(results)} documents with no claims")
    return {
        "title": "Knowledge Gap Analysis",
        "documents": [r['document'] for r in results]
    }

@app.post("/search")
async def search(request: SearchRequest, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info(f"POST /search query='{request.query}'")
    chunks = await ms_graph.search_chunks(request.query, top_k=5)
    if not chunks:
        logger.info("No matching chunks found")
        return []

    chunk_ids = [c["id"] for c in chunks]
    doc_titles = ms_graph.query(
        """
        UNWIND $chunk_ids AS c_id
        MATCH (:__Chunk__ {id: c_id})-[:FROM_DOCUMENT]->(d:Document)
        RETURN DISTINCT d.title AS title
        """,
        params={"chunk_ids": chunk_ids},
    )

    formatted_documents = []
    for record in doc_titles:
        doc_data = _get_formatted_document(ms_graph, title=record['title'])
        if doc_data:
            formatted_documents.append(doc_data)
    logger.debug(f"Returning {len(formatted_documents)} documents for search query")
    return formatted_documents

@app.post("/prompt")
async def stream_rag_prompt(request: PromptRequest, ms_graph: MsGraphRAG = Depends(get_ms_graph_rag)):
    logger.info(f"POST /prompt prompt='{request.prompt[:50]}...'")
    try:
        return StreamingResponse(ms_graph.astream_run(request.prompt))
    except Exception as e:
        logger.exception(f"Error during streaming RAG response: {e}")
        raise

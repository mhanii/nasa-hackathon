import os
from typing import Any, Dict, List, Optional, Type
from neo4j import Driver
from openai import AzureOpenAI, AsyncAzureOpenAI
import asyncio
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph

from ms_graphrag_neo4j.providers.gemini import GeminiEmbeddings
from tqdm.asyncio import tqdm, tqdm_asyncio
from graphdatascience import GraphDataScience


from ms_graphrag_neo4j.cypher_queries import *
from ms_graphrag_neo4j.utils import *
from ms_graphrag_neo4j.prompts import *
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from transformers import pipeline
import torch # To check for GPU availability


device = 0 if torch.cuda.is_available() else -1

class MsGraphRAG:
    """
    MsGraphRAG: Microsoft GraphRAG Implementation for Neo4j

    A class for implementing the Microsoft GraphRAG approach with Neo4j graph database.
    GraphRAG enhances retrieval-augmented generation by leveraging graph structures
    to provide context-aware information for LLM responses.

    This implementation features:
    - Entity and relationship extraction from unstructured text
    - Node and relationship summarization for improved retrieval
    - Community detection and summarization for concept clustering
    - Integration with OpenAI models for generation
    - Semantic search capability over document chunks, document summaries, and community summaries.
    - Creation of Document and Author nodes for rich knowledge graph structure.
    """

    def __init__(
        self,
        driver: Driver,
        model: str = "gpt-4o",
        database: str = "neo4j",
        max_workers: int = 10,
        create_constraints: bool = True,
    ) -> None:
        """
        Initialize MsGraphRAG with Neo4j driver and LLM.

        Args:
            driver (Driver): Neo4j driver instance
            model (str, optional): The language model to use. Defaults to "gpt-4o".
            database (str, optional): Neo4j database name. Defaults to "neo4j".
            max_workers (int, optional): Maximum number of concurrent workers. Defaults to 10.
            create_constraints (bool, optional): Whether to create database constraints and indexes. Defaults to True.
        """


        self._driver = driver
        self.model = model
        self.max_workers = max_workers
        self._database = database
        self.embeddings = GeminiEmbeddings()
        self._openai_client = AsyncAzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="https://gpt5-api-resource.cognitiveservices.azure.com/",
            api_key=os.environ.get("AZURE_API_KEY"),
        )
        # Test for APOC
        try:
            self.query("CALL apoc.help('test')")
        except:
            raise ValueError("You need to install and allow APOC functions")
        # Test for GDS
        try:
            self.query("CALL gds.list('test')")
        except:
            raise ValueError("You need to install and allow GDS functions")
        if create_constraints:
            # Constraints for uniqueness
            self.query("CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Chunk__) REQUIRE e.id IS UNIQUE;")
            self.query("CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.name IS UNIQUE;")
            self.query("CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Community__) REQUIRE e.id IS UNIQUE;")
            self.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS UNIQUE;")
            self.query("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE;")

            # Vector index for searching text chunks
            self.query("""
                CREATE VECTOR INDEX `chunk_text_embeddings` IF NOT EXISTS
                FOR (c:__Chunk__) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768, `vector.similarity_function`: 'cosine'
                }}
            """)
            # NEW: Vector index for searching document summaries
            self.query("""
                CREATE VECTOR INDEX `document_summary_embeddings` IF NOT EXISTS
                FOR (d:Document) ON (d.summary_embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768, `vector.similarity_function`: 'cosine'
                }}
            """)
            # NEW: Vector index for searching community summaries
            self.query("""
                CREATE VECTOR INDEX `community_summary_embeddings` IF NOT EXISTS
                FOR (c:__Community__) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768, `vector.similarity_function`: 'cosine'
                }}
            """)

    # ... (Keep extract_nodes_and_rels, summarize_nodes_and_rels, summarize_document, entity_resolution methods as they are) ...
    async def extract_nodes_and_rels(
        self, input_texts: list, allowed_entities: list, chunk_ids: list
    ) -> str:
        """
        Extract nodes and relationships from input texts using LLM and store them in Neo4j.

        Args:
            input_texts (list): List of text documents to process and extract entities from
            allowed_entities (list): List of entity types to extract from the texts
            chunk_ids (list): List of chunk IDs corresponding to each input text

        Returns:
            str: Success message with count of extracted relationships

        Notes:
            - Uses parallel processing with tqdm progress tracking
            - Extracted entities and relationships are stored directly in Neo4j
            - Each text document is processed independently by the LLM
        """

        async def process_text(input_text):
            prompt = GRAPH_EXTRACTION_PROMPT.format(
                entity_types=allowed_entities,
                input_text=input_text,
                tuple_delimiter=";",
                record_delimiter="|",
                completion_delimiter="\n\n",
            )
            messages = [
                {"role": "user", "content": prompt},
            ]
            # Make the LLM call
            output = await self.achat(messages, model=self.model)
            # Construct JSON from output
            return parse_extraction_output(output.content)

        # Create tasks for all input texts
        tasks = [process_text(text) for text in input_texts]

        # Process tasks with tqdm progress bar
        # Use semaphore to limit concurrent tasks if max_workers is specified
        if self.max_workers:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(task):
                async with semaphore:
                    return await task

            results = []
            for task in tqdm.as_completed(
                [process_with_semaphore(task) for task in tasks],
                total=len(tasks),
                desc="Extracting nodes & relationships",
            ):
                results.append(await task)
        else:
            results = []
            for task in tqdm.as_completed(
                tasks, total=len(tasks), desc="Extracting nodes & relationships"
            ):
                results.append(await task)

        total_relationships = 0
        # Import nodes and relationships
        for text, chunk_id, output in zip(input_texts, chunk_ids, results):
            nodes, relationships = output
            total_relationships += len(relationships)
            # Import nodes
            self.query(
                import_nodes_query,
                params={"text": text, "chunk_id": chunk_id, "data": nodes, "rel" :"MENTIONS"},
            )
            # Import relationships
            self.query(import_relationships_query, params={"data": relationships})

        return f"Successfuly extracted and imported {total_relationships} relationships"

    async def summarize_nodes_and_rels(self) -> str:
        """
        Generate summaries for all nodes and relationships in the graph.

        Returns:
            str: Success message indicating completion of summarization

        Notes:
            - Retrieves candidate nodes and relationships from Neo4j
            - Uses LLM to generate concise summaries for each entity and relationship
            - Stores summarized properties in the graph
        """
        # Summarize nodes
        nodes = self.query(candidate_nodes_summarization)

        async def process_node(node):
            messages = [
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(
                        entity_name=node["entity_name"],
                        description_list=node["description_list"],
                    ),
                },
            ]
            summary = await self.achat(messages, model=self.model)
            return {"entity": node["entity_name"], "summary": summary.content}

        # Create a progress bar for node processing with max_workers limit
        if self.max_workers:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(node):
                async with semaphore:
                    return await process_node(node)

            summaries = await tqdm_asyncio.gather(
                *[process_with_semaphore(node) for node in nodes],
                desc="Summarizing nodes",
            )
        else:
            summaries = await tqdm_asyncio.gather(
                *[process_node(node) for node in nodes], desc="Summarizing nodes"
            )

        # Summarize relationships
        rels = self.query(candidate_rels_summarization)

        async def process_rel(rel):
            entity_name = f"{rel['source']} relationship to {rel['target']}"
            messages = [
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(
                        entity_name=entity_name,
                        description_list=rel["description_list"],
                    ),
                },
            ]
            summary = await self.achat(messages, model=self.model)
            return {
                "source": rel["source"],
                "target": rel["target"],
                "summary": summary.content,
            }

        # Create a progress bar for relationship processing with max_workers limit
        if self.max_workers:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_rel_with_semaphore(rel):
                async with semaphore:
                    return await process_rel(rel)

            rel_summaries = await tqdm_asyncio.gather(
                *[process_rel_with_semaphore(rel) for rel in rels],
                desc="Summarizing relationships",
            )
        else:
            rel_summaries = await tqdm_asyncio.gather(
                *[process_rel(rel) for rel in rels], desc="Summarizing relationships"
            )

        # Import nodes
        self.query(import_entity_summary, params={"data": summaries})
        self.query(import_entity_summary_single)

        # Import relationships
        self.query(import_rel_summary, params={"data": rel_summaries})
        self.query(import_rel_summary_single)

        return "Successfuly summarized nodes and relationships"



    async def summarize_document(self,metadata,document):

        messages = [
                {
                    "role": "user",
                    "content": DOCUMENT_SUMMARY_PROMPT.format(
                        metadata=metadata,document = document
                    ),
                },
            ]
        
        summary = await self.achat(messages, model=self.model)

        return summary.content
    
    
    async def entity_relation_extraction_from_summaries(
        self, top_k: int = 5, similarity_threshold: float = 0.8, use_nli: bool = False
    ) -> str:
        """
        Extracts relationships between documents based on their summaries.
        ...
        """
        if use_nli:
            nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Get all document summaries
        documents = self.query("MATCH (d:Document) RETURN d.title AS title, d.summary AS summary")

        async def process_document(document):
            # Find similar documents
            similar_documents = await self.search_document_summaries(
                query_text=document["summary"], top_k=top_k
            )

            relationships = []
            for similar_document in similar_documents:
                if similar_document["score"] > similarity_threshold:
                    if use_nli:
                        # Use NLI model to determine relationship
                        sequence_to_classify = similar_document["summary"]
                        candidate_labels = ["entailment", "contradiction", "neutral"]
                        result = nli_model(sequence_to_classify, candidate_labels)
                        relationship = result["labels"][0]
                    else:
                        # Use LLM to determine relationship
                        messages = [
                            {
                                "role": "user",
                                "content": f"Document 1: {document['summary']}\n\nDocument 2: {similar_document['summary']}\n\nWhat is the relationship between these two documents? (e.g., 'supports', 'contradicts', 'expands on', 'is related to')",
                            },
                        ]
                        response = await self.achat(messages, model=self.model)
                        relationship = response.content.strip()

                    relationships.append(
                        {
                            "source": document["title"],
                            "target": similar_document["title"],
                            "type": relationship,
                        }
                    )
            return relationships

        all_relationships = []
        for document in documents:
            all_relationships.extend(await process_document(document))

        # Create relationships in the graph
        self.query(
            """
            UNWIND $relationships AS rel
            MATCH (d1:Document {title: rel.source})
            MATCH (d2:Document {title: rel.target})
            MERGE (d1)-[r:RELATED_TO {type: rel.type}]->(d2)
            """,
            params={"relationships": all_relationships},
        )

        return f"Created {len(all_relationships)} relationships between documents."    
    async def entity_resolution(self):
        vector = Neo4jVector.from_existing_graph(
            self.embeddings,
            node_label='__Entity__',
            text_node_properties=['id', 'description'],
            embedding_node_property='embedding'
        )

        gds = GraphDataScience(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
                
        )

        G, result = gds.graph.project(
            "entities",                   # Graph name
            "__Entity__",                 # Node projection
            "*",                          # Relationship projection
            nodeProperties=["embedding"]  # Configuration parameters
        )
        similarity_threshold = 0.95

        gds.knn.mutate(
            G,
            nodeProperties=['embedding'],
            mutateRelationshipType= 'SIMILAR',
            mutateProperty= 'score',
            similarityCutoff=similarity_threshold
        )

        gds.wcc.write(
            G,
            writeProperty="wcc",
            relationshipTypes=["SIMILAR"]
        )

        word_edit_distance = 3
        potential_duplicate_candidates = self.query(query=
            """MATCH (e:`__Entity__`)
            WHERE size(e.id) > 3 // longer than 3 characters
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // Add text distance
            WITH distinct
            [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance 
                        OR node.id CONTAINS n.id | n.id] AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // combine groups together if they share elements
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                    CASE WHEN index <> index2 AND
                        size(apoc.coll.intersection(acc, results[index2])) > 0
                        THEN apoc.coll.union(acc, results[index2])
                        ELSE acc
                    END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // extra filtering
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """, params={'distance': word_edit_distance})
        print(potential_duplicate_candidates)


    async def summarize_communities(self, summarize_all_levels: bool = False) -> str:
        """
        Detect and summarize communities within the graph using the Leiden algorithm.
        This now includes generating and storing a title and embedding for each summary.

        Args:
            summarize_all_levels (bool, optional): Whether to summarize all community levels
                or just the final level. Defaults to False.

        Returns:
            str: Success message with count of generated community summaries
        """
        # Calculate communities
        self.query(drop_gds_graph_query)
        self.query(create_gds_graph_query)
        community_summary_result = self.query(leiden_query)
        community_levels = community_summary_result[0]["ranLevels"]
        print(
            f"Leiden algorithm identified {community_levels} community levels "
            f"with {community_summary_result[0]['communityCount']} communities on the last level."
        )
        self.query(community_hierarchy_query)

        # Community summarization
        if summarize_all_levels:
            levels = list(range(community_levels))
        else:
            levels = [community_levels - 1]
        communities = self.query(community_info_query, params={"levels": levels})

        # Define async function for processing a single community
        async def process_community(community):
            input_text = f"""Entities:
                    {community['nodes']}

                    Relationships:
                    {community['rels']}"""

            messages = [
                {
                    "role": "user",
                    "content": COMMUNITY_REPORT_PROMPT.format(input_text=input_text),
                },
            ]
            summary = await self.achat(messages, model=self.model)
            
            summary_json = extract_json(summary.content)
            
            # --- CHANGE STARTS HERE ---
            # Explicitly pull all expected primitive values, now including the title.
            title_text = summary_json.get('title', 'Untitled Community')
            summary_text = summary_json.get('summary', '')
            explanation_text = summary_json.get('explanation', '')
            
            summary_embedding = self.embeddings.embed_query(summary_text) if summary_text else None
            
            # We now separate the communityId (for matching) from the properties we want to set.
            community_id = community["communityId"]
            
            properties_to_set = {
                'title': title_text,
                'summary': summary_text,
                'explanation': explanation_text,
                'embedding': summary_embedding
            }
            
            return { "communityId": community_id, "properties": properties_to_set }
            # --- CHANGE ENDS HERE ---

        if self.max_workers:
            semaphore = asyncio.Semaphore(self.max_workers)
            async def process_community_with_semaphore(community):
                async with semaphore:
                    return await process_community(community)
            community_summaries = await tqdm_asyncio.gather(
                *(process_community_with_semaphore(c) for c in communities),
                desc="Summarizing communities", total=len(communities)
            )
        else:
            community_summaries = await tqdm_asyncio.gather(
                *(process_community(c) for c in communities),
                desc="Summarizing communities", total=len(communities)
            )
        
        # --- CHANGE STARTS HERE ---
        # This Cypher query is now cleaner. It uses 'd.communityId' for the MATCH
        # and sets the properties from the nested 'd.properties' map.
        import_community_summary_with_embedding = """
        UNWIND $data AS d
        MATCH (c:__Community__ {id: d.communityId})
        SET c += d.properties
        """
        # --- CHANGE ENDS HERE ---

        self.query(import_community_summary_with_embedding, params={"data": community_summaries})
        return f"Generated {len(community_summaries)} community summaries"


    async def run(self, document, allowed_entities):
        """
        Process a single document, create a graph structure, and run the full GraphRAG pipeline.
        The document metadata should include a 'title' (str) and 'authors' (list of str).

        Args:
            document (dict): A dictionary representing the document with 'metadata' and 'chunks'.
            allowed_entities (list): A list of entity types to extract (e.g., ["Person", "Organization"]).
        """
        doc_metadata = document.get("metadata", {})
        doc_title = doc_metadata.get("title", "Untitled Document")
        doc_authors = doc_metadata.get("authors", [])
        doc_content = document.get("chunks", [])
        
        # Step 1: Summarize and embed the document summary
        print(f"Summarizing document: {doc_title}")
        doc_summary = await self.summarize_document(metadata=doc_metadata, document=doc_content)
        doc_summary_embedding = self.embeddings.embed_query(doc_summary)
        doc_metadata["summary"] = doc_summary
        
        # Create Document and Author nodes and connect them
        self.query(
            """
            MERGE (d:Document {title: $title})
            SET d += $props, d.summary_embedding = $embedding
            WITH d
            UNWIND $authors AS author_name
            MERGE (a:Author {name: author_name})
            MERGE (d)-[:AUTHORED_BY]->(a)
            """,
            params={
                "title": doc_title, 
                "props": doc_metadata, 
                "embedding": doc_summary_embedding,
                "authors": doc_authors
            }
        )
        print(f"Created document node for '{doc_title}' and connected {len(doc_authors)} author(s).")

        # --- REVISED LOGIC FOR ROBUST CHUNK CREATION ---

        # Step 2: Prepare all chunk data in a list first
        all_chunks_text = []
        all_chunk_ids = []
        all_chunks_data = [] # This list will be sent to the database

        for chunk in doc_content:
            chunk_id = get_hash(chunk.text)
            all_chunks_text.append(chunk.text)
            all_chunk_ids.append(chunk_id)
            all_chunks_data.append({
                "chunk_id": chunk_id,
                "text": chunk.text,
                "section": chunk.metadata.get("section", "unknown"),
            })

        # Step 3: Create all chunk nodes and link them to the document in ONE batch query
        print(f"Creating and linking {len(all_chunks_data)} chunk nodes in a single batch...")
        self.query(
            """
            MATCH (d:Document {title: $doc_title})
            UNWIND $chunks AS chunk_data
            MERGE (c:__Chunk__ {id: chunk_data.chunk_id})
            SET c.text = chunk_data.text, c.title = chunk_data.section
            WITH d, c
            MERGE (c)-[:FROM_DOCUMENT]->(d)
            """,
            params={
                "doc_title": doc_title,
                "chunks": all_chunks_data
            }
        )

        # --- END OF REVISED LOGIC ---

        # Step 4: Generate and store embeddings for all chunks in a batch
        print("Embedding text chunks...")
        chunk_embeddings = self.embeddings.embed_documents(all_chunks_text)
        chunk_embedding_data = [{"chunk_id": cid, "embedding": emb} for cid, emb in zip(all_chunk_ids, chunk_embeddings)]
        self.query(
            """
            UNWIND $data AS row
            MATCH (c:__Chunk__ {id: row.chunk_id})
            SET c.embedding = row.embedding
            """,
            params={"data": chunk_embedding_data}
        )
        
        # Step 5: Extract entities and relationships from all chunks
        print("Extracting entities and relationships...")
        await self.extract_nodes_and_rels(all_chunks_text, allowed_entities, all_chunk_ids)

        # Step 6: Summarize nodes and relationships
        print("Summarizing nodes and relationships...")
        await self.summarize_nodes_and_rels()

        # Step 7: Perform entity resolution
        print("Performing entity resolution...")
        # await self.entity_resolution()

        # Step 8: Detect and summarize communities
        print("Summarizing communities...")
        await self.summarize_communities()
        # Step 9: Extract relationships between documents
        print("Extracting relationships between documents...")
        await self.entity_relation_extraction_from_summaries()
        
        print("GraphRAG pipeline completed successfully.")

    async def search_chunks(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search for relevant text CHUNKS based on a user query.

        Args:
            query_text (str): The user's search query.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the chunk 'text' and a 'score'.
        """
        print(f"Searching for text chunks similar to: '{query_text}'")
        query_embedding = self.embeddings.embed_query(query_text)
        
        results = self.query(
            """
            CALL db.index.vector.queryNodes('chunk_text_embeddings', $top_k, $embedding)
            YIELD node AS chunk, score
            RETURN chunk.text AS text, score
            """,
            params={'top_k': top_k, 'embedding': query_embedding}
        )
        return results

    async def search_document_summaries(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search for relevant DOCUMENT SUMMARIES based on a user query.

        Args:
            query_text (str): The user's search query.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the doc 'title', 'summary', and 'score'.
        """
        print(f"Searching for document summaries similar to: '{query_text}'")
        query_embedding = self.embeddings.embed_query(query_text)
        
        results = self.query(
            """
            CALL db.index.vector.queryNodes('document_summary_embeddings', $top_k, $embedding)
            YIELD node AS doc, score
            RETURN doc.title AS title, doc.summary AS summary, score
            """,
            params={'top_k': top_k, 'embedding': query_embedding}
        )
        return results

    async def search_community_summaries(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search for relevant COMMUNITY SUMMARIES based on a user query.

        Args:
            query_text (str): The user's search query.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the community 'id', 'summary', and 'score'.
        """
        print(f"Searching for community summaries similar to: '{query_text}'")
        query_embedding = self.embeddings.embed_query(query_text)
        
        results = self.query(
            """
            CALL db.index.vector.queryNodes('community_summary_embeddings', $top_k, $embedding)
            YIELD node AS community, score
            RETURN community.id AS id, community.summary AS summary, score
            """,
            params={'top_k': top_k, 'embedding': query_embedding}
        )
        return results
    # ... (Keep the __init__, query, achat, close, __enter__, __exit__, __del__ methods as they are) ...
    def _check_driver_state(self) -> None:
        """
        Check if the Neo4j driver is still available.

        Raises:
            RuntimeError: If the Neo4j driver has been closed.
        """
        if not hasattr(self, "_driver"):
            raise RuntimeError(
                "This MsGraphRAG instance has been closed, and cannot be used anymore."
            )

    def query(
        self,
        query: str,
        params: dict = {},
        session_params: dict = {},
    ) -> List[Dict[str, Any]]:
        """Query Neo4j database.

        Args:
            query (str): The Cypher query to execute.
            params (dict): The parameters to pass to the query.
            session_params (dict): Parameters to pass to the session used for executing
                the query.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        if not session_params:
            try:
                data, _, _ = self._driver.execute_query(
                    Query(text=query),
                    database_=self._database,
                    parameters_=params,
                )
                return [r.data() for r in data]
            except Neo4jError as e:
                if not (
                    (
                        (  # isCallInTransactionError
                            e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                            or e.code
                            == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                        )
                        and e.message is not None
                        and "in an implicit transaction" in e.message
                    )
                    or (  # isPeriodicCommitError
                        e.code == "Neo.ClientError.Statement.SemanticError"
                        and e.message is not None
                        and (
                            "in an open transaction is not possible" in e.message
                            or "tried to execute in an explicit transaction"
                            in e.message
                        )
                    )
                ):
                    raise
        # fallback to allow implicit transactions
        session_params.setdefault("database", self._database)
        with self._driver.session(**session_params) as session:
            result = session.run(Query(text=query), params=params) # Removed timeout, not a class attribute
            return [r.data() for r in result]

    async def achat(self, messages, model="gpt-4o", config={}):
        response = await self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            **config,
        )
        return response.choices[0].message

    def close(self) -> None:
        """
        Explicitly close the Neo4j driver connection.

        Delegates connection management to the Neo4j driver.
        """
        if hasattr(self, "_driver"):
            self._driver.close()
            # Remove the driver attribute to indicate closure
            delattr(self, "_driver")

    def __enter__(self) -> "MsGraphRAG":
        """
        Enter the runtime context for the Neo4j graph connection.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:

        self.close()

    def __del__(self) -> None:

        try:
            self.close()
        except Exception:
            # Suppress any exceptions during garbage collection
            pass
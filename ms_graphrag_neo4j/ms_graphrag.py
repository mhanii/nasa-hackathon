import os
from typing import Any, Dict, List, Optional, Type
from neo4j import Driver
from openai import AsyncAzureOpenAI

from ms_graphrag_neo4j.providers.gemini import GeminiEmbeddings, GeminiLLM
from ms_graphrag_neo4j.components.graph_constructor import GraphConstructor
from ms_graphrag_neo4j.components.searcher import Searcher
from ms_graphrag_neo4j.components.rag_runner import RAGRunner


class MsGraphRAG(GraphConstructor, Searcher, RAGRunner):
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
        max_workers: int = 5,
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
        self.gemini_llm = GeminiLLM(model_name="gemini-2.5-pro")

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
            self.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Chunk__) REQUIRE e.id IS UNIQUE;"
            )
            self.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.name IS UNIQUE;"
            )
            self.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Community__) REQUIRE e.id IS UNIQUE;"
            )
            self.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS UNIQUE;"
            )
            self.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE;"
            )

            # Vector index for searching text chunks
            self.query(
                """
                CREATE VECTOR INDEX `chunk_text_embeddings` IF NOT EXISTS
                FOR (c:__Chunk__) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768, `vector.similarity_function`: 'cosine'
                }}
            """
            )
            # NEW: Vector index for searching document summaries
            self.query(
                """
                CREATE VECTOR INDEX `document_summary_embeddings` IF NOT EXISTS
                FOR (d:Document) ON (d.summary_embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768, `vector.similarity_function`: 'cosine'
                }}
            """
            )
            # NEW: Vector index for searching community summaries
            self.query(
                """
                CREATE VECTOR INDEX `community_summary_embeddings` IF NOT EXISTS
                FOR (c:__Community__) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768, `vector.similarity_function`: 'cosine'
                }}
            """
            )

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
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and e.message is not None
                    and "in an implicit transaction" in e.message
                ) or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and e.message is not None
                    and (
                        "in an open transaction is not possible" in e.message
                        or "tried to execute in an explicit transaction" in e.message
                    )
                ):
                    raise
        # fallback to allow implicit transactions
        session_params.setdefault("database", self._database)
        with self._driver.session(**session_params) as session:
            result = session.run(
                Query(text=query), params=params
            )  # Removed timeout, not a class attribute
            return [r.data() for r in result]

    async def achat(self, messages, model='gpt-5-mini', config={}):
        response = await self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            **config,
        )
        return response.choices[0].message


    async def astream_chat(self, messages, model='gpt-5-mini', config={}):
        response = await self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **config,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content



    # Replace your existing diagnostic methods with this full block.

# Replace your existing diagnostic methods with this full, corrected block.

    def run_graph_diagnostics(self, saturation_threshold: int = 25, sparsity_threshold: int = 3):
        """
        Runs a full suite of diagnostic checks on the graph and prints a comprehensive report.
        
        Args:
            saturation_threshold (int): The degree/count above which a node is considered "saturated".
            sparsity_threshold (int): The entity count below which a document is considered "sparse".
        """
        print("\n" + "="*50)
        print("      Running Knowledge Graph Health Diagnostics")
        print("="*50)
        
        # Section 1: Check for missing information
        self.identify_gaps()
        
        # Section 2: Check for overly dense areas
        self.find_saturated_areas(degree_threshold=saturation_threshold)
        
        # Section 3: Check for under-discovered areas
        self.find_sparse_areas(entity_threshold=sparsity_threshold)
        
        # Section 4: Analyze the relationships between documents
        self.analyze_document_relationships()
        
        print("\n" + "="*50)
        print("            Diagnostic Run Complete")
        print("="*50)

    def identify_gaps(self):
        """
        Identifies potential gaps in the knowledge graph, such as missing summaries,
        disconnected entities, or isolated documents.
        """
        print("\n--- 1. Identifying Gaps (Missing Information) ---")
        
        # Gap 1: Entities that were extracted but never summarized
        unsummarized_entities = self.query("""
            MATCH (e:__Entity__)
            WHERE e.summary IS NULL AND size(e.description) > 1
            RETURN e.name AS name, size(e.description) AS description_count
            ORDER BY description_count DESC LIMIT 10
        """)
        if unsummarized_entities:
            print(f"\n[GAP] Found {len(unsummarized_entities)} multi-mention entities needing summarization:")
            for record in unsummarized_entities:
                print(f"  - Entity: '{record['name']}' ({record['description_count']} descriptions)")
        else:
            print("\n✓ No unsummarized entities found.")
            
        # Gap 2: Orphaned entities with no relationships at all
        orphaned_entities = self.query("""
            MATCH (e:__Entity__) WHERE NOT (e)--()
            RETURN e.name AS name LIMIT 10
        """)
        if orphaned_entities:
            print(f"\n[GAP] Found {len(orphaned_entities)} orphaned entities with no relationships:")
            for record in orphaned_entities:
                print(f"  - Entity: '{record['name']}'")
        else:
            print("\n✓ No orphaned entities found.")

        # Gap 3: Documents that have no typed relationships with other documents
        isolated_docs = self.query("""
            MATCH (d:Document)
            WHERE NOT (d)-[:SUPPORTS|CONTRADICTS|BUILDS_UPON|NEUTRAL]-()
            AND NOT ()-[:SUPPORTS|CONTRADICTS|BUILDS_UPON|NEUTRAL]->(d)
            RETURN d.title AS title LIMIT 10
        """)
        if isolated_docs:
            print(f"\n[GAP] Found {len(isolated_docs)} documents isolated from inter-document analysis:")
            for record in isolated_docs:
                print(f"  - Document: '{record['title']}'")
        else:
            print("\n✓ All documents are integrated into the inter-document analysis.")


    def find_saturated_areas(self, degree_threshold: int = 25):
        """
        Finds "saturated" parts of the graph, such as hub nodes or documents that
        generated an unusually high number of entities.
        """
        print(f"\n--- 2. Finding Saturated Areas (High Density) ---")
        
        # Saturation 1: "Hub" entities with many connections
        # CORRECTED a syntax error here: size((e)--()) is deprecated.
        hub_entities = self.query("""
            MATCH (e:__Entity__)
            WITH e, COUNT { (e)--() } AS degree
            WHERE degree > $threshold
            RETURN e.name AS name, degree
            ORDER BY degree DESC LIMIT 10
        """, params={"threshold": degree_threshold})
        
        if hub_entities:
            print(f"\n[SATURATED] Found {len(hub_entities)} hub entities (degree > {degree_threshold}):")
            for record in hub_entities:
                print(f"  - Entity: '{record['name']}' ({record['degree']} relationships)")
        else:
            print(f"\n✓ No significant hub entities found (degree > {degree_threshold}).")
            
        # Saturation 2: Documents that generated a very high number of entities
        high_yield_docs = self.query("""
            MATCH (d:Document)<--(:__Chunk__)-->(e:__Entity__)
            WITH d, count(DISTINCT e) AS entity_count
            WHERE entity_count > $threshold
            RETURN d.title AS title, entity_count
            ORDER BY entity_count DESC LIMIT 10
        """, params={"threshold": degree_threshold})
        
        if high_yield_docs:
            print(f"\n[SATURATED] Found {len(high_yield_docs)} high-yield documents (> {degree_threshold} entities):")
            for record in high_yield_docs:
                print(f"  - Document: '{record['title']}' ({record['entity_count']} entities)")
        else:
            print(f"\n✓ No overly dense documents found.")


    def find_sparse_areas(self, entity_threshold: int = 3):
        """
        Finds "sparse" or "under-discovered" areas, such as leaf nodes or documents
        that yielded few entities.
        """
        print(f"\n--- 3. Finding Sparse Areas (Low Discovery) ---")
        
        # Sparsity 1: Leaf entities with only one connection
        # CORRECTED a syntax error here: size((e)--()) is deprecated.
        leaf_entities = self.query("""
            MATCH (e:__Entity__)
            WHERE COUNT { (e)--() } = 1
            RETURN e.name AS name LIMIT 10
        """)
        if leaf_entities:
            print(f"\n[SPARSE] Found {len(leaf_entities)} leaf entities (only one connection):")
            for record in leaf_entities:
                print(f"  - Entity: '{record['name']}'")
        else:
            print("\n✓ No leaf entities found.")
                
        # Sparsity 2: Documents that generated very few entities
        low_yield_docs = self.query("""
            MATCH (d:Document)
            WITH d, size([(d)<--(:__Chunk__)--(e:__Entity__) | e]) as entity_count
            WHERE entity_count < $threshold
            RETURN d.title AS title, entity_count
            ORDER BY entity_count ASC LIMIT 10
        """, params={"threshold": entity_threshold})
        
        if low_yield_docs:
            print(f"\n[SPARSE] Found {len(low_yield_docs)} documents yielding < {entity_threshold} entities:")
            for record in low_yield_docs:
                print(f"  - Document: '{record['title']}' ({record['entity_count']} entities)")
        else:
            print(f"\n✓ All documents appear to generate a sufficient number of entities.")

    def analyze_document_relationships(self):
        """
        Analyzes and aggregates the typed relationships between documents to understand
        the overall discourse within the knowledge base.
        """
        print("\n--- 4. Analyzing Inter-Document Discourse ---")

        # Analysis 1: Overall relationship counts
        rel_counts = self.query("""
            MATCH ()-[r]->()
            WHERE type(r) IN ['SUPPORTS', 'CONTRADICTS', 'BUILDS_UPON', 'NEUTRAL']
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """)
        if rel_counts:
            print("\n[ANALYSIS] Overall distribution of document relationships:")
            for record in rel_counts:
                print(f"  - {record['rel_type']}: {record['count']} relationships")
        else:
            print("\n✓ No inter-document relationships found to analyze.")
            return # Stop here if there's nothing to analyze

        # Analysis 2: Find the most influential documents (high outgoing support/builds_upon)
        influential_docs = self.query("""
            MATCH (d:Document)-[r:SUPPORTS|BUILDS_UPON]->()
            RETURN d.title AS title, count(r) AS influence_score
            ORDER BY influence_score DESC LIMIT 5
        """)
        if influential_docs:
            print("\n[ANALYSIS] Top 5 most influential documents (providing support to others):")
            for record in influential_docs:
                print(f"  - '{record['title']}' (supports/builds upon {record['influence_score']} other docs)")
                
        # Analysis 3: Find the most widely supported documents (high incoming support)
        supported_docs = self.query("""
            MATCH (d:Document)<-[r:SUPPORTS|BUILDS_UPON]-()
            RETURN d.title AS title, count(r) AS support_score
            ORDER BY support_score DESC LIMIT 5
        """)
        if supported_docs:
            print("\n[ANALYSIS] Top 5 most widely supported documents (receiving support):")
            for record in supported_docs:
                print(f"  - '{record['title']}' (supported by {record['support_score']} other docs)")
                
        # Analysis 4: Find the most controversial/contradictory documents
        contradictory_docs = self.query("""
            MATCH (d:Document)-[r:CONTRADICTS]->()
            RETURN d.title AS title, count(r) AS contradiction_score
            ORDER BY contradiction_score DESC LIMIT 5
        """)
        if contradictory_docs:
            print("\n[ANALYSIS] Top 5 most contradictory documents:")
            for record in contradictory_docs:
                print(f"  - '{record['title']}' (contradicts {record['contradiction_score']} other docs)")


    def get_all_document_metadata(self) -> List[Dict[str, Any]]:
        """
        Fetches metadata for all Document nodes, including authors from relationships,
        and uses the generated summary instead of the original abstract.
        
        It attempts to collect all necessary properties (id, title, year, summary, topics, organisms, missions).
        """
        # The query assumes that:
        # 1. The document's ID is stored as a property, e.g., d.id.
        # 2. The generated document summary is stored in d.summary.
        # 3. List properties (topics, organisms, missions) are stored as array properties on the node.
        
        query = """
            MATCH (d:Document)<-[:FROM_DOCUMENT]-(c:__Chunk__)
            WHERE c.title = 'Abstract'
            RETURN d.title AS Title, d.authors AS Authors, c.text AS Abstract
        """
        
        return self.query(query)


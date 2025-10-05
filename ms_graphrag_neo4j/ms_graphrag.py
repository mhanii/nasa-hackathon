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



    def run_graph_diagnostics(self, saturation_threshold: int = 25, sparsity_threshold: int = 3):
        """
        Runs a full suite of diagnostic checks on the graph (entity and document level)
        and prints a comprehensive report.
        
        Args:
            saturation_threshold (int): The degree/count above which a node is considered "saturated".
            sparsity_threshold (int): The entity count below which a document is considered "sparse".
        """
        print("\n" + "="*50)
        print("      Running Knowledge Graph Health Diagnostics")
        print("="*50)
        
        self.identify_gaps()
        self.find_saturated_areas(degree_threshold=saturation_threshold)
        self.find_sparse_areas(entity_threshold=sparsity_threshold)
        
        print("\n" + "="*50)
        print("            Diagnostic Run Complete")
        print("="*50)

    def identify_gaps(self):
        """
        Identifies potential gaps in the knowledge graph, such as missing summaries
        or disconnected entities.
        """
        print("\n--- 1. Identifying Gaps (Missing Information) ---")
        
        # Gap 1: Entities that were extracted but never summarized
        unsummarized_entities = self.query("""
            MATCH (e:__Entity__)
            WHERE e.summary IS NULL AND size(e.description) > 1
            RETURN e.name AS name, size(e.description) AS description_count
            ORDER BY description_count DESC
            LIMIT 10
        """)
        if unsummarized_entities:
            print(f"\n[GAP] Found {len(unsummarized_entities)} multi-mention entities needing summarization:")
            for record in unsummarized_entities:
                print(f"  - Entity: '{record['name']}' ({record['description_count']} descriptions)")
        else:
            print("\n✓ No unsummarized entities found.")
            
        # Gap 2: Orphaned entities with no relationships at all
        orphaned_entities = self.query("""
            MATCH (e:__Entity__)
            WHERE NOT (e)--()
            RETURN e.name AS name
            LIMIT 10
        """)
        if orphaned_entities:
            print(f"\n[GAP] Found {len(orphaned_entities)} orphaned entities with no relationships:")
            for record in orphaned_entities:
                print(f"  - Entity: '{record['name']}'")
        else:
            print("\n✓ No orphaned entities found.")

    def find_saturated_areas(self, degree_threshold: int = 25):
        """
        Finds "saturated" parts of the graph, such as hub nodes or documents that
        generated an unusually high number of entities.
        
        Args:
            degree_threshold (int): The number of relationships/entities above which to flag an item.
        """
        print(f"\n--- 2. Finding Saturated Areas (High Density) ---")
        
        # Saturation 1: "Hub" entities with many connections
        hub_entities = self.query("""
            MATCH (e:__Entity__)
            WITH e, size((e)--()) AS degree
            WHERE degree > $threshold
            RETURN e.name AS name, degree
            ORDER BY degree DESC
            LIMIT 10
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
            ORDER BY entity_count DESC
            LIMIT 10
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
        
        Args:
            entity_threshold (int): The count below which a document is considered sparse.
        """
        print(f"\n--- 3. Finding Sparse Areas (Low Discovery) ---")
        
        # Sparsity 1: Leaf entities with only one connection
        leaf_entities = self.query("""
            MATCH (e:__Entity__)
            WHERE size((e)--()) = 1
            RETURN e.name AS name
            LIMIT 10
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
            ORDER BY entity_count ASC
            LIMIT 10
        """, params={"threshold": entity_threshold})
        
        if low_yield_docs:
            print(f"\n[SPARSE] Found {len(low_yield_docs)} documents yielding < {entity_threshold} entities:")
            for record in low_yield_docs:
                print(f"  - Document: '{record['title']}' ({record['entity_count']} entities)")
        else:
            print(f"\n✓ All documents appear to generate a sufficient number of entities.")

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
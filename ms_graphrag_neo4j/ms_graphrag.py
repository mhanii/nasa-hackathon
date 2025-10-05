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
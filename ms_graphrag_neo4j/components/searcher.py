from typing import Any, Dict, List


class Searcher:
    async def search_chunks(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search for relevant text CHUNKS based on a user query.
        """
        print(f"Searching for text chunks similar to: '{query_text}'")
        query_embedding = self.embeddings.embed_query(query_text)

        results = self.query(
            """
            CALL db.index.vector.queryNodes('chunk_text_embeddings', $top_k, $embedding)
            YIELD node AS chunk, score
            RETURN chunk.text AS text, score, chunk.id as id
            """,
            params={"top_k": top_k, "embedding": query_embedding},
        )
        return results

    async def search_document_summaries(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search for relevant DOCUMENT SUMMARIES based on a user query.
        """
        print(f"Searching for document summaries similar to: '{query_text}'")
        query_embedding = self.embeddings.embed_query(query_text)

        results = self.query(
            """
            CALL db.index.vector.queryNodes('document_summary_embeddings', $top_k, $embedding)
            YIELD node AS doc, score
            RETURN doc.title AS title, doc.summary AS summary, score
            """,
            params={"top_k": top_k, "embedding": query_embedding},
        )
        return results

    async def search_community_summaries(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search for relevant COMMUNITY SUMMARIES based on a user query.
        """
        print(f"Searching for community summaries similar to: '{query_text}'")
        query_embedding = self.embeddings.embed_query(query_text)

        results = self.query(
            """
            CALL db.index.vector.queryNodes('community_summary_embeddings', $top_k, $embedding)
            YIELD node AS community, score
            RETURN community.id AS id, community.summary AS summary, score
            """,
            params={"top_k": top_k, "embedding": query_embedding},
        )
        return results
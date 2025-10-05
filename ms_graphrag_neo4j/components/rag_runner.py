class RAGRunner:
    async def run(self, prompt: str, top_k: int = 5, top_m: int = 2) -> str:
        """
        Executes the full GraphRAG pipeline to answer a prompt.
        """
        print(f"\n--- Running GraphRAG pipeline for prompt: '{prompt}' ---")

        # 1. Retrieve relevant chunks
        print(f"Step 1: Searching for top {top_k} relevant chunks...")
        chunks = await self.search_chunks(prompt, top_k=top_k)
        if not chunks:
            return "I couldn't find any relevant information in the documents to answer your question."

        chunk_ids = [c["id"] for c in chunks]
        context_str = "CONTEXT:\n\n"

        # 2. Traverse graph from chunks
        print("Step 2: Traversing graph from chunks for local context (2 hops)...")
        graph_context = self.query(
            """
            UNWIND $chunk_ids AS c_id
            MATCH (c:__Chunk__ {id: c_id})
            CALL apoc.path.subgraphNodes(c, {maxLevel: 2}) YIELD node
            WITH c_id, node
            // Exclude the original chunk itself from this context to avoid redundancy
            WHERE NOT node:__Chunk__ OR node.id <> c_id
            RETURN node.name AS name, node.summary AS summary, labels(node) AS labels
            """,
            params={"chunk_ids": chunk_ids},
        )
        context_str += "Graph Context:\n"
        for item in graph_context:
            context_str += f"- Node: {item['name']}, Labels: {item['labels']}, Summary: {item['summary']}\n"

        # 3. Get document summaries for top_m chunks
        print(f"Step 3: Retrieving document summaries for top {top_m} chunks...")
        seen_doc_titles = set()
        doc_summaries_context = self.query(
            """
            UNWIND $chunk_ids AS c_id
            MATCH (:__Chunk__ {id: c_id})-[:FROM_DOCUMENT]->(d:Document)
            RETURN d.title AS title, d.summary AS summary
            """,
            params={"chunk_ids": chunk_ids[:top_m]},
        )
        context_str += "\nPrimary Document Summaries:\n"
        for doc in doc_summaries_context:
            context_str += f"- Document '{doc['title']}': {doc['summary']}\n"
            seen_doc_titles.add(doc["title"])

        # 4. Global search for other relevant documents
        print("Step 4: Performing global search for additional relevant documents...")
        additional_docs = await self.search_document_summaries(prompt, top_k=top_k)

        added_docs_count = 0
        context_str += "\nAdditional Relevant Document Summaries:\n"
        for doc in additional_docs:
            if doc["title"] not in seen_doc_titles:
                context_str += f"- Document '{doc['title']}': {doc['summary']}\n"
                added_docs_count += 1
        if added_docs_count == 0:
            context_str += "- No additional documents found.\n"

        # 5. Provide context to LLM and get an answer
        print("Step 5: Synthesizing context and generating final answer with Gemini...")
        final_prompt = (
            f"{context_str}\n\nBased *only* on the context provided above, "
            f"please answer the following question. Do not use any prior knowledge. "
            f"If the context does not contain the answer, say so.\n\n"
            f"Question: {prompt}"
        )

        response = await self.achat([{"role": "user", "content": final_prompt}])

        print("--- GraphRAG pipeline complete. ---")
        return response.content
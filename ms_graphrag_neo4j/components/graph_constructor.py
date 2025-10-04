import asyncio
from tqdm.asyncio import tqdm, tqdm_asyncio
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from ms_graphrag_neo4j.utils import get_hash, parse_extraction_output, extract_json
from ms_graphrag_neo4j.cypher_queries import (
    import_nodes_query,
    import_relationships_query,
    candidate_nodes_summarization,
    candidate_rels_summarization,
    import_entity_summary,
    import_entity_summary_single,
    import_rel_summary,
    import_rel_summary_single,
    drop_gds_graph_query,
    create_gds_graph_query,
    leiden_query,
    community_hierarchy_query,
    community_info_query,
)
from ms_graphrag_neo4j.prompts import (
    GRAPH_EXTRACTION_PROMPT,
    SUMMARIZE_PROMPT,
    DOCUMENT_SUMMARY_PROMPT,
    COMMUNITY_REPORT_PROMPT,
)


class GraphConstructor:
    async def extract_nodes_and_rels(
        self, input_texts: list, allowed_entities: list, chunk_ids: list
    ) -> str:
        """
        Extract nodes and relationships from input texts using LLM and store them in Neo4j.
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
            output = await self.achat(messages, model=self.model)
            return parse_extraction_output(output.content)

        tasks = [process_text(text) for text in input_texts]

        if self.max_workers:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(task):
                async with semaphore:
                    return await task

            results = await tqdm_asyncio.gather(
                *[process_with_semaphore(task) for task in tasks],
                desc="Extracting nodes & relationships",
            )
        else:
            results = await tqdm_asyncio.gather(
                *tasks, desc="Extracting nodes & relationships"
            )

        total_relationships = 0
        for text, chunk_id, output in zip(input_texts, chunk_ids, results):
            nodes, relationships = output
            total_relationships += len(relationships)
            self.query(
                import_nodes_query,
                params={"text": text, "chunk_id": chunk_id, "data": nodes, "rel": "MENTIONS"},
            )
            self.query(import_relationships_query, params={"data": relationships})

        return f"Successfuly extracted and imported {total_relationships} relationships"

    async def summarize_nodes_and_rels(self) -> str:
        """
        Generate summaries for all nodes and relationships in the graph.
        """
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
            summary = await self.achat(messages, model="gpt-4-mini")
            return {
                "source": rel["source"],
                "target": rel["target"],
                "summary": summary.content,
            }

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

        self.query(import_entity_summary, params={"data": summaries})
        self.query(import_entity_summary_single)

        self.query(import_rel_summary, params={"data": rel_summaries})
        self.query(import_rel_summary_single)

        return "Successfuly summarized nodes and relationships"

    async def summarize_document(self, metadata, document):
        messages = [
            {
                "role": "user",
                "content": DOCUMENT_SUMMARY_PROMPT.format(
                    metadata=metadata, document=document
                ),
            },
        ]
        summary = await self.achat(messages, model="gpt-4-mini")
        return summary.content

    async def entity_relation_extraction_from_summaries(
        self, top_k: int = 5, similarity_threshold: float = 0.8, use_nli: bool = False
    ) -> str:
        if use_nli:
            from transformers import pipeline
            nli_model = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )

        documents = self.query(
            "MATCH (d:Document) RETURN d.title AS title, d.summary AS summary"
        )

        async def process_document(document):
            similar_documents = await self.search_document_summaries(
                query_text=document["summary"], top_k=top_k
            )
            relationships = []
            for similar_document in similar_documents:
                if similar_document["score"] > similarity_threshold:
                    if use_nli:
                        sequence_to_classify = similar_document["summary"]
                        candidate_labels = ["entailment", "contradiction", "neutral"]
                        result = nli_model(sequence_to_classify, candidate_labels)
                        relationship = result["labels"][0]
                    else:
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

    async def summarize_communities(self, summarize_all_levels: bool = False) -> str:
        self.query(drop_gds_graph_query)
        self.query(create_gds_graph_query)
        community_summary_result = self.query(leiden_query)
        community_levels = community_summary_result[0]["ranLevels"]
        print(
            f"Leiden algorithm identified {community_levels} community levels "
            f"with {community_summary_result[0]['communityCount']} communities on the last level."
        )
        self.query(community_hierarchy_query)

        if summarize_all_levels:
            levels = list(range(community_levels))
        else:
            levels = [community_levels - 1]
        communities = self.query(community_info_query, params={"levels": levels})

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
            summary = await self.achat(messages, model="gpt-4-mini")
            summary_json = extract_json(summary.content)
            title_text = summary_json.get("title", "Untitled Community")
            summary_text = summary_json.get("summary", "")
            explanation_text = summary_json.get("explanation", "")
            summary_embedding = (
                self.embeddings.embed_query(summary_text) if summary_text else None
            )
            community_id = community["communityId"]
            properties_to_set = {
                "title": title_text,
                "summary": summary_text,
                "explanation": explanation_text,
                "embedding": summary_embedding,
            }
            return {"communityId": community_id, "properties": properties_to_set}

        if self.max_workers:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_community_with_semaphore(community):
                async with semaphore:
                    return await process_community(community)

            community_summaries = await tqdm_asyncio.gather(
                *(process_community_with_semaphore(c) for c in communities),
                desc="Summarizing communities",
                total=len(communities),
            )
        else:
            community_summaries = await tqdm_asyncio.gather(
                *(process_community(c) for c in communities),
                desc="Summarizing communities",
                total=len(communities),
            )

        import_community_summary_with_embedding = """
        UNWIND $data AS d
        MATCH (c:__Community__ {id: d.communityId})
        SET c += d.properties
        """
        self.query(
            import_community_summary_with_embedding,
            params={"data": community_summaries},
        )
        return f"Generated {len(community_summaries)} community summaries"

    async def load_data(self, document, allowed_entities):
        doc_metadata = document.get("metadata", {})
        doc_title = doc_metadata.get("title", "Untitled Document")
        doc_authors = doc_metadata.get("authors", [])
        doc_content = document.get("sections", [])

        print(f"Summarizing document: {doc_title}")
        doc_summary = await self.summarize_document(
            metadata=doc_metadata, document=doc_content
        )
        doc_summary_embedding = self.embeddings.embed_query(doc_summary)
        doc_metadata["summary"] = doc_summary

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
                "authors": doc_authors,
            },
        )
        print(
            f"Created document node for '{doc_title}' and connected {len(doc_authors)} author(s)."
        )

        all_chunks_text = []
        all_chunk_ids = []
        all_chunks_data = []

        for chunk in doc_content:
            chunk_id = get_hash(chunk["text"])
            all_chunks_text.append(chunk["text"])
            all_chunk_ids.append(chunk_id)
            all_chunks_data.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "section": chunk["title"],
                }
            )

        print(
            f"Creating and linking {len(all_chunks_data)} chunk nodes in a single batch..."
        )
        self.query(
            """
            MATCH (d:Document {title: $doc_title})
            UNWIND $chunks AS chunk_data
            MERGE (c:__Chunk__ {id: chunk_data.chunk_id})
            SET c.text = chunk_data.text, c.title = chunk_data.section
            WITH d, c
            MERGE (c)-[:FROM_DOCUMENT]->(d)
            """,
            params={"doc_title": doc_title, "chunks": all_chunks_data},
        )

        print("Embedding text chunks...")
        chunk_embeddings = self.embeddings.embed_documents(all_chunks_text)
        chunk_embedding_data = [
            {"chunk_id": cid, "embedding": emb}
            for cid, emb in zip(all_chunk_ids, chunk_embeddings)
        ]
        self.query(
            """
            UNWIND $data AS row
            MATCH (c:__Chunk__ {id: row.chunk_id})
            SET c.embedding = row.embedding
            """,
            params={"data": chunk_embedding_data},
        )

        print("Extracting entities and relationships...")
        await self.extract_nodes_and_rels(
            all_chunks_text, allowed_entities, all_chunk_ids
        )

        print("Summarizing nodes and relationships...")
        await self.summarize_nodes_and_rels()

        print("Summarizing communities...")
        await self.summarize_communities()

        print("Extracting relationships between documents...")
        await self.entity_relation_extraction_from_summaries()

        print("GraphRAG pipeline completed successfully.")
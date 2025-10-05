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
from collections import defaultdict

from ms_graphrag_neo4j.prompts import (
    GRAPH_EXTRACTION_PROMPT,
    SUMMARIZE_PROMPT,
    DOCUMENT_SUMMARY_PROMPT,
    COMMUNITY_REPORT_PROMPT,
    DOCUMENT_RELATIONSHIP_PROMPT
)


class GraphConstructor:
    async def extract_nodes_and_rels(
        self, input_texts: list, allowed_entities: list,allowed_relationships:list, chunk_ids: list
    ) -> str:
        """
        Extracts nodes and relationships from input texts using an LLM, creating richly typed
        relationships in Neo4j based on the LLM's output.
        """

        async def process_text(input_text):
            prompt = GRAPH_EXTRACTION_PROMPT.format(
                entity_types=allowed_entities,
                relationship_types=allowed_relationships,
                input_text=input_text,
                tuple_delimiter=";",
                record_delimiter="|",
                completion_delimiter="\n\n",
            )
            messages = [{"role": "user", "content": prompt}]
            output = await self.achat(messages)
            return parse_extraction_output(output.content)

        tasks = [process_text(text) for text in input_texts]

        # Use a semaphore to control concurrency if max_workers is set
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

        all_nodes = []
        all_relationships = []

        # Process nodes and link them to their source chunks
        for text, chunk_id, (nodes, rels) in zip(input_texts, chunk_ids, results):
            if nodes:
                # Import nodes and create MENTIONS relationship to the chunk
                self.query(
                    import_nodes_query,
                    params={"text": text, "chunk_id": chunk_id, "data": nodes},
                )
            all_relationships.extend(rels)

        # Group relationships by their type for batch import
        grouped_rels = defaultdict(list)
        for rel in all_relationships:
            # The parser defaults to 'RELATED_TO' if no type is found
            rel_type = rel.get("relationship_type", "RELATED_TO")
            grouped_rels[rel_type].append(rel)

        # Dynamically create and execute queries for each relationship type
        total_relationships_imported = 0
        for rel_type, rel_data in grouped_rels.items():
            # Security validation: Ensure relationship type is a safe value


            # Dynamically construct the query with the sanitized relationship type
            import_relationships_query_dynamic = f"""
            UNWIND $data AS row
            MERGE (s:__Entity__ {{name: row.source_entity}})
            MERGE (t:__Entity__ {{name: row.target_entity}})
            CREATE (s)-[r:{rel_type} {{
                description: row.relationship_description,
                strength: row.relationship_strength
            }}]->(t)
            """
            
            try:
                self.query(import_relationships_query_dynamic, params={"data": rel_data})
                total_relationships_imported += len(rel_data)
                print(f"Imported {len(rel_data)} relationships of type :{rel_type}")
            except Exception as e:
                print(f"Failed to import relationships of type :{rel_type}. Error: {e}")

        return (
            f"Successfully extracted and imported {total_relationships_imported} relationships "
            f"across {len(grouped_rels)} types."
        )

    def create_custom_relationship(
            self,
            source_entity: str,
            target_entity: str,
            relationship_type: str,
            properties: dict = None,
        ) -> str:
            """
            Creates a custom, typed relationship between two existing entities.

            Args:
                source_entity (str): The name of the source entity node.
                target_entity (str): The name of the target entity node.
                relationship_type (str): The type of the relationship (e.g., 'WORKS_FOR', 'IS_A').
                                        This will be used as the relationship label.
                properties (dict, optional): A dictionary of properties to set on the relationship.
                                            Defaults to None.

            Returns:
                str: A confirmation message indicating the result of the operation.
            """
            if not relationship_type.isalnum() or not relationship_type.isascii():
                raise ValueError("Relationship type must be alphanumeric ASCII")

            if properties is None:
                properties = {}

            # The relationship type cannot be parameterized directly in Cypher,
            # so we format it into the string after validation.
            query = f"""
            MATCH (s:__Entity__ {{name: $source_name}})
            MATCH (t:__Entity__ {{name: $target_name}})
            MERGE (s)-[r:{relationship_type}]->(t)
            SET r += $props
            RETURN s.name AS source, type(r) AS type, t.name AS target
            """

            params = {
                "source_name": source_entity,
                "target_name": target_entity,
                "props": properties,
            }

            try:
                result = self.query(query, params=params)
                if result:
                    res = result[0]
                    return (
                        f"Successfully created relationship: "
                        f"({res['source']})-[:{res['type']}]->({res['target']})"
                    )
                else:
                    return "Failed to create relationship. Make sure both source and target entities exist."
            except Exception as e:
                return f"An error occurred: {e}" 
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
            summary = await self.achat(messages)
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
            summary = await self.achat(messages)
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
        summary = await self.achat(messages)
        return summary.content

    async def entity_relation_extraction_from_summaries(
        self, top_k: int = 5, similarity_threshold: float = 0.8, use_nli: bool = False
    ) -> str:
        """
        Extracts and creates typed relationships between document summaries based on similarity
        and Natural Language Inference (NLI).
        """
        if use_nli:
            from transformers import pipeline
            nli_model = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )

        documents = self.query(
            "MATCH (d:Document) RETURN d.title AS title, d.summary AS summary"
        )
        
        # This is our predefined set of relationship types
        allowed_rel_types = ["SUPPORTS", "CONTRADICTS", "BUILDS_UPON", "NEUTRAL", "NOT_RELATED"]

        async def process_document(document):
            # Ensure the document has a summary to work with
            if not document.get("summary"):
                return []

            similar_documents = await self.search_document_summaries(
                query_text=document["summary"], top_k=top_k
            )
            
            relationships = []
            for similar_doc in similar_documents:
                # Skip if the document is being compared with itself or score is below threshold
                if similar_doc["title"] == document["title"] or similar_doc["score"] < similarity_threshold:
                    continue

                relationship_type = "NOT_RELATED" # Default value

                if use_nli:
                    # NLI model is generally better at "entailment", "contradiction", and "neutral"
                    # We can map these to our custom types.
                    nli_labels = ["entailment", "contradiction", "neutral"]
                    sequence_to_classify = similar_doc["summary"]
                    # The premise is the source document
                    premise = document["summary"]
                    
                    # NLI models expect a premise and a hypothesis
                    result = nli_model(sequence_to_classify, nli_labels, hypothesis_template=f"This text is about {premise}")
                    
                    # Map NLI output to our custom types
                    top_label = result["labels"][0]
                    if top_label == "entailment":
                        relationship_type = "SUPPORTS"
                    elif top_label == "contradiction":
                        relationship_type = "CONTRADICTS"
                    else: # neutral
                        relationship_type = "NEUTRAL"
                else:
                    # Use the more powerful general LLM for classification
                    prompt = DOCUMENT_RELATIONSHIP_PROMPT.format(
                        document_1_summary=document["summary"],
                        document_2_summary=similar_doc["summary"]
                    )
                    messages = [{"role": "user", "content": prompt}]
                    response = await self.achat(messages)
                    
                    # Clean up the response to get just the label
                    extracted_type = response.content.strip().upper()
                    # Ensure the extracted type is one of the allowed ones
                    if extracted_type in allowed_rel_types:
                        relationship_type = extracted_type

                relationships.append({
                    "source": document["title"],
                    "target": similar_doc["title"],
                    "type": relationship_type,
                    "similarity_score": similar_doc["score"]
                })
            return relationships

        # The outer 'for doc in documents:' loop has been removed to prevent redundant processing.
        processed_rels_lists = await asyncio.gather(*[process_document(doc) for doc in documents])

        all_relationships = []
        for rel_list in processed_rels_lists:
            all_relationships.extend(rel_list)

        # Group relationships by type for efficient, batch import
        grouped_rels = defaultdict(list)
        for rel in all_relationships:
            grouped_rels[rel["type"]].append(rel)

        total_rels_created = 0
        for rel_type, rel_data in grouped_rels.items():
            # Skip creating relationships for "NOT_RELATED"
            if rel_type == "NOT_RELATED":
                continue

            # Dynamically construct the query with the relationship type
            # Note: Relationship types in Cypher cannot be parameterized directly.
            # This approach is safe as `rel_type` is validated against `allowed_rel_types`.
            query = f"""
            UNWIND $relationships AS rel
            MATCH (d1:Document {{title: rel.source}})
            MATCH (d2:Document {{title: rel.target}})
            MERGE (d1)-[r:{rel_type}]->(d2)
            SET r.similarity = rel.similarity_score
            """
            
            self.query(query, params={"relationships": rel_data})
            total_rels_created += len(rel_data)
            print(f"Created {len(rel_data)} relationships of type :{rel_type}")

        return f"Created a total of {total_rels_created} typed relationships between documents."
    
    def analyze_leiden_output(self) -> str:
        """
        Analyze the Leiden algorithm output to understand community structure.
        This helps diagnose why orphaned communities are being created.
        """
        print("\n=== Analyzing Leiden Community Detection ===")
        
        # Check entity community assignments
        entity_analysis = self.query("""
            MATCH (e:__Entity__)
            WHERE e.communities IS NOT NULL
            RETURN 
                size(e.communities) AS hierarchy_depth,
                count(e) AS entity_count
            ORDER BY hierarchy_depth
        """)
        
        print("\nEntity Community Hierarchy Distribution:")
        for row in entity_analysis:
            print(f"  Depth {row['hierarchy_depth']}: {row['entity_count']} entities")
        
        # Check for entities with community arrays but no connections
        orphaned_entities = self.query("""
            MATCH (e:__Entity__)
            WHERE e.communities IS NOT NULL
            AND NOT (e)-[:IN_COMMUNITY]->()
            RETURN count(e) AS orphaned_entity_count
        """)
        
        print(f"\nEntities with community arrays but no connections: "
            f"{orphaned_entities[0]['orphaned_entity_count']}")
        
        # Check community node creation vs entity assignments
        community_counts = self.query("""
            MATCH (e:__Entity__)
            WHERE e.communities IS NOT NULL
            UNWIND range(0, size(e.communities) - 1) AS level
            WITH level, e.communities[level] AS community_id
            RETURN level, 
                count(DISTINCT community_id) AS unique_communities,
                count(*) AS total_assignments
            ORDER BY level
        """)
        
        print("\nExpected communities per level (from entity assignments):")
        for row in community_counts:
            print(f"  Level {row['level']}: {row['unique_communities']} unique communities, "
                f"{row['total_assignments']} total assignments")
        
        actual_communities = self.query("""
            MATCH (c:__Community__)
            RETURN c.level AS level, count(c) AS actual_count
            ORDER BY level
        """)
        
        print("\nActual community nodes created:")
        for row in actual_communities:
            print(f"  Level {row['level']}: {row['actual_count']} communities")
        
        return "Analysis complete"
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
        self.query("""
            MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
            WITH c, count(distinct d) AS rank
            SET c.community_rank = rank;
        """)
        if summarize_all_levels:
            levels = list(range(community_levels))
        else:
            levels = [community_levels - 1,community_levels]
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
            summary = await self.achat(messages)
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

    async def load_data(self, document, allowed_entities,allowed_relationships):
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
            all_chunks_text, allowed_entities,allowed_relationships, all_chunk_ids,
        )

        print("Summarizing nodes and relationships...")
        await self.summarize_nodes_and_rels()
        self.analyze_leiden_output()
        # self.diagnose_and_prune_communities()
        print("Summarizing communities...")
        # await self.summarize_communities()

        print("Extracting relationships between documents...")
        await self.entity_relation_extraction_from_summaries()

        print("GraphRAG pipeline completed successfully.")
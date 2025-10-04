from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks






class SectionSplitter:
    def __init__(self):
        
        pass


    def run(self, html_content: str) -> TextChunks:
        """
        Process HTML content and split into sections.
        
        Args:
            html_content (str): The HTML content or URL to process
            
        Returns:
            TextChunks: Object containing list of TextChunk objects with content and metadata
        """
        # Mock data for testing - this will be replaced with actual HTML processing
        mock_sections = [
            {
                "content": "Neo4j is a graph database management system developed by Neo4j, Inc. "
                          "It is described by its developers as an ACID-compliant transactional database "
                          "with native graph storage and processing.",
                "metadata": {
                    "section": "introduction",
                }
            },
            {
                "content": "Graph databases are designed to treat the relationships between data as equally "
                          "important to the data itself. They are intended to hold data without constricting "
                          "it to a pre-defined model. Instead, the data is stored like a map of references.",
                "metadata": {
                    "section": "overview",
                }
            },
            {
                "content": "Cypher is Neo4j's graph query language that allows users to store and retrieve data "
                          "from the graph database. It is a declarative, SQL-inspired language for describing "
                          "visual patterns in graphs using ASCII-art syntax.",
                "metadata": {
                    "section": "technical",
                }
            },
            {
                "content": "Neo4j is used by thousands of organizations across industries including financial services, "
                          "retail, manufacturing, telecommunications, and healthcare. Common use cases include fraud detection, "
                          "recommendation engines, network management, and knowledge graphs.",
                "metadata": {
                    "section": "applications",
                }
            }
        ]
        
        # Create TextChunk objects with content and metadata
        text_chunks = [
            TextChunk(
                text=section["content"],
                index=idx,
                metadata=section["metadata"]
            )
            for idx, section in enumerate(mock_sections)
        ]
        data = {}
        data["metadata"] = {
            "title": "This is about rats going to space",
            "authors": ["Hani"],
            "data":"2022"
        }
        data["chunks"] = text_chunks
        return data
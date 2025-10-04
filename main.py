import os
import asyncio
from ms_graphrag_neo4j.ms_graphrag import MsGraphRAG
from ms_graphrag_neo4j.text_chunker.section_splitter import SectionSplitter
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"], 
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

ms_graph = MsGraphRAG(driver=driver, model='gpt-5-mini')


allowed_entities = ["Person", "Organization", "Location"]
text_chunker = SectionSplitter()
async def main():
    # Extract entities and relationships
    await ms_graph.run(text_chunker.run("html"),allowed_entities=allowed_entities)

    # Close the connection (sync)
    ms_graph.close()

if __name__ == "__main__":
    asyncio.run(main())

import os
import asyncio
from ms_graphrag_neo4j.ms_graphrag import MsGraphRAG
from ms_graphrag_neo4j.text_chunker.section_splitter import SectionSplitter
from neo4j import GraphDatabase
from dotenv import load_dotenv
from ms_graphrag_neo4j.article_downloader.handler import get_sample
load_dotenv()

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"], 
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

ms_graph = MsGraphRAG(driver=driver, model='gpt-5-mini')


allowed_entities = [
    "Person",
    "Organization",
    "Location",
    "ResearchInstitution",
    "SpaceMission",
    "CelestialBody",
    "Species",
    "ChemicalSubstance",
    "Molecule",
    "Gene",
    "Protein",
    "Process",
    "Phenomenon",
    "Measurement",
    "Instrument",
    "Dataset",
    "Publication",
    "Experiment",
    "Method",
    "Theory",
    "Model",
    "Equation",
    "Date",
    "Unit",
    "Telescope",
    "Satellite",
    "Planet",
    "Region",
    "Environment",
    "Discipline"
]

allowed_relationships = [
    # General / Hierarchical
    "IS_A",
    "PART_OF",
    "HAS_A",
    "RELATED_TO",
    "DERIVED_FROM",
    
    # Professional / Organizational
    "WORKS_FOR",
    "FOUNDED_BY",
    "CEO_OF",
    "MANAGES",
    "PARTNERED_WITH",
    "ACQUIRED",
    "INVESTED_IN",
    "COMPETES_WITH",
    "CUSTOMER_OF",

    # Locational
    "LOCATED_IN",
    "BASED_IN",
    "BORN_IN",
    
    # Technical / Conceptual
    "USES",
    "DEVELOPS",
    "SUPPORTS",
    "DEPENDS_ON",
    "INTEGRATES_WITH",
    
    # Causal / Influence
    "CAUSES",
    "INFLUENCES",
    "PREcedes",
    
    # Personal
    "SPOUSE_OF",
    "CHILD_OF",
    "KNOWS",
]
text_chunker = SectionSplitter()
async def main():
    # Extract entities and relationships
    await ms_graph.load_data(get_sample(),allowed_entities=allowed_entities,allowed_relationships=allowed_relationships)

    # Close the connection (sync)
    ms_graph.close()

if __name__ == "__main__":
    asyncio.run(main())

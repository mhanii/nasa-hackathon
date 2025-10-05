import asyncio
import csv
import os
from neo4j import GraphDatabase
from ms_graphrag_neo4j.article_downloader.handler import extract_ncbi_article
from ms_graphrag_neo4j.ms_graphrag import MsGraphRAG

# Load environment variables if .env file exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv not found, relying on environment variables set manually.")

# Allowed entity types for extraction
ALLOWED_ENTITIES = [
    "organization",
    "person",
    "drug",
    "disease",
    "symptom",
    "gene",
    "protein",
    "medical_condition",
]
ALLOWED_RELATIONSHIPS = [
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

async def load_documents_to_graph(
    csv_file_path: str,
    start_index: int = 0,
    end_index: int = None,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None
):
    """
    Load documents from a CSV file into a Neo4j graph database.
    
    Args:
        csv_file_path: Path to the input CSV file
        start_index: Start index of documents to process (default: 0)
        end_index: End index of documents to process (default: None, processes all)
        neo4j_uri: Neo4j connection URI (default: from NEO4J_URI env var)
        neo4j_user: Neo4j username (default: from NEO4J_USERNAME env var)
        neo4j_password: Neo4j password (default: from NEO4J_PASSWORD env var)
    
    Returns:
        dict: Summary of processing results with counts of success/failures
    """
    # Get Neo4j credentials from environment if not provided
    uri = neo4j_uri or os.getenv("NEO4J_URI")
    user = neo4j_user or os.getenv("NEO4J_USERNAME")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise ValueError(
            "Neo4j credentials must be provided either as arguments or through "
            "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables."
        )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    graph_rag = MsGraphRAG(driver)
    
    results = {
        "total": 0,
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "failures": []
    }

    try:
        with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
            reader = list(csv.DictReader(csvfile))
            results["total"] = len(reader)

            # Set end_index to length if not specified
            if end_index is None:
                end_index = len(reader)
            else:
                end_index = min(end_index, len(reader))

            if start_index >= end_index:
                print("Start index is greater than or equal to end index. Nothing to process.")
                return results

            for i in range(start_index, end_index):
                row = reader[i]
                link = row.get("Link")
                
                if not link:
                    print(f"Skipping row {i+1} due to missing Link.")
                    results["skipped"] += 1
                    continue

                print(f"\nProcessing document {i+1}/{len(reader)}: {link}")
                
                try:
                    # Download article
                    article_data = extract_ncbi_article(link)

                    if article_data:
                        # Load data into the graph
                        await graph_rag.load_data(article_data, ALLOWED_ENTITIES,ALLOWED_RELATIONSHIPS)
                        print(f"✓ Successfully loaded document: {link}")
                        results["processed"] += 1
                    else:
                        print(f"✗ Could not retrieve article for: {link}")
                        results["failed"] += 1
                        results["failures"].append({
                            "index": i+1,
                            "link": link,
                            "error": "Could not retrieve article"
                        })
                        
                except Exception as e:
                    print(f"✗ Failed to process document '{link}'. Error: {e}")
                    results["failed"] += 1
                    results["failures"].append({
                        "index": i+1,
                        "link": link,
                        "error": str(e)
                    })
        graph_rag.run_graph_diagnostics()

    finally:
        graph_rag.close()

        driver.close()
    

    return results


def print_summary(results: dict):
    """Print a summary of processing results."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total documents in CSV: {results['total']}")
    print(f"Successfully processed: {results['processed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    
    if results['failures']:
        print(f"\nFailed documents:")
        for failure in results['failures'][:10]:  # Show first 10
            print(f"  [{failure['index']}] {failure['link']}")
            print(f"      Error: {failure['error'][:100]}")
        if len(results['failures']) > 10:
            print(f"  ... and {len(results['failures']) - 10} more")


# Example usage functions
async def process_all_documents(csv_file_path: str):
    """Process all documents in the CSV file."""
    results = await load_documents_to_graph(csv_file_path)
    print_summary(results)
    return results


async def process_document_range(csv_file_path: str, start: int, end: int):
    """Process a specific range of documents."""
    results = await load_documents_to_graph(
        csv_file_path=csv_file_path,
        start_index=start,
        end_index=end
    )
    print_summary(results)
    return results


async def process_single_document(csv_file_path: str, index: int):
    """Process a single document at the given index."""
    results = await load_documents_to_graph(
        csv_file_path=csv_file_path,
        start_index=index,
        end_index=index + 1
    )
    print_summary(results)
    return results


# Main execution block for testing
if __name__ == "__main__":
    # Example 1: Process all documents
    # asyncio.run(process_all_documents("yourfile.csv"))
    
    # Example 2: Process documents 0-10
    asyncio.run(process_document_range("SB_publication_PMC.csv", start=0, end=10))
    
    # Example 3: Process a single document at index 5
    # asyncio.run(process_single_document("yourfile.csv", index=5))
    
    # Example 4: Direct usage with custom parameters
    # results = asyncio.run(load_documents_to_graph(
    #     csv_file_path="yourfile.csv",
    #     start_index=0,
    #     end_index=50,
    #     neo4j_uri="bolt://localhost:7687",
    #     neo4j_user="neo4j",
    #     neo4j_password="password"
    # ))
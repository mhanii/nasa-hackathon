import argparse
import asyncio
import csv
import os
from neo4j import GraphDatabase
from ms_graphrag_neo4j.article_downloader.handler import extract_ncbi_article
from ms_graphrag_neo4j.ms_graphrag import MsGraphRAG

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


async def main():
    """
    Main function to parse arguments and run the document loading process.
    """
    parser = argparse.ArgumentParser(
        description="Load documents from a CSV file into the graph."
    )
    parser.add_argument(
        "--input-file", required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        required=True,
        help="Start index of documents to process.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        required=True,
        help="End index of documents to process.",
    )
    args = parser.parse_args()

    # Neo4j connection details from environment variables
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise ValueError(
            "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables must be set."
        )

    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Initialize MsGraphRAG
    graph_rag = MsGraphRAG(driver)

    try:
        with open(args.input_file, newline="", encoding="utf-8") as csvfile:
            reader = list(csv.DictReader(csvfile))

            # Clamp end_index to the number of rows
            end_index = min(args.end_index, len(reader))

            if args.start_index >= end_index:
                print("Start index is greater than or equal to end index. Nothing to process.")
                return

            for i in range(args.start_index, end_index):
                row = reader[i]
                title = row.get("title")  # Assuming the column is named 'title'
                if not title:
                    print(f"Skipping row {i+1} due to missing title.")
                    continue

                print(f"Processing document {i+1}/{len(reader)}: {title}")
                try:
                    # Download article
                    article_data = extract_ncbi_article(title)

                    if article_data:
                        # Load data into the graph
                        await graph_rag.load_data(article_data, ALLOWED_ENTITIES)
                        print(f"Successfully loaded document: {title}")
                    else:
                        print(f"Could not retrieve article for: {title}")
                except Exception as e:
                    print(f"Failed to process document '{title}'. Error: {e}")

    finally:
        graph_rag.close()
        driver.close()


if __name__ == "__main__":
    # Load environment variables if .env file exists
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print("dotenv not found, relying on environment variables set manually.")

    asyncio.run(main())
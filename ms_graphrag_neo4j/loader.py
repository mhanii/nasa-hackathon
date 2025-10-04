import csv
from ms_graphrag_neo4j.article_downloader.handler import extract_ncbi_article
from ms_graphrag_neo4j.ms_graphrag import MsGraphRAG
csv_file_path = 'yourfile.csv'



with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        title = row['title']  # Adjust if the title column name differs
        result = extract_ncbi_article(title)
        # Do something with result if needed, like print or store it
        print(result)

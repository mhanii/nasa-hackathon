




from typing import Dict


class ArticleDownloader: 

    def __init__(self):
        pass

    def download_article(self, link: str) -> str:
        # Dummy implementation for illustration purposes
        return "Full article content from " + link
    
    def parse_article(self, content: str) -> Dict[str, str]:
        # Dummy implementation for illustration purposes
        return {
            "title": "Sample Article Title",
            "author": "Author Name",
            
        }
    def split_into_sections(self, link: str) -> list[Dict[str, str]]:
        # Dummy implementation for illustration purposes
        return [
            {"title": "Abstract", "content": "This is the abstract of the article " + link},
            {"title": "Section 2", "content": "Section 2 content from " + link},
            {"title": "Section 3", "content": "Section 3 content from " + link},
        ]
    
    def run(self, link: str) -> Dict[str, str]:
        content = self.download_article(link)
        metadata = self.parse_article(content)
        return {**metadata, "content": content}
        return metadata
    
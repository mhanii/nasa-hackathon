import requests
from bs4 import BeautifulSoup
import json

def extract_ncbi_article(url):
    """
    Extrae un artículo completo de NCBI PMC con metadata y contenido por secciones.
    
    Args:
        url: URL del artículo en NCBI PMC
        
    Returns:
        dict: Diccionario con metadata y secciones del artículo
    """
    try:
        # Configurar headers para evitar bloqueo 403
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Realizar petición HTTP
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parsear HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Inicializar estructura del artículo
        article = {
            'metadata': {},
            'sections': []
        }
        
        # Extraer metadata
        # Título - intentar diferentes selectores
        title_tag = soup.find('h1', class_='content-title')
        if not title_tag:
            title_tag = soup.find('h1', class_='heading-title')
        if not title_tag:
            # Buscar cualquier h1 en el artículo
            title_tag = soup.find('h1')
        
        if title_tag:
            article['metadata']['title'] = title_tag.get_text(strip=True)
        else:
            article['metadata']['title'] = 'Título no encontrado'
        
        # Autores - intentar diferentes selectores
        authors = []
        
        # Método 1: buscar en contrib-group
        author_section = soup.find('div', class_='contrib-group')
        if author_section:
            author_tags = author_section.find_all('a', class_='contrib-name')
            authors = [author.get_text(strip=True) for author in author_tags]
        
        # Método 2: buscar enlaces de autores directamente
        if not authors:
            author_links = soup.find_all('a', class_='author-name')
            authors = [author.get_text(strip=True) for author in author_links]
        
        # Método 3: buscar en la sección de autores por aria-label
        if not authors:
            citation_section = soup.find('section', {'aria-label': 'Article citation and metadata'})
            if citation_section:
                author_links = citation_section.find_all('a', href=lambda x: x and '/pubmed/' in x)
                authors = [author.get_text(strip=True) for author in author_links]
        
        # Método 4: buscar todos los enlaces de autores en el header
        if not authors:
            # Buscar en cualquier elemento que contenga información de autores
            author_elements = soup.find_all('a', href=lambda x: x and 'term=' in x and 'author' in x.lower())
            authors = [author.get_text(strip=True) for author in author_elements if author.get_text(strip=True)]
        
        article['metadata']['authors'] = authors if authors else ['Autores no encontrados']
        
        # Extraer Abstract
        abstract_section = soup.find('section', class_='abstract', id='abstract1')
        if abstract_section:
            # Buscar el título del abstract
            abstract_title_tag = abstract_section.find('h2')
            abstract_title = abstract_title_tag.get_text(strip=True) if abstract_title_tag else 'Abstract'
            
            # Extraer contenido de los párrafos
            abstract_paragraphs = abstract_section.find_all('p')
            abstract_content = '\n\n'.join([p.get_text(strip=True) for p in abstract_paragraphs])
            
            article['sections'].append({
                'section_id': 'abstract',
                'title': abstract_title,
                'text': abstract_content
            })
        
        # Extraer secciones principales (s1, s2, s3, etc.)
        # Buscar todas las secciones con id que empiece con 's'
        main_body = soup.find('section', class_='body main-article-body')
        if main_body:
            sections = main_body.find_all('section', id=True)
            
            for section in sections:
                section_id = section.get('id', '')
                
                # Solo procesar secciones con id tipo s1, s2, s3, etc.
                if section_id.startswith('s') and section_id[1:].isdigit():
                    # Extraer título de la sección
                    title_tag = section.find(['h2', 'h3', 'h4'])
                    section_title = title_tag.get_text(strip=True) if title_tag else f'Section {section_id}'
                    
                    # Extraer contenido de los párrafos
                    paragraphs = section.find_all('p', recursive=True)
                    section_content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs])
                    
                    article['sections'].append({
                        'section_id': section_id,
                        'title': section_title,
                        'text': section_content
                    })
        
        # Buscar sección de Acknowledgments
        ack_section = soup.find('section', id='ack1', class_='ack')
        if ack_section:
            title_tag = ack_section.find('h2')
            ack_title = title_tag.get_text(strip=True) if title_tag else 'Acknowledgments'
            
            paragraphs = ack_section.find_all('p')
            ack_content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs])
            
            article['sections'].append({
                'section_id': 'acknowledgments',
                'title': ack_title,
                'text': ack_content
            })
        
        return article
        
    except requests.RequestException as e:
        print(f"Error al descargar el artículo: {e}")
        return None
    except Exception as e:
        print(f"Error al procesar el artículo: {e}")
        return None

def get_sample():
    url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4136787/"
    
    print("Extrayendo artículo...")
    article = extract_ncbi_article(url)
    
    if article:
        print("\n" + "="*80)
        print("METADATA")
        print("="*80)
        print(f"Título: {article['metadata'].get('title', 'No disponible')}")
        print(f"Autores: {', '.join(article['metadata'].get('authors', []))}")
        
        print("\n" + "="*80)
        print("SECCIONES")
        print("="*80)
        
        for i, section in enumerate(article['sections'], 1):
            print(f"\n{i}. {section['title']} (ID: {section['section_id']})")
            print("-" * 80)
            # Mostrar solo los primeros 200 caracteres del contenido
            content_preview = section['text'][:200] + "..." if len(section['text']) > 200 else section['text']
            print(content_preview)
        
        # Guardar en JSON
        with open('article_output.json', 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print("Artículo guardado en 'article_output.json'")
        print("="*80)
    else:
        print("No se pudo extraer el artículo.")

    return article
# Probar con el enlace proporcionado
if __name__ == "__main__":
    get_sample()
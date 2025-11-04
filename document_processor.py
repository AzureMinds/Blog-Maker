"""
Document processing module for extracting text from various file formats.
"""
import os
import re
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from bs4 import BeautifulSoup
import markdown

class DocumentProcessor:
    """Handles extraction of text from various document formats."""
    
    def __init__(self, supported_formats: List[str] = None):
        self.supported_formats = supported_formats or ['.pdf', '.html', '.htm', '.txt', '.md']
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_html(self, file_path: str) -> str:
        """Extract text from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            print(f"Error reading HTML {file_path}: {e}")
            return ""
    
    def extract_text_from_markdown(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Convert markdown to HTML then extract text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            
            return text.strip()
        except Exception as e:
            print(f"Error reading Markdown {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from file based on its extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.html', '.htm']:
            return self.extract_text_from_html(file_path)
        elif file_ext == '.md':
            return self.extract_text_from_markdown(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return ""
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all supported files in a folder."""
        documents = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Folder does not exist: {folder_path}")
            return documents
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                print(f"Processing: {file_path}")
                text = self.extract_text_from_file(str(file_path))
                
                if text.strip():  # Only add non-empty documents
                    documents.append({
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'file_type': file_path.suffix.lower(),
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
        
        print(f"Processed {len(documents)} documents from {folder_path}")
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for processing."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks

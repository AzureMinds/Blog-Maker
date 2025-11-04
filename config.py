"""
Configuration settings for the Blog Maker tool.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LLMConfig:
    """Configuration for the LLM model."""
    model_name: str = "llama3.2:3b"  # Default lightweight model
    max_length: int = 1024
    temperature: float = 0.7
    do_sample: bool = True

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_documents: int = 50
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.html', '.htm', '.txt', '.md']

@dataclass
class StyleLearningConfig:
    """Configuration for writing style learning."""
    min_articles_for_style: int = 3
    style_features: List[str] = None
    
    def __post_init__(self):
        if self.style_features is None:
            self.style_features = [
                'sentence_length',
                'paragraph_length', 
                'vocabulary_complexity',
                'punctuation_usage',
                'transition_words'
            ]

@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig = LLMConfig()
    processing: ProcessingConfig = ProcessingConfig()
    style_learning: StyleLearningConfig = StyleLearningConfig()
    
    # Paths
    data_dir: str = "data"
    articles_dir: str = "articles"
    output_dir: str = "output"
    style_samples_dir: str = "style_samples"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.articles_dir, self.output_dir, self.style_samples_dir]:
            os.makedirs(dir_path, exist_ok=True)

# Global configuration instance
config = AppConfig()

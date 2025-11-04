"""
Article generation module that combines document analysis with style learning.
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from document_processor import DocumentProcessor
from ollama_processor import OllamaProcessor
from style_analyzer import StyleAnalyzer
from config import config

class ArticleGenerator:
    """Main class for generating articles from documents using learned style."""
    
    def __init__(self, llm_model: str = None, device: str = "auto"):
        self.document_processor = DocumentProcessor(config.processing.supported_formats)        
        self.llm_processor = OllamaProcessor(model_name=llm_model or config.llm.model_name)
        
        self.style_analyzer = StyleAnalyzer()
        self.style_profile = {}
        
    def load_style_profile(self, style_samples_dir: str = None) -> bool:
        """Load the style profile from past articles."""
        samples_dir = style_samples_dir or config.style_samples_dir
        
        if not os.path.exists(samples_dir):
            print(f"Style samples directory not found: {samples_dir}")
            return False
        
        # Check if style profile already exists
        profile_path = "style_profile.json"
        if os.path.exists(profile_path):
            self.style_profile = self.style_analyzer.load_style_profile(profile_path)
            if self.style_profile:
                print("Loaded existing style profile")
                return True
        
        # Learn style from articles
        print("Learning style from past articles...")
        self.style_profile = self.style_analyzer.learn_from_articles(samples_dir)
        
        if self.style_profile:
            print("Style profile created successfully")
            return True
        else:
            print("Failed to create style profile")
            return False
    
    def generate_article(self, 
                        documents_folder: str,
                        topic: str = "",
                        instructions: str = "",
                        output_file: str = None) -> Dict[str, Any]:
        """Generate an article from documents in a folder."""
        
        print(f"Starting article generation for folder: {documents_folder}")
        
        # Step 1: Process documents
        print("Processing documents...")
        documents = self.document_processor.process_folder(documents_folder)
        
        if not documents:
            return {
                "success": False,
                "error": "No documents found in the specified folder",
                "article": ""
            }
        
        print(f"Processed {len(documents)} documents")
        
        # Step 2: Analyze topic coherence
        print("Analyzing topic coherence...")
        topic_analysis = self.llm_processor.analyze_topic_coherence(documents)
        
        print(f"Topic coherence score: {topic_analysis['coherence_score']:.2f}")
        print(f"Main topics: {topic_analysis['main_topics'][:3]}")
        
        # Step 3: Generate article outline
        print("Generating article outline...")
        outline = self.llm_processor.generate_article_outline(topic_analysis, self.style_profile)
        
        # Step 4: Generate article content
        print("Generating article content...")
        article_content = self._generate_article_content(
            documents, topic_analysis, outline, topic, instructions
        )
        
        # Step 5: Apply style adjustments
        if self.style_profile:
            print("Applying style adjustments...")
            article_content = self._apply_style_adjustments(article_content)
        
        # Step 6: Save article
        if output_file:
            self._save_article(article_content, output_file)
        
        return {
            "success": True,
            "article": article_content,
            "metadata": {
                "documents_processed": len(documents),
                "topic_analysis": topic_analysis,
                "outline": outline,
                "style_applied": bool(self.style_profile),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _generate_article_content(self, 
                                documents: List[Dict[str, Any]],
                                topic_analysis: Dict[str, Any],
                                outline: List[Dict[str, str]],
                                topic: str,
                                instructions: str) -> str:
        """Generate the main article content."""
        
        # Combine all document text
        all_text = " ".join([doc['text'] for doc in documents])
        
        # Create article sections
        article_sections = []
        
        # Introduction
        intro_prompt = f"Write an engaging introduction about {topic or 'this topic'}. "
        intro_prompt += f"Key points to mention: {', '.join(topic_analysis['main_topics'][:3])}. "
        intro_prompt += f"Instructions: {instructions}. "
        intro_prompt += "Keep it concise and engaging."
        
        introduction = self.llm_processor.generate_article_section(
            intro_prompt, max_length=150, temperature=0.7
        )
        article_sections.append(f"# {topic or 'Article'}\n\n{introduction}\n")
        
        # Main sections
        for i, section in enumerate(outline[1:-1], 1):  # Skip intro and conclusion
            section_prompt = f"Write a detailed section about: {section['description']}. "
            section_prompt += f"Use information from these documents: {all_text[:1000]}... "
            section_prompt += f"Instructions: {instructions}. "
            section_prompt += "Make it informative and well-structured."
            
            section_content = self.llm_processor.generate_article_section(
                section_prompt, max_length=300, temperature=0.6
            )
            
            article_sections.append(f"## {section['section']}\n\n{section_content}\n")
        
        # Conclusion
        conclusion_prompt = f"Write a conclusion that summarizes the key points about {topic or 'this topic'}. "
        conclusion_prompt += f"Main topics covered: {', '.join(topic_analysis['main_topics'][:3])}. "
        conclusion_prompt += f"Instructions: {instructions}. "
        conclusion_prompt += "Make it impactful and thought-provoking."
        
        conclusion = self.llm_processor.generate_article_section(
            conclusion_prompt, max_length=150, temperature=0.7
        )
        article_sections.append(f"## Conclusion\n\n{conclusion}\n")
        
        # Combine all sections
        full_article = "\n".join(article_sections)
        
        return full_article
    
    def _apply_style_adjustments(self, article_content: str) -> str:
        """Apply style adjustments based on the learned style profile."""
        # This is a simplified version - in a full implementation,
        # you would use the style profile to make more sophisticated adjustments
        
        style_guidance = self.style_analyzer.get_style_guidance(self.style_profile)
        
        # For now, just add a comment about the style
        if style_guidance:
            style_note = "\n\n<!-- Style guidance applied: " + "; ".join(style_guidance) + " -->"
            article_content += style_note
        
        return article_content
    
    def _save_article(self, article_content: str, output_file: str):
        """Save the generated article to a file."""
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(article_content)
            
            print(f"Article saved to: {output_file}")
        except Exception as e:
            print(f"Error saving article: {e}")
    
    def generate_article_with_custom_prompt(self, 
                                          documents_folder: str,
                                          custom_prompt: str,
                                          output_file: str = None) -> Dict[str, Any]:
        """Generate an article using a custom prompt."""
        
        print(f"Generating article with custom prompt for folder: {documents_folder}")
        
        # Process documents
        documents = self.document_processor.process_folder(documents_folder)
        
        if not documents:
            return {
                "success": False,
                "error": "No documents found in the specified folder",
                "article": ""
            }
        
        # Combine document text
        all_text = " ".join([doc['text'] for doc in documents])
        
        # Generate article using custom prompt
        full_prompt = f"{custom_prompt}\n\nSource material: {all_text[:2000]}..."
        
        article_content = self.llm_processor.generate_article_section(
            full_prompt, max_length=1000, temperature=0.7
        )
        
        # Apply style if available
        if self.style_profile:
            article_content = self._apply_style_adjustments(article_content)
        
        # Save if requested
        if output_file:
            self._save_article(article_content, output_file)
        
        return {
            "success": True,
            "article": article_content,
            "metadata": {
                "documents_processed": len(documents),
                "custom_prompt_used": True,
                "style_applied": bool(self.style_profile),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def get_style_summary(self) -> Dict[str, Any]:
        """Get a summary of the current style profile."""
        if not self.style_profile:
            return {"error": "No style profile loaded"}
        
        style_guidance = self.style_analyzer.get_style_guidance(self.style_profile)
        
        return {
            "style_loaded": True,
            "article_count": self.style_profile.get('article_count', 0),
            "guidance": style_guidance,
            "key_characteristics": {
                "avg_sentence_length": self.style_profile.get('sentence_length', {}).get('mean', 0),
                "avg_paragraph_length": self.style_profile.get('paragraph_length', {}).get('mean', 0),
                "vocabulary_diversity": self.style_profile.get('vocabulary_complexity', {}).get('unique_ratio', 0)
            }
        }

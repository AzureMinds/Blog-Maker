"""
Ollama integration module for local LLM processing.
"""
import ollama
import json
from typing import List, Dict, Any, Optional
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class OllamaProcessor:
    """Handles LLM operations using Ollama for local processing."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and the model is available."""
        try:
            # Check if Ollama is running
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name not in available_models:
                print(f"Model {self.model_name} not found. Available models: {available_models}")
                print(f"Attempting to pull {self.model_name}...")
                try:
                    ollama.pull(self.model_name)
                    print(f"Successfully pulled {self.model_name}")
                except Exception as e:
                    print(f"Error pulling model: {e}")
                    # Fallback to first available model
                    if available_models:
                        self.model_name = available_models[0]
                        print(f"Using fallback model: {self.model_name}")
                    else:
                        raise Exception("No models available in Ollama")
            
            print(f"Using Ollama model: {self.model_name}")
            
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise
        
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize a given text using Ollama."""
        try:
            # Send the full text - let Ollama handle the length
            prompt = f"""Summarize the following text in approximately {max_length} words. 
            Focus on the main ideas, key points, and overall message. 
            Maintain the logical flow and important connections between ideas:

            {text}"""
            
            return self._call_ollama(prompt)
        except Exception as e:
            print(f"Error in summarization: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text

    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points from text."""
        try:
            prompt = f"""Extract the {num_points} most important key points from the following text. 
            Return only the key points, one per line, without numbering or bullets:

            {text[:1500]}..."""
            
            response = self._call_ollama(prompt)
            key_points = [point.strip() for point in response.split('\n') if point.strip()]
            
            # Fallback to simple sentence extraction if Ollama response is poor
            if len(key_points) < 3:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                key_points = sentences[:num_points]
            
            return key_points[:num_points]
        except Exception as e:
            print(f"Error extracting key points: {e}")
            # Fallback to simple sentence selection
            sentences = re.split(r'[.!?]+', text)
            return sentences[:num_points]
    
    def generate_article_section(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate text based on a prompt using Ollama."""
        try:
            enhanced_prompt = f"""Write a well-structured section for an article based on the following prompt. 
            Make it informative, engaging, and approximately {max_length} words:

            {prompt}"""
            
            response = self._call_ollama(enhanced_prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    def analyze_topic_coherence(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how coherent the documents are around a central topic."""
        try:
            if not documents:
                return {"coherence_score": 0, "main_topics": [], "summary": ""}
            
            # Extract all text
            all_text = " ".join([doc['text'] for doc in documents])
            
            # Get key points from all documents
            key_points = self.extract_key_points(all_text, num_points=10)
            
            # Calculate document similarity using embeddings
            doc_embeddings = []
            for doc in documents:
                embedding = self.sentence_model.encode([doc['text']])
                doc_embeddings.append(embedding[0])
            
            if len(doc_embeddings) > 1:
                similarity_matrix = cosine_similarity(doc_embeddings)
                avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            else:
                avg_similarity = 1.0
            
            # Generate summary using Ollama
            summary = self.summarize_text(all_text, max_length=200)
            
            return {
                "coherence_score": float(avg_similarity),
                "main_topics": key_points[:5],
                "summary": summary,
                "document_count": len(documents),
                "total_words": sum(doc['word_count'] for doc in documents)
            }
        except Exception as e:
            print(f"Error analyzing topic coherence: {e}")
            return {"coherence_score": 0, "main_topics": [], "summary": ""}
    
    def generate_article_outline(self, topic_analysis: Dict[str, Any], style_preferences: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Generate an article outline based on topic analysis."""
        try:
            topics = ", ".join(topic_analysis.get('main_topics', [])[:3])
            
            prompt = f"""Create a detailed outline for an article based on these main topics: {topics}
            
            Provide 4-6 sections with clear descriptions. Format as:
            Section 1: [Title] - [Description]
            Section 2: [Title] - [Description]
            etc."""
            
            response = self._call_ollama(prompt)
            
            # Parse the response into outline structure
            outline = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('Section') or line.startswith('Introduction') or line.startswith('Conclusion')):
                    if ':' in line:
                        title, description = line.split(':', 1)
                        outline.append({
                            "section": title.strip(),
                            "description": description.strip()
                        })
            
            # Fallback outline if parsing fails
            if not outline:
                outline = [
                    {"section": "Introduction", "description": f"Introduce the topic and provide context. Key points: {topics}"},
                    {"section": "Main Discussion", "description": "Explore the main themes and concepts"},
                    {"section": "Conclusion", "description": "Summarize key findings and provide closing thoughts"}
                ]
            
            return outline
        except Exception as e:
            print(f"Error generating outline: {e}")
            # Fallback outline
            return [
                {"section": "Introduction", "description": "Introduce the topic"},
                {"section": "Main Discussion", "description": "Explore the main themes"},
                {"section": "Conclusion", "description": "Summarize key findings"}
            ]
    
    def _call_ollama(self, prompt: str) -> str:
        """Make a call to Ollama with error handling."""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 2000
                }
            )
            return response['response']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks for processing."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []

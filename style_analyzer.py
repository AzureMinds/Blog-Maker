"""
Style analysis module for learning and replicating writing style from past articles.
"""
import re
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from collections import Counter
import json

class StyleAnalyzer:
    """Analyzes writing style from past articles and provides style guidance."""
    
    def __init__(self):
        self.style_features = {
            'sentence_length': [],
            'paragraph_length': [],
            'vocabulary_complexity': [],
            'punctuation_usage': {},
            'transition_words': [],
            'sentence_structure': {},
            'tone_indicators': {}
        }
        self.processed_articles = []
    
    def analyze_article_style(self, text: str, article_title: str = "") -> Dict[str, Any]:
        """Analyze the writing style of a single article."""
        style_data = {
            'title': article_title,
            'sentence_length': self._analyze_sentence_length(text),
            'paragraph_length': self._analyze_paragraph_length(text),
            'vocabulary_complexity': self._analyze_vocabulary_complexity(text),
            'punctuation_usage': self._analyze_punctuation_usage(text),
            'transition_words': self._analyze_transition_words(text),
            'sentence_structure': self._analyze_sentence_structure(text),
            'tone_indicators': self._analyze_tone_indicators(text)
        }
        return style_data
    
    def _analyze_sentence_length(self, text: str) -> Dict[str, float]:
        """Analyze sentence length patterns."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        lengths = [len(s.split()) for s in sentences]
        
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'count': len(lengths)
        }
    
    def _analyze_paragraph_length(self, text: str) -> Dict[str, float]:
        """Analyze paragraph length patterns."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        lengths = [len(p.split()) for p in paragraphs]
        
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'count': len(lengths)
        }
    
    def _analyze_vocabulary_complexity(self, text: str) -> Dict[str, float]:
        """Analyze vocabulary complexity and diversity."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return {'unique_ratio': 0, 'avg_word_length': 0, 'complex_words': 0}
        
        unique_words = set(words)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Count complex words (more than 6 characters)
        complex_words = sum(1 for word in words if len(word) > 6)
        
        return {
            'unique_ratio': len(unique_words) / len(words),
            'avg_word_length': avg_word_length,
            'complex_words': complex_words / len(words),
            'total_words': len(words),
            'unique_words': len(unique_words)
        }
    
    def _analyze_punctuation_usage(self, text: str) -> Dict[str, float]:
        """Analyze punctuation usage patterns."""
        total_chars = len(text)
        if total_chars == 0:
            return {}
        
        punctuation_counts = {
            'periods': text.count('.'),
            'commas': text.count(','),
            'semicolons': text.count(';'),
            'colons': text.count(':'),
            'exclamations': text.count('!'),
            'questions': text.count('?'),
            'dashes': text.count('—') + text.count('-'),
            'parentheses': text.count('(') + text.count(')'),
            'quotes': text.count('"') + text.count("'")
        }
        
        # Convert to ratios
        punctuation_ratios = {
            key: count / total_chars for key, count in punctuation_counts.items()
        }
        
        return punctuation_ratios
    
    def _analyze_transition_words(self, text: str) -> Dict[str, float]:
        """Analyze use of transition words and phrases."""
        transition_words = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'contrast': ['however', 'nevertheless', 'on the other hand', 'conversely', 'although'],
            'cause': ['because', 'since', 'due to', 'as a result', 'therefore'],
            'sequence': ['first', 'second', 'next', 'then', 'finally', 'subsequently'],
            'emphasis': ['indeed', 'certainly', 'undoubtedly', 'clearly', 'obviously']
        }
        
        text_lower = text.lower()
        transition_usage = {}
        
        for category, words in transition_words.items():
            count = sum(text_lower.count(word) for word in words)
            transition_usage[category] = count
        
        total_words = len(text.split())
        if total_words == 0:
            return {category: 0 for category in transition_words.keys()}
        
        return {
            category: count / total_words 
            for category, count in transition_usage.items()
        }
    
    def _analyze_sentence_structure(self, text: str) -> Dict[str, float]:
        """Analyze sentence structure patterns."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {'simple_ratio': 0, 'complex_ratio': 0, 'compound_ratio': 0}
        
        simple_count = 0
        complex_count = 0
        compound_count = 0
        
        for sentence in sentences:
            # Simple sentence: one independent clause
            if ',' not in sentence and ';' not in sentence:
                simple_count += 1
            # Complex sentence: has dependent clause indicators
            elif any(word in sentence.lower() for word in ['because', 'although', 'since', 'while', 'if', 'when']):
                complex_count += 1
            # Compound sentence: has coordinating conjunctions
            elif any(word in sentence.lower() for word in ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']):
                compound_count += 1
        
        total = len(sentences)
        return {
            'simple_ratio': simple_count / total,
            'complex_ratio': complex_count / total,
            'compound_ratio': compound_count / total
        }
    
    def _analyze_tone_indicators(self, text: str) -> Dict[str, float]:
        """Analyze tone indicators in the text."""
        text_lower = text.lower()
        
        # Define tone indicators
        tone_indicators = {
            'formal': ['therefore', 'furthermore', 'consequently', 'nevertheless', 'accordingly'],
            'informal': ['actually', 'basically', 'literally', 'totally', 'really'],
            'academic': ['research', 'study', 'analysis', 'findings', 'conclusion'],
            'personal': ['i think', 'i believe', 'in my opinion', 'personally', 'i feel'],
            'authoritative': ['clearly', 'obviously', 'undoubtedly', 'certainly', 'definitely']
        }
        
        tone_scores = {}
        total_words = len(text.split())
        
        if total_words == 0:
            return {tone: 0 for tone in tone_indicators.keys()}
        
        for tone, indicators in tone_indicators.items():
            count = sum(text_lower.count(indicator) for indicator in indicators)
            tone_scores[tone] = count / total_words
        
        return tone_scores
    
    def learn_from_articles(self, articles_dir: str) -> Dict[str, Any]:
        """Learn style patterns from a directory of past articles."""
        articles_path = Path(articles_dir)
        
        if not articles_path.exists():
            print(f"Articles directory does not exist: {articles_dir}")
            return {}
        
        # Find all text files in the directory
        text_files = []
        for ext in ['.txt', '.md', '.html', '.htm']:
            text_files.extend(articles_path.glob(f'**/*{ext}'))
        
        if not text_files:
            print(f"No articles found in {articles_dir}")
            return {}
        
        print(f"Analyzing {len(text_files)} articles for style patterns...")
        
        all_style_data = []
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                style_data = self.analyze_article_style(content, file_path.name)
                all_style_data.append(style_data)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if not all_style_data:
            return {}
        
        # Aggregate style patterns
        aggregated_style = self._aggregate_style_patterns(all_style_data)
        
        # Save the learned style
        self._save_style_profile(aggregated_style)
        
        return aggregated_style
    
    def _aggregate_style_patterns(self, style_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate style patterns from multiple articles."""
        aggregated = {}
        
        # Aggregate sentence length
        sentence_lengths = [data['sentence_length'] for data in style_data_list]
        aggregated['sentence_length'] = {
            'mean': np.mean([sl['mean'] for sl in sentence_lengths]),
            'std': np.mean([sl['std'] for sl in sentence_lengths]),
            'min': np.mean([sl['min'] for sl in sentence_lengths]),
            'max': np.mean([sl['max'] for sl in sentence_lengths])
        }
        
        # Aggregate paragraph length
        paragraph_lengths = [data['paragraph_length'] for data in style_data_list]
        aggregated['paragraph_length'] = {
            'mean': np.mean([pl['mean'] for pl in paragraph_lengths]),
            'std': np.mean([pl['std'] for pl in paragraph_lengths]),
            'min': np.mean([pl['min'] for pl in paragraph_lengths]),
            'max': np.mean([pl['max'] for pl in paragraph_lengths])
        }
        
        # Aggregate vocabulary complexity
        vocab_complexities = [data['vocabulary_complexity'] for data in style_data_list]
        aggregated['vocabulary_complexity'] = {
            'unique_ratio': np.mean([vc['unique_ratio'] for vc in vocab_complexities]),
            'avg_word_length': np.mean([vc['avg_word_length'] for vc in vocab_complexities]),
            'complex_words': np.mean([vc['complex_words'] for vc in vocab_complexities])
        }
        
        # Aggregate punctuation usage
        punct_usages = [data['punctuation_usage'] for data in style_data_list]
        aggregated['punctuation_usage'] = {}
        for punct_type in punct_usages[0].keys():
            aggregated['punctuation_usage'][punct_type] = np.mean([
                pu.get(punct_type, 0) for pu in punct_usages
            ])
        
        # Aggregate transition words
        transition_usages = [data['transition_words'] for data in style_data_list]
        aggregated['transition_words'] = {}
        for transition_type in transition_usages[0].keys():
            aggregated['transition_words'][transition_type] = np.mean([
                tu.get(transition_type, 0) for tu in transition_usages
            ])
        
        # Aggregate sentence structure
        sentence_structures = [data['sentence_structure'] for data in style_data_list]
        aggregated['sentence_structure'] = {}
        for structure_type in sentence_structures[0].keys():
            aggregated['sentence_structure'][structure_type] = np.mean([
                ss.get(structure_type, 0) for ss in sentence_structures
            ])
        
        # Aggregate tone indicators
        tone_indicators = [data['tone_indicators'] for data in style_data_list]
        aggregated['tone_indicators'] = {}
        for tone_type in tone_indicators[0].keys():
            aggregated['tone_indicators'][tone_type] = np.mean([
                ti.get(tone_type, 0) for ti in tone_indicators
            ])
        
        aggregated['article_count'] = len(style_data_list)
        
        return aggregated
    
    def _save_style_profile(self, style_profile: Dict[str, Any], filename: str = "style_profile.json"):
        """Save the learned style profile to a file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(style_profile, f, indent=2, ensure_ascii=False)
            print(f"Style profile saved to {filename}")
        except Exception as e:
            print(f"Error saving style profile: {e}")
    
    def load_style_profile(self, filename: str = "style_profile.json") -> Dict[str, Any]:
        """Load a previously saved style profile."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"Style profile file not found: {filename}")
                return {}
        except Exception as e:
            print(f"Error loading style profile: {e}")
            return {}
    
    def get_style_guidance(self, style_profile: Dict[str, Any]) -> List[str]:
        """Generate style guidance based on the learned profile."""
        guidance = []
        
        if not style_profile:
            return ["No style profile available. Please analyze some articles first."]
        
        # Sentence length guidance
        avg_sentence_length = style_profile.get('sentence_length', {}).get('mean', 0)
        if avg_sentence_length > 20:
            guidance.append(f"Use longer sentences (average: {avg_sentence_length:.1f} words)")
        elif avg_sentence_length < 15:
            guidance.append(f"Use shorter sentences (average: {avg_sentence_length:.1f} words)")
        
        # Paragraph length guidance
        avg_paragraph_length = style_profile.get('paragraph_length', {}).get('mean', 0)
        if avg_paragraph_length > 100:
            guidance.append(f"Write longer paragraphs (average: {avg_paragraph_length:.1f} words)")
        elif avg_paragraph_length < 50:
            guidance.append(f"Write shorter paragraphs (average: {avg_paragraph_length:.1f} words)")
        
        # Vocabulary guidance
        unique_ratio = style_profile.get('vocabulary_complexity', {}).get('unique_ratio', 0)
        if unique_ratio > 0.7:
            guidance.append("Use diverse vocabulary")
        elif unique_ratio < 0.5:
            guidance.append("Consider using more varied vocabulary")
        
        # Tone guidance
        tone_indicators = style_profile.get('tone_indicators', {})
        dominant_tone = max(tone_indicators.items(), key=lambda x: x[1]) if tone_indicators else None
        if dominant_tone:
            guidance.append(f"Maintain a {dominant_tone[0]} tone")
        
        # Transition words guidance
        transition_words = style_profile.get('transition_words', {})
        if transition_words.get('contrast', 0) > 0.01:
            guidance.append("Use contrast transitions to present different viewpoints")
        if transition_words.get('sequence', 0) > 0.01:
            guidance.append("Use sequence transitions to organize ideas")
        
        return guidance

"""
Streamlit web interface for the Blog Maker tool.
"""
import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime
from article_generator import ArticleGenerator
from ollama_processor import OllamaProcessor
from config import config
import threading
import time

# Page configuration
st.set_page_config(
    page_title="Blog Maker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_ollama_health_async(processor, status_container):
    """Check Ollama health asynchronously and update UI."""
    health = processor.check_ollama_health()
    status_container.empty()
    
    if health["connected"]:
        if health["model_available"]:
            status_container.success(f"✅ Connected to Ollama. Model '{health['selected_model']}' is available.")
        else:
            status_container.warning(f"⚠️ Connected to Ollama, but model '{health['selected_model']}' is not available.")
            status_container.info(f"Available models: {', '.join(health['available_models']) if health['available_models'] else 'None'}")
            if st.button(f"Pull Model '{health['selected_model']}'", key="pull_model_btn"):
                with st.spinner(f"Pulling model {health['selected_model']}... This may take a few minutes."):
                    result = processor.pull_model()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")
    else:
        status_container.error(f"❌ Cannot connect to Ollama: {health.get('error', 'Unknown error')}")
        status_container.info("Make sure Ollama is running: `ollama serve`")

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📝 Blog Maker</h1>', unsafe_allow_html=True)
    st.markdown("Transform your documents into well-crafted articles using AI and your personal writing style.")
    
    # Initialize session state
    if 'article_generator' not in st.session_state:
        st.session_state.article_generator = None
    if 'style_loaded' not in st.session_state:
        st.session_state.style_loaded = False
    if 'ollama_models' not in st.session_state:
        st.session_state.ollama_models = []
    if 'ollama_processor' not in st.session_state:
        st.session_state.ollama_processor = None
    if 'health_check_done' not in st.session_state:
        st.session_state.health_check_done = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Backend selection
        st.subheader("LLM Backend")

        # Ollama model selection
        st.subheader("Ollama Model")
        
        # Initialize processor without connection check
        if st.session_state.ollama_processor is None:
            try:
                st.session_state.ollama_processor = OllamaProcessor(
                    model_name=config.llm.model_name,
                    check_connection=False
                )
            except Exception as e:
                st.error(f"Error creating processor: {e}")
        
        # Health check section
        health_status = st.container()
        
        if st.button("Check Ollama Status", key="check_health"):
            with st.spinner("Checking Ollama connection..."):
                check_ollama_health_async(st.session_state.ollama_processor, health_status)
                st.session_state.health_check_done = True
        
        # Auto-check on first load
        if not st.session_state.health_check_done and st.session_state.ollama_processor:
            with st.spinner("Checking Ollama connection..."):
                check_ollama_health_async(st.session_state.ollama_processor, health_status)
                st.session_state.health_check_done = True
        
        # Get available Ollama models
        if st.button("Refresh Ollama Models"):
            if st.session_state.ollama_processor:
                with st.spinner("Fetching available models..."):
                    health = st.session_state.ollama_processor.check_ollama_health()
                    if health["connected"]:
                        st.session_state.ollama_models = health["available_models"]
                        st.success("Ollama models refreshed!")
                    else:
                        st.error(f"Error connecting to Ollama: {health.get('error', 'Unknown error')}")
                        st.info("Make sure Ollama is running: `ollama serve`")
        
        # Default Ollama models if none loaded
        if not st.session_state.ollama_models:
            st.session_state.ollama_models = [
                "llama3.2:3b",
                "llama3.2:1b", 
                "phi3:mini",
                "gemma2:2b",
                "qwen2.5:3b"
            ]
        
        selected_model = st.selectbox(
            "Choose Ollama Model:",
            options=st.session_state.ollama_models,
            help="Select the Ollama model for article generation"
        )
        
        # Update processor model name if changed
        if st.session_state.ollama_processor and st.session_state.ollama_processor.model_name != selected_model:
            st.session_state.ollama_processor.model_name = selected_model
        
        # Initialize generator with Ollama
        if st.button("Initialize Ollama Generator", type="primary"):
            if st.session_state.ollama_processor:
                # Check health first
                with st.spinner("Checking model availability..."):
                    health = st.session_state.ollama_processor.check_ollama_health()
                    
                    if not health["connected"]:
                        st.error(f"Cannot connect to Ollama: {health.get('error', 'Unknown error')}")
                        st.info("Make sure Ollama is running: `ollama serve`")
                    elif not health["model_available"]:
                        st.warning(f"Model '{selected_model}' is not available.")
                        st.info(f"Available models: {', '.join(health['available_models']) if health['available_models'] else 'None'}")
                        if st.button(f"Pull Model '{selected_model}'", key="pull_on_init"):
                            with st.spinner(f"Pulling model {selected_model}... This may take several minutes."):
                                result = st.session_state.ollama_processor.pull_model(selected_model)
                                if result["success"]:
                                    st.success(result["message"])
                                    st.rerun()
                                else:
                                    st.error(f"Error: {result['error']}")
                    else:
                        # Model is available, initialize generator
                        with st.spinner("Initializing generator (loading models)..."):
                            try:
                                st.session_state.article_generator = ArticleGenerator(
                                    llm_model=selected_model
                                )
                                st.success("Ollama generator initialized successfully!")
                            except Exception as e:
                                st.error(f"Error initializing generator: {e}")
            else:
                st.error("Ollama processor not initialized")
        
        # Style learning section
        st.subheader("📚 Style Learning")
        
        style_samples_path = st.text_input(
            "Style Samples Directory:",
            value=config.style_samples_dir,
            help="Path to directory containing your past articles for style learning"
        )
        
        if st.button("Load Style Profile"):
            if st.session_state.article_generator:
                with st.spinner("Analyzing writing style..."):
                    success = st.session_state.article_generator.load_style_profile(style_samples_path)
                    if success:
                        st.session_state.style_loaded = True
                        st.success("Style profile loaded successfully!")
                    else:
                        st.error("Failed to load style profile")
            else:
                st.error("Please initialize the generator first")
        
        # Show style summary
        if st.session_state.style_loaded and st.session_state.article_generator:
            style_summary = st.session_state.article_generator.get_style_summary()
            if "error" not in style_summary:
                st.subheader("Style Summary")
                st.write(f"Articles analyzed: {style_summary['article_count']}")
                st.write(f"Avg sentence length: {style_summary['key_characteristics']['avg_sentence_length']:.1f} words")
                st.write(f"Avg paragraph length: {style_summary['key_characteristics']['avg_paragraph_length']:.1f} words")
                st.write(f"Vocabulary diversity: {style_summary['key_characteristics']['vocabulary_diversity']:.2f}")
    
    # Main content area
    if not st.session_state.article_generator:
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to Blog Maker!</h3>
            <p>To get started:</p>
            <ol>
                <li>Choose your LLM backend (Ollama for local/free)</li>
                <li>Configure your model and click "Initialize Generator"</li>
                <li>Optionally load your writing style from past articles</li>
                <li>Select a folder containing your documents</li>
                <li>Generate your article!</li>
            </ol>
            <p><strong>💡 Tip:</strong> If you have Ollama installed, check "Use Ollama" for free local processing!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Article generation section
    st.markdown('<h2 class="section-header">📁 Document Processing</h2>', unsafe_allow_html=True)
    
    # Folder selection
    documents_folder = st.text_input(
        "Documents Folder Path:",
        help="Path to folder containing PDF, HTML, or text files to process"
    )
    
    if documents_folder and os.path.exists(documents_folder):
        st.success(f"✅ Folder found: {documents_folder}")
        
        # Show folder contents
        try:
            folder_path = Path(documents_folder)
            files = list(folder_path.rglob('*'))
            supported_files = [f for f in files if f.suffix.lower() in ['.pdf', '.html', '.htm', '.txt', '.md']]
            
            if supported_files:
                st.write(f"Found {len(supported_files)} supported files:")
                for file in supported_files[:10]:  # Show first 10 files
                    st.write(f"  📄 {file.name}")
                if len(supported_files) > 10:
                    st.write(f"  ... and {len(supported_files) - 10} more files")
            else:
                st.warning("No supported files found in this folder")
        except Exception as e:
            st.error(f"Error reading folder: {e}")
    
    # Article generation options
    st.markdown('<h2 class="section-header">✍️ Article Generation</h2>', unsafe_allow_html=True)
    
    # Generation mode selection
    generation_mode = st.radio(
        "Generation Mode:",
        ["Automatic", "Custom Prompt"],
        help="Choose how to generate the article"
    )
    
    if generation_mode == "Automatic":
        # Automatic generation
        topic = st.text_input(
            "Article Topic (optional):",
            help="Main topic or title for the article"
        )
        
        instructions = st.text_area(
            "Additional Instructions (optional):",
            help="Specific instructions for article generation",
            height=100
        )
        
        if st.button("Generate Article", type="primary"):
            if documents_folder and os.path.exists(documents_folder):
                with st.spinner("Generating article..."):
                    try:
                        result = st.session_state.article_generator.generate_article(
                            documents_folder=documents_folder,
                            topic=topic,
                            instructions=instructions
                        )
                        
                        if result["success"]:
                            st.success("Article generated successfully!")
                            
                            # Display article
                            st.markdown('<h2 class="section-header">📄 Generated Article</h2>', unsafe_allow_html=True)
                            st.markdown(result["article"])
                            
                            # Show metadata
                            with st.expander("Generation Metadata"):
                                st.json(result["metadata"])
                            
                            # Download button
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"article_{timestamp}.md"
                            
                            st.download_button(
                                label="Download Article",
                                data=result["article"],
                                file_name=filename,
                                mime="text/markdown"
                            )
                        else:
                            st.error(f"Error: {result['error']}")
                    except Exception as e:
                        st.error(f"Error generating article: {e}")
            else:
                st.error("Please provide a valid documents folder path")
    
    else:
        # Custom prompt generation
        custom_prompt = st.text_area(
            "Custom Prompt:",
            help="Write a detailed prompt for article generation. The AI will use this along with your documents.",
            height=150,
            placeholder="Write an article about the main themes in these documents. Focus on practical applications and include specific examples..."
        )
        
        if st.button("Generate with Custom Prompt", type="primary"):
            if documents_folder and os.path.exists(documents_folder) and custom_prompt:
                with st.spinner("Generating article with custom prompt..."):
                    try:
                        result = st.session_state.article_generator.generate_article_with_custom_prompt(
                            documents_folder=documents_folder,
                            custom_prompt=custom_prompt
                        )
                        
                        if result["success"]:
                            st.success("Article generated successfully!")
                            
                            # Display article
                            st.markdown('<h2 class="section-header">📄 Generated Article</h2>', unsafe_allow_html=True)
                            st.markdown(result["article"])
                            
                            # Show metadata
                            with st.expander("Generation Metadata"):
                                st.json(result["metadata"])
                            
                            # Download button
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"article_{timestamp}.md"
                            
                            st.download_button(
                                label="Download Article",
                                data=result["article"],
                                file_name=filename,
                                mime="text/markdown"
                            )
                        else:
                            st.error(f"Error: {result['error']}")
                    except Exception as e:
                        st.error(f"Error generating article: {e}")
            else:
                st.error("Please provide both a valid documents folder path and custom prompt")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Blog Maker - AI-Powered Article Generation</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

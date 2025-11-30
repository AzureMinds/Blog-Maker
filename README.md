# Blog Maker

A desktop tool that transforms your documents into well-crafted articles using open-source AI and your personal writing style.

## Features

- 📁 **Document Processing**: Supports PDF, HTML, Markdown, and text files
- 🤖 **AI-Powered Generation**: Uses open-source language models for article creation
- 🆓 **Local Processing**: Ollama integration for completely free, local AI processing
- 📚 **Style Learning**: Analyzes your past articles to learn and replicate your writing style
- 🎯 **Topic Analysis**: Automatically identifies key themes and topics from your documents
- ✍️ **Flexible Generation**: Automatic article generation or custom prompt-based creation
- 💻 **Easy-to-Use Interface**: Streamlit-based web interface for desktop use

## Installation

1. **Clone or download this repository**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Optional: Install Ollama for local processing**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull a model (e.g., Llama 3.2 3B)
   ollama pull llama3.2:3b
   
   # Start Ollama server
   ollama serve
   ```
4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Quick Start

1. **Launch the application** by running `streamlit run app.py`
2. **Configure your settings** in the sidebar:
   - **Choose LLM Backend**: 
     - Check "Use Ollama" for free local processing (recommended) - only option currently
   - **Select Model**: Choose from available Ollama models
   - Click "Initialize Generator"
3. **Learn your writing style** (optional but recommended):
   - Create a folder with your past articles (text, markdown, or HTML files)
   - Enter the path in "Style Samples Directory"
   - Click "Load Style Profile"
4. **Generate an article**:
   - Enter the path to a folder containing your source documents
   - Choose generation mode (Automatic or Custom Prompt)
   - Click "Generate Article"

## Supported File Formats

- **PDF**: `.pdf`
- **HTML**: `.html`, `.htm`
- **Markdown**: `.md`
- **Text**: `.txt`

## Configuration

### LLM Models

The tool supports two backends:

#### Ollama Models (Recommended - Free & Local)
- **llama3.2:3b**: Good balance of speed and quality (recommended)
- **llama3.2:1b**: Fastest, basic quality
- **phi3:mini**: Microsoft's efficient model
- **gemma2:2b**: Google's efficient model
- **qwen2.5:3b**: Alibaba's efficient model

### Style Learning

The style analyzer learns from your past articles by analyzing:

- Sentence length patterns
- Paragraph structure
- Vocabulary complexity and diversity
- Punctuation usage
- Transition word frequency
- Sentence structure types
- Tone indicators

## Directory Structure

```
Blog Maker/
├── app.py                 # Main Streamlit application
├── article_generator.py   # Core article generation logic
├── document_processor.py  # Document text extraction
├── style_analyzer.py     # Writing style analysis
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Default data directory
├── articles/            # Default articles directory
├── output/              # Generated articles output
└── style_samples/       # Your past articles for style learning
```

### Custom Prompt Generation

1. Place your source documents in a folder
2. Enter the folder path
3. Write a detailed custom prompt
4. Click "Generate with Custom Prompt"

### Style Learning

To teach the tool your writing style:

1. Create a folder with your past articles
2. Supported formats: text, markdown, HTML
3. Enter the folder path in "Style Samples Directory"
4. Click "Load Style Profile"

The tool will analyze patterns in:
- Sentence and paragraph length
- Vocabulary usage
- Punctuation patterns
- Writing tone and structure

## Troubleshooting

### Common Issues

**"No documents found"**
- Check that your folder path is correct
- Ensure files have supported extensions (.pdf, .html, .htm, .txt, .md)

**"Error loading models"**
- Try using CPU instead of GPU
- Ensure you have enough RAM (4GB+ recommended)
- Try a smaller model like "gpt2"

**"Style profile not found"**
- Make sure your style samples directory contains text files
- Check that files are readable and contain actual text content

### Performance Tips

- Use GPU acceleration if available (select "cuda" or "mps")
- For faster generation, use smaller models like "gpt2"
- Process documents in smaller batches if memory is limited
- Use CPU mode if GPU memory is insufficient

## Advanced Configuration

You can modify `config.py` to customize:

- Default model settings
- Document processing parameters
- Style learning features
- Output directories

## Contributing

This is a personal tool, but feel free to adapt it for your needs. Key areas for improvement:

- Additional file format support
- More sophisticated style learning
- Better article structure generation
- Export to different formats

## License

This tool is provided as-is for personal use. The underlying AI models have their own licenses.

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Internet connection for initial model download
- Optional: GPU for faster processing

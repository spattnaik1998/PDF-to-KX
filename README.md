# PDF Knowledge Graph Extraction

A Python application that automatically extracts knowledge graphs from PDF documents using AI-powered entity and relation extraction.

## What This App Does

This application provides an end-to-end pipeline for converting PDF documents into structured knowledge graphs:

1. **Upload PDF**: Users can upload PDF documents through a web interface
2. **Extract Text**: Text content is extracted from PDF files
3. **Chunk Text**: Documents are split into manageable chunks for processing
4. **Extract Entities & Relations**: AI models identify entities and relationships
5. **Deduplicate**: Similar entities are merged using embedding-based similarity
6. **Build Knowledge Graph**: A structured graph (nodes + edges) is constructed
7. **Visualize**: Interactive graph visualization in the browser

## Tech Stack

### Core Frameworks
- **FastAPI**: RESTful API backend
- **Streamlit**: Interactive web UI
- **Uvicorn**: ASGI server

### PDF & Text Processing
- **PyMuPDF (fitz)**: PDF text extraction

### AI/ML
- **OpenAI API**: GPT-based entity/relation extraction
- **Hugging Face Transformers**: Alternative extraction using REBEL model
- **Sentence Transformers**: Embedding generation for deduplication
- **scikit-learn**: Similarity computation and clustering

### Graph Processing & Visualization
- **NetworkX**: Graph data structure and algorithms
- **Pyvis**: Interactive graph visualization

## Pipeline Overview

```
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Text Chunking
    ↓
Entity/Relation Extraction
    ├─ OpenAI GPT (option 1)
    └─ Hugging Face REBEL (option 2)
    ↓
Entity Deduplication (embeddings + similarity)
    ↓
Knowledge Graph Construction (NetworkX)
    ↓
Visualization (Pyvis) + API Export (JSON)
```

## PDF Parsing

The `app/pdf_utils.py` module provides robust text extraction from PDF documents using PyMuPDF (fitz). It includes multiple extraction methods and comprehensive error handling.

### Features

- **Full Text Extraction**: Extract all text from a PDF as a single string
- **Page-by-Page Extraction**: Extract text while preserving page boundaries
- **Metadata Extraction**: Get page numbers along with text content
- **PDF Validation**: Verify files are valid, readable PDFs
- **Error Handling**: Custom exceptions for clear error messages
- **PDF Info**: Retrieve document metadata (title, author, page count, file size)

### Available Functions

```python
from app.pdf_utils import (
    extract_text_from_pdf,
    extract_text_by_page,
    extract_pages_with_metadata,
    validate_pdf,
    get_pdf_info
)

# Extract all text from a PDF
text = extract_text_from_pdf("document.pdf")
print(f"Extracted {len(text)} characters")

# Extract text by page (returns list of strings)
pages = extract_text_by_page("document.pdf")
print(f"Document has {len(pages)} pages")

# Extract with page metadata (returns list of dicts)
pages_with_meta = extract_pages_with_metadata("document.pdf")
for page in pages_with_meta:
    print(f"Page {page['page_number']}: {len(page['text'])} chars")

# Validate a PDF file
if validate_pdf("document.pdf"):
    print("Valid PDF")

# Get PDF metadata
info = get_pdf_info("document.pdf")
print(f"Title: {info['title']}, Pages: {info['page_count']}")
```

### Error Handling

The module uses custom `PDFExtractionError` exceptions for clear error reporting:

- `FileNotFoundError`: When the PDF file doesn't exist
- `PDFExtractionError`: When the PDF is corrupted or cannot be processed
- `ValueError`: When invalid parameters are provided

### Implementation Details

- Uses PyMuPDF (fitz) for reliable PDF parsing
- Text is extracted using `get_text("text")` method for clean, plain text
- Pages are joined with double newlines (`\n\n`) for readability
- All file paths are validated before processing
- Documents are properly closed after extraction to prevent resource leaks

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

Edit `.env` to set:
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI extraction)
- `HF_TOKEN`: Hugging Face token (optional, for gated models)
- `EXTRACTION_ENGINE`: Choose "openai" or "rebel"
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Text chunking parameters
- `DEDUP_THRESHOLD`: Similarity threshold for entity deduplication

## Usage

### Run API Server
```bash
uvicorn app.api:app --reload
```

### Run Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration and environment variables
│   ├── pdf_utils.py           # PDF text extraction
│   ├── chunking.py            # Text chunking utilities
│   ├── extraction_openai.py   # OpenAI-based extraction
│   ├── extraction_hf.py       # Hugging Face model extraction
│   ├── deduplication.py       # Entity deduplication logic
│   ├── graph_builder.py       # Knowledge graph construction
│   └── api.py                 # FastAPI routes
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py       # Streamlit web interface
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT

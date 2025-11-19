"""
FastAPI application for PDF Knowledge Graph extraction.

This module provides REST API endpoints for uploading PDFs and
extracting knowledge graphs using the complete pipeline:
PDF → Text → Chunks → Entity/Relation Extraction → Deduplication → Graph

Example usage:
    # Start the server
    uvicorn app.api:app --reload

    # Upload PDF and get knowledge graph
    curl -X POST http://localhost:8000/pdf-to-kg \\
        -F "file=@document.pdf" \\
        -F "engine=openai"
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import tempfile
import os
import traceback

# Import pipeline components
from app.config import settings, validate_config
from app.pdf_utils import extract_pages_with_metadata, extract_text_from_pdf
from app.chunking import chunk_pages, simple_chunk
from app.extraction_openai import extract_from_chunks as extract_openai
from app.extraction_hf import extract_from_chunks as extract_hf
from app.deduplication import deduplicate_entities
from app.graph_builder import build_graph_json, get_graph_statistics


# Initialize FastAPI app
app = FastAPI(
    title="PDF to Knowledge Graph API",
    description="Extract knowledge graphs from PDF documents using AI-powered entity and relation extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class GraphResponse(BaseModel):
    """Response model for graph data."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    statistics: Dict[str, Any]


class PipelineStatus(BaseModel):
    """Status response for pipeline execution."""
    status: str
    message: str
    step: Optional[str] = None
    progress: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler - validate configuration.
    """
    try:
        validate_config()
        print("✓ Configuration validated successfully")
    except ValueError as e:
        print(f"⚠ Configuration warning: {e}")
        print("  Some features may not be available")


@app.get("/")
async def root():
    """
    Root endpoint - API information.

    Returns:
        Dict: API status and version info
    """
    return {
        "name": "PDF to Knowledge Graph API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "pdf_to_kg": "/pdf-to-kg",
            "extract_text": "/extract-text",
            "config": "/config",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Dict: Health status
    """
    return {
        "status": "healthy",
        "service": "pdf-to-kg",
        "version": "1.0.0"
    }


@app.post("/pdf-to-kg", response_model=GraphResponse)
async def pdf_to_knowledge_graph(
    file: UploadFile = File(..., description="PDF file to process"),
    engine: Literal["openai", "rebel"] = Form(
        default="openai",
        description="Extraction engine: 'openai' or 'rebel'"
    ),
    chunk_size: Optional[int] = Form(
        default=None,
        description="Maximum characters per chunk (default from config)"
    ),
    chunk_overlap: Optional[int] = Form(
        default=None,
        description="Overlapping characters between chunks (default from config)"
    ),
    dedup_threshold: Optional[float] = Form(
        default=None,
        description="Entity deduplication similarity threshold 0-1 (default from config)"
    )
):
    """
    Complete PDF to Knowledge Graph pipeline.

    This endpoint executes the full pipeline:
    1. Extract text from PDF
    2. Chunk text into processable segments
    3. Extract entities and relations using selected engine
    4. Deduplicate similar entities
    5. Build knowledge graph JSON

    Args:
        file: PDF file upload
        engine: Extraction engine ("openai" or "rebel")
        chunk_size: Text chunk size (default: from config)
        chunk_overlap: Chunk overlap size (default: from config)
        dedup_threshold: Deduplication threshold (default: from config)

    Returns:
        GraphResponse: Knowledge graph with nodes, edges, and statistics

    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF document"
        )

    # Set defaults from config
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap
    if dedup_threshold is None:
        dedup_threshold = settings.dedup_threshold

    # Validate parameters
    if chunk_size <= chunk_overlap:
        raise HTTPException(
            status_code=400,
            detail="chunk_size must be greater than chunk_overlap"
        )
    if not (0 <= dedup_threshold <= 1):
        raise HTTPException(
            status_code=400,
            detail="dedup_threshold must be between 0 and 1"
        )

    temp_file_path = None

    try:
        # Step 1: Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        print(f"\n{'='*60}")
        print(f"Processing PDF: {file.filename}")
        print(f"Engine: {engine}")
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"Dedup threshold: {dedup_threshold}")
        print(f"{'='*60}\n")

        # Step 2: Extract text from PDF with page metadata
        print("Step 1/5: Extracting text from PDF...")
        pages = extract_pages_with_metadata(temp_file_path)
        print(f"  ✓ Extracted {len(pages)} pages")

        # Step 3: Chunk the pages
        print("\nStep 2/5: Chunking text...")
        chunks = chunk_pages(
            pages,
            max_chars=chunk_size,
            overlap=chunk_overlap
        )
        print(f"  ✓ Created {len(chunks)} chunks")

        # Step 4: Extract entities and relations
        print(f"\nStep 3/5: Extracting entities and relations ({engine})...")
        if engine == "openai":
            extraction_result = extract_openai(chunks, show_progress=True)
        elif engine == "rebel":
            extraction_result = extract_hf(chunks, show_progress=True)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid engine: {engine}. Must be 'openai' or 'rebel'"
            )

        print(f"  ✓ Extracted {extraction_result['entity_count']} entities")
        print(f"  ✓ Extracted {extraction_result['relation_count']} relations")

        # Step 5: Deduplicate entities
        print("\nStep 4/5: Deduplicating entities...")
        dedup_result = deduplicate_entities(
            extraction_result["entities"],
            extraction_result["relations"],
            threshold=dedup_threshold,
            show_progress=True
        )
        print(f"  ✓ Reduced to {dedup_result['final_count']} unique entities")
        print(f"  ✓ Reduced to {len(dedup_result['relations'])} unique relations")

        # Step 6: Build knowledge graph
        print("\nStep 5/5: Building knowledge graph...")
        graph_json = build_graph_json(
            dedup_result["entities"],
            dedup_result["relations"]
        )

        # Get statistics
        statistics = get_graph_statistics(graph_json)
        print(f"  ✓ Final graph: {statistics['num_nodes']} nodes, {statistics['num_edges']} edges")

        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"{'='*60}\n")

        # Return result
        return GraphResponse(
            nodes=graph_json["nodes"],
            edges=graph_json["edges"],
            statistics=statistics
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=f"PDF file error: {str(e)}"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameters: {str(e)}"
        )

    except Exception as e:
        # Log full traceback for debugging
        print(f"\n{'='*60}")
        print("ERROR in pipeline:")
        print(traceback.format_exc())
        print(f"{'='*60}\n")

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")


@app.post("/extract-text")
async def extract_text_from_pdf_endpoint(
    file: UploadFile = File(..., description="PDF file to process")
):
    """
    Extract only text from PDF (no graph construction).

    Args:
        file: PDF file upload

    Returns:
        Dict: Extracted text and page information
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF document"
        )

    temp_file_path = None

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        # Extract text
        full_text = extract_text_from_pdf(temp_file_path)
        pages = extract_pages_with_metadata(temp_file_path)

        return {
            "filename": file.filename,
            "num_pages": len(pages),
            "total_characters": len(full_text),
            "full_text": full_text,
            "pages": pages
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction error: {str(e)}"
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@app.get("/config")
async def get_config():
    """
    Get current configuration (without sensitive data).

    Returns:
        Dict: Current configuration settings
    """
    return {
        "extraction_engine": settings.extraction_engine,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "dedup_threshold": settings.dedup_threshold,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "openai_configured": bool(settings.openai_api_key),
        "hf_configured": bool(settings.hf_token),
    }


@app.get("/supported-engines")
async def get_supported_engines():
    """
    Get list of supported extraction engines and their availability.

    Returns:
        Dict: Available extraction engines
    """
    return {
        "engines": [
            {
                "name": "openai",
                "display_name": "OpenAI GPT",
                "available": bool(settings.openai_api_key),
                "description": "GPT-based extraction with specific entity types"
            },
            {
                "name": "rebel",
                "display_name": "Hugging Face REBEL",
                "available": True,  # Always available (local model)
                "description": "Local REBEL model (no API key required)"
            }
        ]
    }


# Development server
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("Starting PDF to Knowledge Graph API")
    print("="*60)
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print("="*60 + "\n")

    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

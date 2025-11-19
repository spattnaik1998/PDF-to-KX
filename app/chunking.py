"""
Text chunking utilities.

This module handles splitting text into smaller chunks for processing
by language models with context length limitations.

The chunking strategies implemented here are designed to:
1. Keep text segments within LLM token/character limits
2. Preserve context through overlapping windows
3. Maintain page-level metadata for traceability
4. Optimize for entity/relation extraction downstream

Example usage:
    >>> from app.chunking import simple_chunk, chunk_pages
    >>> from app.pdf_utils import extract_pages_with_metadata
    >>>
    >>> # Simple text chunking
    >>> text = "Long document text..."
    >>> chunks = simple_chunk(text, max_chars=1000, overlap=100)
    >>> print(f"Created {len(chunks)} chunks")
    >>>
    >>> # Page-aware chunking
    >>> pages = extract_pages_with_metadata("document.pdf")
    >>> chunks = chunk_pages(pages, max_chars=1000, overlap=100)
    >>> for chunk in chunks:
    ...     print(f"Chunk {chunk['chunk_id']} from page {chunk['page_number']}")
"""

from typing import List, Dict
import re
from app.config import settings


def simple_chunk(
    text: str,
    max_chars: int = 3000,
    overlap: int = 300
) -> List[Dict[str, any]]:
    """
    Split text into overlapping chunks with metadata.

    This function creates chunks using a sliding window approach. Each chunk
    overlaps with the previous one to preserve context for entity extraction.

    The overlap ensures that entities and relationships spanning chunk boundaries
    are more likely to be captured in at least one chunk.

    Args:
        text (str): The text to chunk
        max_chars (int): Maximum characters per chunk. Default: 3000
        overlap (int): Number of overlapping characters between chunks. Default: 300

    Returns:
        List[Dict]: List of chunk dictionaries with keys:
            - chunk_id (int): 0-indexed chunk identifier
            - start_char (int): Starting character index in original text
            - end_char (int): Ending character index in original text
            - text (str): The chunk text content (trimmed)

    Raises:
        ValueError: If max_chars <= overlap or if overlap < 0

    Example:
        >>> text = "This is a long document. " * 200
        >>> chunks = simple_chunk(text, max_chars=1000, overlap=100)
        >>> print(f"Created {len(chunks)} chunks")
        >>> print(f"First chunk: {chunks[0]['text'][:50]}...")
        >>> print(f"Overlap check: {chunks[0]['text'][-50:] == chunks[1]['text'][:50]}")

    Downstream usage:
        These chunks will be fed to extraction_openai.py or extraction_hf.py
        for entity and relation extraction. The chunk_id helps track which
        chunk produced which triples.
    """
    # Validate parameters
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")
    if not text or not text.strip():
        return []

    # Trim input text
    text = text.strip()

    chunks = []
    chunk_id = 0
    start = 0

    while start < len(text):
        # Calculate end position for this chunk
        end = start + max_chars

        # Extract chunk text
        chunk_text = text[start:end].strip()

        # Only add non-empty chunks
        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": min(end, len(text)),
                "text": chunk_text
            })
            chunk_id += 1

        # Move to next chunk position (with overlap)
        # Next chunk starts at (current_end - overlap)
        start = end - overlap

        # Break if we've covered the text
        if end >= len(text):
            break

    return chunks


def chunk_pages(
    pages: List[Dict[str, any]],
    max_chars: int = 3000,
    overlap: int = 300
) -> List[Dict[str, any]]:
    """
    Chunk text from multiple pages while preserving page metadata.

    This function processes pages from extract_pages_with_metadata(),
    creating chunks that include page number information. This is useful
    for tracing extracted entities back to their source pages.

    Each page is chunked independently, but overlap can span page boundaries
    if needed (though currently pages are processed separately).

    Args:
        pages (List[Dict]): List of page dictionaries from extract_pages_with_metadata()
            Each page should have:
                - page_number (int): Page number
                - text (str): Page text content
        max_chars (int): Maximum characters per chunk. Default: 3000
        overlap (int): Overlap between chunks. Default: 300

    Returns:
        List[Dict]: List of chunk dictionaries with keys:
            - chunk_id (int): Global chunk identifier across all pages
            - page_number (int): Source page number
            - text (str): The chunk text content (trimmed)

    Example:
        >>> from app.pdf_utils import extract_pages_with_metadata
        >>> pages = extract_pages_with_metadata("document.pdf")
        >>> chunks = chunk_pages(pages, max_chars=1000, overlap=100)
        >>> for chunk in chunks[:3]:
        ...     print(f"Chunk {chunk['chunk_id']} from page {chunk['page_number']}")
        Chunk 0 from page 1
        Chunk 1 from page 1
        Chunk 2 from page 2

    Downstream usage:
        The page_number field allows us to:
        1. Track which page entities were extracted from
        2. Display source pages in the UI
        3. Build page-level knowledge graphs
        4. Validate extraction quality per page
    """
    # Validate input
    if not pages:
        return []

    all_chunks = []
    global_chunk_id = 0

    for page in pages:
        # Extract page data
        page_number = page.get("page_number", 0)
        page_text = page.get("text", "").strip()

        # Skip empty pages
        if not page_text:
            continue

        # Chunk this page's text
        page_chunks = simple_chunk(page_text, max_chars=max_chars, overlap=overlap)

        # Add page metadata to each chunk
        for chunk in page_chunks:
            all_chunks.append({
                "chunk_id": global_chunk_id,
                "page_number": page_number,
                "text": chunk["text"]
            })
            global_chunk_id += 1

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[str]:
    """
    Split text into overlapping chunks (simple string list output).

    This is a simplified version of simple_chunk() that returns just the
    text strings without metadata. Useful for basic chunking needs.

    Args:
        text (str): The text to chunk
        chunk_size (int, optional): Maximum characters per chunk.
            Defaults to settings.chunk_size
        chunk_overlap (int, optional): Number of overlapping characters.
            Defaults to settings.chunk_overlap

    Returns:
        List[str]: List of text chunks

    Example:
        >>> text = "Long document text..."
        >>> chunks = chunk_text(text)
        >>> print(f"Created {len(chunks)} chunks")
        >>> print(chunks[0][:100])
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    # Use simple_chunk and extract just the text
    chunk_dicts = simple_chunk(text, max_chars=chunk_size, overlap=chunk_overlap)
    return [chunk["text"] for chunk in chunk_dicts]


def chunk_text_by_sentences(
    text: str,
    max_sentences_per_chunk: int = 5
) -> List[str]:
    """
    Split text into chunks by sentence boundaries.

    This method tries to break on sentence boundaries for more semantic chunks.
    Useful when you want to preserve complete sentences in each chunk.

    Args:
        text (str): The text to chunk
        max_sentences_per_chunk (int): Maximum sentences per chunk

    Returns:
        List[str]: List of text chunks

    Example:
        >>> text = "First sentence. Second sentence. Third sentence. Fourth."
        >>> chunks = chunk_text_by_sentences(text, max_sentences_per_chunk=2)
        >>> print(len(chunks))  # Should be 2 chunks
    """
    if not text or not text.strip():
        return []

    # Simple sentence splitting using regex
    # Matches periods, question marks, exclamation points followed by space/newline
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())

    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Group sentences into chunks
    chunks = []
    for i in range(0, len(sentences), max_sentences_per_chunk):
        chunk_sentences = sentences[i:i + max_sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_text_smart(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    respect_sentences: bool = True
) -> List[str]:
    """
    Smart chunking that respects both size limits and sentence boundaries.

    This function tries to break at sentence boundaries when possible,
    but falls back to character-based chunking if sentences are too long.

    Args:
        text (str): The text to chunk
        chunk_size (int, optional): Target characters per chunk
        chunk_overlap (int, optional): Overlap between chunks
        respect_sentences (bool): Try to break on sentence boundaries

    Returns:
        List[str]: List of text chunks

    Example:
        >>> text = "First sentence. " * 100 + "Last sentence."
        >>> chunks = chunk_text_smart(text, chunk_size=500, overlap=50)
        >>> # Chunks will try to end on sentence boundaries
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    if not respect_sentences:
        # Fall back to simple chunking
        return chunk_text(text, chunk_size, chunk_overlap)

    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        # Calculate initial end position
        end = start + chunk_size

        if end >= len(text):
            # Last chunk - take everything remaining
            chunk_text = text[start:].strip()
            if chunk_text:
                chunks.append(chunk_text)
            break

        # Try to find a sentence boundary near the end
        # Look for sentence endings in the last 20% of the chunk
        search_start = end - int(chunk_size * 0.2)
        search_region = text[search_start:end + 50]  # Look a bit ahead too

        # Find sentence endings (., !, ?)
        sentence_endings = []
        for match in re.finditer(r'[.!?]\s+', search_region):
            sentence_endings.append(search_start + match.end())

        if sentence_endings:
            # Use the last sentence boundary found
            end = sentence_endings[-1]

        # Extract chunk
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Move to next position with overlap
        start = end - chunk_overlap

    return chunks


def validate_chunks(chunks: List[Dict[str, any]]) -> bool:
    """
    Validate that chunks are properly formed.

    Checks:
    1. All chunks have required fields
    2. Text is non-empty and trimmed
    3. Chunk IDs are sequential

    Args:
        chunks (List[Dict]): List of chunk dictionaries

    Returns:
        bool: True if all chunks are valid

    Raises:
        ValueError: If validation fails with description of issue
    """
    if not chunks:
        return True

    required_fields = {"chunk_id", "text"}

    for i, chunk in enumerate(chunks):
        # Check required fields
        missing_fields = required_fields - set(chunk.keys())
        if missing_fields:
            raise ValueError(
                f"Chunk {i} missing required fields: {missing_fields}"
            )

        # Check text is non-empty
        if not chunk["text"] or not chunk["text"].strip():
            raise ValueError(f"Chunk {i} has empty text")

        # Check text is trimmed
        if chunk["text"] != chunk["text"].strip():
            raise ValueError(f"Chunk {i} text is not trimmed")

        # Check chunk_id is sequential
        if chunk["chunk_id"] != i:
            raise ValueError(
                f"Chunk {i} has non-sequential chunk_id: {chunk['chunk_id']}"
            )

    return True


def get_chunk_statistics(chunks: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Get statistics about a list of chunks.

    Args:
        chunks (List[Dict]): List of chunk dictionaries

    Returns:
        Dict: Statistics including:
            - total_chunks (int): Number of chunks
            - total_chars (int): Total characters across all chunks
            - avg_chunk_size (float): Average chunk size in characters
            - min_chunk_size (int): Smallest chunk size
            - max_chunk_size (int): Largest chunk size

    Example:
        >>> chunks = simple_chunk("text " * 1000, max_chars=500, overlap=50)
        >>> stats = get_chunk_statistics(chunks)
        >>> print(f"Created {stats['total_chunks']} chunks")
        >>> print(f"Average size: {stats['avg_chunk_size']:.1f} chars")
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_chars": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }

    chunk_sizes = [len(chunk["text"]) for chunk in chunks]
    total_chars = sum(chunk_sizes)

    return {
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_size": total_chars / len(chunks),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes)
    }

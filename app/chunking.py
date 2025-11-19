"""
Text chunking utilities.

This module handles splitting text into smaller chunks for processing
by language models with context length limitations.
"""

from typing import List
from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text (str): The text to chunk
        chunk_size (int, optional): Maximum characters per chunk.
            Defaults to settings.chunk_size
        chunk_overlap (int, optional): Number of overlapping characters.
            Defaults to settings.chunk_overlap

    Returns:
        List[str]: List of text chunks

    TODO: Implement text chunking logic
    - Use sliding window approach
    - Respect chunk_size and chunk_overlap parameters
    - Try to break on sentence/word boundaries when possible
    - Handle edge cases (text shorter than chunk_size)
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    raise NotImplementedError("Text chunking not yet implemented")


def chunk_text_by_sentences(
    text: str,
    max_sentences_per_chunk: int = 5
) -> List[str]:
    """
    Split text into chunks by sentence boundaries.

    Args:
        text (str): The text to chunk
        max_sentences_per_chunk (int): Maximum sentences per chunk

    Returns:
        List[str]: List of text chunks

    TODO: Implement sentence-based chunking
    - Split text into sentences (using simple rules or nltk)
    - Group sentences into chunks
    - More semantic than character-based chunking
    """
    raise NotImplementedError("Sentence-based chunking not yet implemented")


def chunk_text_smart(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    respect_sentences: bool = True
) -> List[str]:
    """
    Smart chunking that respects both size limits and sentence boundaries.

    Args:
        text (str): The text to chunk
        chunk_size (int, optional): Target characters per chunk
        chunk_overlap (int, optional): Overlap between chunks
        respect_sentences (bool): Try to break on sentence boundaries

    Returns:
        List[str]: List of text chunks

    TODO: Implement smart chunking
    - Combine benefits of character-based and sentence-based chunking
    - Prefer breaking at sentence boundaries
    - Fall back to character-based if sentences too long
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    raise NotImplementedError("Smart chunking not yet implemented")

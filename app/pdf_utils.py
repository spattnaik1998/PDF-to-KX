"""
PDF text extraction utilities.

This module handles extracting text content from PDF files using PyMuPDF (fitz).

Example usage:
    >>> from app.pdf_utils import extract_text_from_pdf, extract_pages_with_metadata
    >>>
    >>> # Extract all text from a PDF
    >>> text = extract_text_from_pdf("document.pdf")
    >>> print(f"Extracted {len(text)} characters")
    >>>
    >>> # Extract text with page metadata
    >>> pages = extract_pages_with_metadata("document.pdf")
    >>> for page in pages:
    ...     print(f"Page {page['page_number']}: {len(page['text'])} chars")
"""

import fitz  # PyMuPDF
from typing import List, Dict
from pathlib import Path


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.

    This function opens a PDF file, iterates through all pages, extracts text
    from each page, and returns the concatenated result.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content from all pages, joined with double newlines

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        PDFExtractionError: If PDF cannot be opened or processed
        ValueError: If pdf_path is empty or None

    Example:
        >>> text = extract_text_from_pdf("research_paper.pdf")
        >>> print(f"Document has {len(text)} characters")
        >>> print(text[:100])  # First 100 characters
    """
    # Validate input
    if not pdf_path:
        raise ValueError("pdf_path cannot be empty or None")

    # Check if file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Validate it's a PDF file
    if not validate_pdf(pdf_path):
        raise PDFExtractionError(f"File is not a valid PDF: {pdf_path}")

    try:
        # Open the PDF
        doc = fitz.open(pdf_path)

        # Extract text from all pages
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            text_parts.append(text)

        # Close the document
        doc.close()

        # Join all text with double newline for page separation
        full_text = "\n\n".join(text_parts)

        return full_text

    except fitz.FileDataError as e:
        raise PDFExtractionError(f"Corrupted or invalid PDF file: {e}")
    except Exception as e:
        raise PDFExtractionError(f"Error extracting text from PDF: {e}")


def extract_text_by_page(pdf_path: str) -> List[str]:
    """
    Extract text from PDF, returning a list of text per page.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[str]: List where each element is text from one page

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        PDFExtractionError: If PDF cannot be opened or processed

    Example:
        >>> pages = extract_text_by_page("document.pdf")
        >>> print(f"Document has {len(pages)} pages")
        >>> print(f"First page: {pages[0][:100]}...")
    """
    if not pdf_path:
        raise ValueError("pdf_path cannot be empty or None")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not validate_pdf(pdf_path):
        raise PDFExtractionError(f"File is not a valid PDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)

        # Extract text from each page
        page_texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            page_texts.append(text)

        doc.close()

        return page_texts

    except fitz.FileDataError as e:
        raise PDFExtractionError(f"Corrupted or invalid PDF file: {e}")
    except Exception as e:
        raise PDFExtractionError(f"Error extracting text from PDF: {e}")


def extract_pages_with_metadata(pdf_path: str) -> List[Dict[str, any]]:
    """
    Extract text from PDF with page metadata.

    This function provides structured output with page numbers and text content,
    useful for tracking which page content came from.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[Dict]: List of dictionaries with keys:
            - page_number (int): 1-indexed page number
            - text (str): Extracted text from the page

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        PDFExtractionError: If PDF cannot be opened or processed

    Example:
        >>> pages = extract_pages_with_metadata("document.pdf")
        >>> for page in pages:
        ...     print(f"Page {page['page_number']}: {len(page['text'])} chars")
        Page 1: 1523 chars
        Page 2: 2041 chars
        Page 3: 1876 chars
    """
    if not pdf_path:
        raise ValueError("pdf_path cannot be empty or None")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not validate_pdf(pdf_path):
        raise PDFExtractionError(f"File is not a valid PDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)

        # Extract text with metadata
        pages_data = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            pages_data.append({
                "page_number": page_num + 1,  # 1-indexed for human readability
                "text": text
            })

        doc.close()

        return pages_data

    except fitz.FileDataError as e:
        raise PDFExtractionError(f"Corrupted or invalid PDF file: {e}")
    except Exception as e:
        raise PDFExtractionError(f"Error extracting text from PDF: {e}")


def validate_pdf(pdf_path: str) -> bool:
    """
    Validate that a file is a readable PDF.

    This function performs multiple checks to ensure the file is a valid PDF:
    1. File exists
    2. File has .pdf extension
    3. File can be opened by PyMuPDF

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        bool: True if valid PDF, False otherwise

    Example:
        >>> if validate_pdf("document.pdf"):
        ...     print("Valid PDF file")
        ... else:
        ...     print("Invalid or corrupted PDF")
    """
    try:
        # Check if path is provided
        if not pdf_path:
            return False

        # Check if file exists
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return False

        # Check file extension (case-insensitive)
        if pdf_file.suffix.lower() != '.pdf':
            return False

        # Try to open the PDF with fitz
        doc = fitz.open(pdf_path)

        # Check if document has at least one page
        page_count = len(doc)
        doc.close()

        return page_count > 0

    except Exception:
        # Any exception means the PDF is not valid or readable
        return False


def get_pdf_info(pdf_path: str) -> Dict[str, any]:
    """
    Get metadata information about a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        Dict: Dictionary containing PDF metadata:
            - page_count (int): Number of pages
            - title (str): Document title (if available)
            - author (str): Document author (if available)
            - subject (str): Document subject (if available)
            - file_size (int): File size in bytes

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        PDFExtractionError: If PDF cannot be opened

    Example:
        >>> info = get_pdf_info("document.pdf")
        >>> print(f"Pages: {info['page_count']}, Size: {info['file_size']} bytes")
    """
    if not pdf_path:
        raise ValueError("pdf_path cannot be empty or None")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)

        # Get metadata
        metadata = doc.metadata

        info = {
            "page_count": len(doc),
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "file_size": pdf_file.stat().st_size
        }

        doc.close()

        return info

    except fitz.FileDataError as e:
        raise PDFExtractionError(f"Corrupted or invalid PDF file: {e}")
    except Exception as e:
        raise PDFExtractionError(f"Error reading PDF info: {e}")

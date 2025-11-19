"""
Configuration module for PDF Knowledge Graph application.

Loads environment variables and provides application-wide constants.
"""

import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class ExtractionEngine(str, Enum):
    """Supported extraction engines for entity/relation extraction."""
    OPENAI = "openai"
    REBEL = "rebel"  # Hugging Face REBEL model


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")

    # Extraction Configuration
    extraction_engine: str = os.getenv("EXTRACTION_ENGINE", "openai")

    # Chunking Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Embedding Configuration
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Deduplication Configuration
    dedup_threshold: float = float(os.getenv("DEDUP_THRESHOLD", "0.85"))

    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Streamlit Configuration
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Validation helper
def validate_config() -> bool:
    """
    Validate that required configuration is present.

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If required configuration is missing
    """
    if settings.extraction_engine == ExtractionEngine.OPENAI:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when using 'openai' extraction engine"
            )

    return True


# Constants
DEFAULT_MODEL_OPENAI = "gpt-4-turbo-preview"
DEFAULT_MODEL_HF_REBEL = "Babelscape/rebel-large"

# Prompt templates (to be used in extraction modules)
EXTRACTION_PROMPT_TEMPLATE = """
Extract entities and relationships from the following text.
Return the result as a list of triples in the format: (subject, predicate, object)

Text:
{text}
"""

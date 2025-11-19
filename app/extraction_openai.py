"""
OpenAI-based entity and relation extraction.

This module uses OpenAI's GPT models to extract entities and relationships
from text chunks using structured JSON output.

The extraction pipeline:
1. Takes a text chunk
2. Sends it to GPT with a structured schema
3. Receives entities and relations in JSON format
4. Validates and returns the structured data

Example usage:
    >>> from app.extraction_openai import extract_entities_relations
    >>>
    >>> text = "John Smith works at Microsoft in Seattle."
    >>> result = extract_entities_relations(text)
    >>> print(f"Found {len(result['entities'])} entities")
    >>> print(f"Found {len(result['relations'])} relations")
"""

from typing import List, Dict, Optional
import json
import time
from pydantic import BaseModel, Field
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from app.config import settings, DEFAULT_MODEL_OPENAI


# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


# Pydantic models for structured output
class Entity(BaseModel):
    """Represents an extracted entity."""
    id: str = Field(description="Unique identifier for the entity (e.g., 'person_1', 'org_1')")
    label: str = Field(description="The entity name or label as it appears in text")
    type: str = Field(description="Entity type (e.g., 'Person', 'Organization', 'Location', 'Concept', 'Event')")


class Relation(BaseModel):
    """Represents a relationship between two entities."""
    source: str = Field(description="ID of the source entity")
    target: str = Field(description="ID of the target entity")
    type: str = Field(description="Relationship type (e.g., 'works_at', 'located_in', 'founded_by')")
    evidence_span: str = Field(description="Text span that evidences this relationship")


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph extraction result."""
    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")
    relations: List[Relation] = Field(default_factory=list, description="List of extracted relations")


# System prompt for extraction
EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting structured knowledge from text.
Your task is to identify entities (people, organizations, locations, concepts, events, etc.)
and the relationships between them.

Guidelines:
1. Extract all meaningful entities with clear, concise labels
2. Identify relationships that are explicitly stated or strongly implied
3. Use descriptive relationship types (e.g., 'works_at', 'founded', 'located_in')
4. Provide evidence spans that support each relationship
5. Assign unique IDs to entities (e.g., 'person_1', 'org_1', 'loc_1')
6. Use consistent entity types (Person, Organization, Location, Concept, Event, Product, etc.)

Be thorough but accurate - only extract information that is present in the text."""


def extract_entities_relations(
    text: str,
    model: str = None,
    temperature: float = 0.0
) -> Dict[str, List[Dict]]:
    """
    Extract entities and relations from text using OpenAI API.

    This function uses OpenAI's structured output feature to ensure
    the response matches our schema for entities and relations.

    Args:
        text (str): Text chunk to process
        model (str, optional): OpenAI model to use. Defaults to DEFAULT_MODEL_OPENAI
        temperature (float): Model temperature. Default 0.0 for deterministic output

    Returns:
        Dict with keys:
            - entities: List[Dict] - Each with 'id', 'label', 'type'
            - relations: List[Dict] - Each with 'source', 'target', 'type', 'evidence_span'

    Raises:
        ValueError: If text is empty or API key is not configured
        APIError: If OpenAI API request fails

    Example:
        >>> text = "Albert Einstein developed the theory of relativity at Princeton."
        >>> result = extract_entities_relations(text)
        >>> print(result['entities'])
        [{'id': 'person_1', 'label': 'Albert Einstein', 'type': 'Person'},
         {'id': 'concept_1', 'label': 'theory of relativity', 'type': 'Concept'},
         {'id': 'org_1', 'label': 'Princeton', 'type': 'Organization'}]
        >>> print(result['relations'])
        [{'source': 'person_1', 'target': 'concept_1', 'type': 'developed',
          'evidence_span': 'Albert Einstein developed the theory of relativity'}]
    """
    # Validate inputs
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured in settings")

    if model is None:
        model = DEFAULT_MODEL_OPENAI

    # Prepare user prompt
    user_prompt = f"""Extract all entities and relationships from the following text.

Text:
{text}

Return a complete knowledge graph with entities and their relationships."""

    try:
        # Call OpenAI API with structured output
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=KnowledgeGraph,
            temperature=temperature
        )

        # Extract the parsed response
        knowledge_graph = completion.choices[0].message.parsed

        # Convert Pydantic models to dictionaries
        result = {
            "entities": [entity.model_dump() for entity in knowledge_graph.entities],
            "relations": [relation.model_dump() for relation in knowledge_graph.relations]
        }

        return result

    except RateLimitError as e:
        raise APIError(f"OpenAI rate limit exceeded: {e}")
    except APIConnectionError as e:
        raise APIError(f"OpenAI connection error: {e}")
    except Exception as e:
        raise APIError(f"OpenAI API error: {e}")


def extract_entities_relations_with_retry(
    text: str,
    model: str = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, List[Dict]]:
    """
    Extract entities and relations with retry logic for handling rate limits.

    Args:
        text (str): Text chunk to process
        model (str, optional): OpenAI model to use
        max_retries (int): Maximum number of retry attempts. Default: 3
        retry_delay (float): Initial delay between retries in seconds. Default: 1.0

    Returns:
        Dict with 'entities' and 'relations' keys

    Raises:
        APIError: If all retry attempts fail

    Example:
        >>> result = extract_entities_relations_with_retry(text, max_retries=5)
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return extract_entities_relations(text, model=model)
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise APIError(f"Failed after {max_retries} attempts: {e}")
        except Exception as e:
            raise APIError(f"Extraction failed: {e}")

    raise APIError(f"Failed after {max_retries} attempts: {last_error}")


def extract_from_chunks(
    chunks: List[Dict[str, any]],
    model: str = None,
    show_progress: bool = False
) -> Dict[str, any]:
    """
    Extract entities and relations from multiple text chunks.

    This function processes each chunk independently and combines the results.
    Entity IDs are made globally unique by prefixing with chunk ID.

    Args:
        chunks (List[Dict]): List of chunk dictionaries with 'chunk_id' and 'text' keys
        model (str, optional): OpenAI model to use
        show_progress (bool): Whether to print progress updates

    Returns:
        Dict with keys:
            - entities: List[Dict] - All entities from all chunks
            - relations: List[Dict] - All relations from all chunks
            - chunk_count: int - Number of chunks processed
            - entity_count: int - Total entities extracted
            - relation_count: int - Total relations extracted

    Example:
        >>> from app.chunking import simple_chunk
        >>> chunks = simple_chunk(long_text, max_chars=1000)
        >>> result = extract_from_chunks(chunks)
        >>> print(f"Extracted {result['entity_count']} entities from {result['chunk_count']} chunks")
    """
    if model is None:
        model = DEFAULT_MODEL_OPENAI

    all_entities = []
    all_relations = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", i)
        text = chunk.get("text", "")

        if not text.strip():
            continue

        if show_progress:
            print(f"Processing chunk {i+1}/{len(chunks)} (ID: {chunk_id})...")

        try:
            # Extract from this chunk
            result = extract_entities_relations_with_retry(text, model=model)

            # Make entity IDs globally unique by prefixing with chunk ID
            for entity in result["entities"]:
                entity["id"] = f"chunk{chunk_id}_{entity['id']}"
                entity["chunk_id"] = chunk_id
                all_entities.append(entity)

            # Update relation IDs to match new entity IDs
            for relation in result["relations"]:
                relation["source"] = f"chunk{chunk_id}_{relation['source']}"
                relation["target"] = f"chunk{chunk_id}_{relation['target']}"
                relation["chunk_id"] = chunk_id
                all_relations.append(relation)

        except Exception as e:
            if show_progress:
                print(f"  Warning: Failed to process chunk {chunk_id}: {e}")
            continue

    return {
        "entities": all_entities,
        "relations": all_relations,
        "chunk_count": len(chunks),
        "entity_count": len(all_entities),
        "relation_count": len(all_relations)
    }


def format_extraction_prompt(text: str) -> str:
    """
    Format the extraction prompt with the given text.

    This is a simple helper that formats text for extraction.
    The actual prompt is handled by extract_entities_relations().

    Args:
        text (str): Text to insert into prompt template

    Returns:
        str: Formatted prompt

    Example:
        >>> prompt = format_extraction_prompt("Sample text here")
        >>> print(prompt)
    """
    return f"""Extract all entities and relationships from the following text.

Text:
{text}

Return a complete knowledge graph with entities and their relationships."""


def validate_extraction_result(result: Dict[str, List[Dict]]) -> bool:
    """
    Validate that an extraction result has the expected structure.

    Args:
        result (Dict): Extraction result to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If validation fails

    Example:
        >>> result = extract_entities_relations(text)
        >>> validate_extraction_result(result)
        True
    """
    # Check top-level keys
    if "entities" not in result or "relations" not in result:
        raise ValueError("Result must have 'entities' and 'relations' keys")

    # Validate entities
    for i, entity in enumerate(result["entities"]):
        required_fields = {"id", "label", "type"}
        missing = required_fields - set(entity.keys())
        if missing:
            raise ValueError(f"Entity {i} missing fields: {missing}")

    # Validate relations
    for i, relation in enumerate(result["relations"]):
        required_fields = {"source", "target", "type", "evidence_span"}
        missing = required_fields - set(relation.keys())
        if missing:
            raise ValueError(f"Relation {i} missing fields: {missing}")

    # Check that relation references exist in entities
    entity_ids = {e["id"] for e in result["entities"]}
    for i, relation in enumerate(result["relations"]):
        if relation["source"] not in entity_ids:
            raise ValueError(f"Relation {i} references unknown source: {relation['source']}")
        if relation["target"] not in entity_ids:
            raise ValueError(f"Relation {i} references unknown target: {relation['target']}")

    return True


def get_entity_types_summary(entities: List[Dict]) -> Dict[str, int]:
    """
    Get a summary of entity types and their counts.

    Args:
        entities (List[Dict]): List of entity dictionaries

    Returns:
        Dict[str, int]: Mapping of entity type to count

    Example:
        >>> entities = [
        ...     {"id": "1", "label": "John", "type": "Person"},
        ...     {"id": "2", "label": "Microsoft", "type": "Organization"},
        ...     {"id": "3", "label": "Jane", "type": "Person"}
        ... ]
        >>> summary = get_entity_types_summary(entities)
        >>> print(summary)
        {'Person': 2, 'Organization': 1}
    """
    type_counts = {}
    for entity in entities:
        entity_type = entity.get("type", "Unknown")
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    return type_counts


def get_relation_types_summary(relations: List[Dict]) -> Dict[str, int]:
    """
    Get a summary of relation types and their counts.

    Args:
        relations (List[Dict]): List of relation dictionaries

    Returns:
        Dict[str, int]: Mapping of relation type to count

    Example:
        >>> relations = [
        ...     {"source": "1", "target": "2", "type": "works_at", "evidence_span": "..."},
        ...     {"source": "1", "target": "3", "type": "knows", "evidence_span": "..."}
        ... ]
        >>> summary = get_relation_types_summary(relations)
        >>> print(summary)
        {'works_at': 1, 'knows': 1}
    """
    type_counts = {}
    for relation in relations:
        relation_type = relation.get("type", "Unknown")
        type_counts[relation_type] = type_counts.get(relation_type, 0) + 1
    return type_counts

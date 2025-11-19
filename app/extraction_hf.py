"""
Hugging Face model-based entity and relation extraction.

This module uses the REBEL (Relation Extraction By End-to-end Language generation)
model from Hugging Face for extracting entities and relationships.

REBEL is a seq2seq model that generates linearized triplets in the format:
    <triplet> subject <subj> relation <obj> object </triplet>

Example usage:
    >>> from app.extraction_hf import extract_triples_rebel
    >>>
    >>> text = "Marie Curie won the Nobel Prize in Physics."
    >>> result = extract_triples_rebel(text, chunk_id=0)
    >>> print(f"Found {len(result['entities'])} entities")
    >>> print(f"Found {len(result['relations'])} relations")
"""

from typing import List, Dict, Tuple, Optional
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import settings, DEFAULT_MODEL_HF_REBEL


# Global model cache for singleton pattern
_rebel_pipeline = None


def get_rebel_pipeline(model_name: str = None):
    """
    Get or create the REBEL pipeline (singleton pattern).

    This function ensures the model is loaded only once and reused
    across multiple calls, saving memory and initialization time.

    Args:
        model_name (str, optional): HuggingFace model identifier.
            Defaults to DEFAULT_MODEL_HF_REBEL

    Returns:
        transformers.Pipeline: REBEL text2text-generation pipeline

    Example:
        >>> pipeline = get_rebel_pipeline()
        >>> result = pipeline("Einstein developed relativity.")
    """
    global _rebel_pipeline

    if _rebel_pipeline is None:
        if model_name is None:
            model_name = DEFAULT_MODEL_HF_REBEL

        print(f"Loading REBEL model: {model_name}...")
        print("This may take a few minutes on first run...")

        # Create pipeline for text2text generation
        _rebel_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=-1  # Use CPU (-1), or 0 for GPU
        )

        print("REBEL model loaded successfully!")

    return _rebel_pipeline


def _parse_rebel_output(generated_text: str) -> Dict[str, List[Dict]]:
    """
    Parse REBEL's linearized triplet format into structured entities and relations.

    REBEL outputs text in this format:
        <triplet> Marie Curie <subj> won <obj> Nobel Prize in Physics </triplet>
        <triplet> Marie Curie <subj> birth place <obj> Warsaw </triplet>

    This function extracts:
    1. All unique entities (subjects and objects)
    2. All relations between entities

    Args:
        generated_text (str): Raw output from REBEL model

    Returns:
        Dict with keys:
            - entities: List[Dict] with 'id', 'label', 'type'
            - relations: List[Dict] with 'source', 'target', 'type', 'evidence_span'

    Example:
        >>> text = "<triplet> Einstein <subj> developed <obj> relativity </triplet>"
        >>> result = _parse_rebel_output(text)
        >>> print(result['entities'])
        [{'id': 'e0', 'label': 'Einstein', 'type': 'Concept'},
         {'id': 'e1', 'label': 'relativity', 'type': 'Concept'}]
    """
    entities = []
    relations = []
    entity_map = {}  # Maps entity label -> entity id
    entity_counter = 0

    # Extract all triplet blocks using regex
    # Pattern: <triplet> ... </triplet>
    triplet_pattern = r'<triplet>(.*?)</triplet>'
    triplets = re.findall(triplet_pattern, generated_text, re.DOTALL)

    for triplet_text in triplets:
        # Parse the triplet structure: subject <subj> relation <obj> object
        # Split by special tokens
        parts = re.split(r'<subj>|<obj>', triplet_text)

        if len(parts) != 3:
            # Malformed triplet, skip it
            continue

        # Extract and clean components
        subject = parts[0].strip()
        relation = parts[1].strip()
        obj = parts[2].strip()

        if not subject or not relation or not obj:
            # Empty component, skip
            continue

        # Add subject entity if not seen before
        if subject not in entity_map:
            entity_id = f"e{entity_counter}"
            entity_map[subject] = entity_id
            entities.append({
                "id": entity_id,
                "label": subject,
                "type": "Concept"  # REBEL doesn't provide entity types, use generic
            })
            entity_counter += 1

        # Add object entity if not seen before
        if obj not in entity_map:
            entity_id = f"e{entity_counter}"
            entity_map[obj] = entity_id
            entities.append({
                "id": entity_id,
                "label": obj,
                "type": "Concept"
            })
            entity_counter += 1

        # Create relation
        relations.append({
            "source": entity_map[subject],
            "target": entity_map[obj],
            "type": relation,
            "evidence_span": f"{subject} {relation} {obj}"
        })

    return {
        "entities": entities,
        "relations": relations
    }


def extract_triples_rebel(
    chunk_text: str,
    chunk_id: Optional[int] = None,
    max_length: int = 256,
    num_beams: int = 3
) -> Dict[str, List[Dict]]:
    """
    Run the REBEL model on a chunk and return entities + relations.

    This function's output format matches the OpenAI extractor for compatibility
    with downstream processing (deduplication, graph building).

    Args:
        chunk_text (str): Text to extract from
        chunk_id (int, optional): Chunk identifier to attach to results
        max_length (int): Maximum length for generated output. Default: 256
        num_beams (int): Number of beams for beam search. Default: 3

    Returns:
        Dict with keys:
            - entities: List[Dict] - Each with 'id', 'label', 'type', 'chunk_id'
            - relations: List[Dict] - Each with 'source', 'target', 'type',
                                      'evidence_span', 'chunk_id'

    Raises:
        ValueError: If chunk_text is empty

    Example:
        >>> text = "Albert Einstein developed the theory of relativity."
        >>> result = extract_triples_rebel(text, chunk_id=0)
        >>> print(result['entities'])
        [{'id': 'e0', 'label': 'Albert Einstein', 'type': 'Concept', 'chunk_id': 0},
         {'id': 'e1', 'label': 'theory of relativity', 'type': 'Concept', 'chunk_id': 0}]
        >>> print(result['relations'])
        [{'source': 'e0', 'target': 'e1', 'type': 'developed',
          'evidence_span': 'Albert Einstein developed theory of relativity', 'chunk_id': 0}]
    """
    # Validate input
    if not chunk_text or not chunk_text.strip():
        raise ValueError("chunk_text cannot be empty")

    # Get the pipeline (lazy-loaded)
    rebel_pipeline = get_rebel_pipeline()

    # Preprocess text (truncate if needed)
    processed_text = preprocess_text_for_rebel(chunk_text, max_length=512)

    # Run REBEL model
    try:
        outputs = rebel_pipeline(
            processed_text,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1
        )

        # Extract generated text
        if outputs and len(outputs) > 0:
            generated_text = outputs[0]['generated_text']
        else:
            generated_text = ""

    except Exception as e:
        print(f"Warning: REBEL generation failed: {e}")
        generated_text = ""

    # Parse the output
    result = _parse_rebel_output(generated_text)

    # Add chunk_id to all entities and relations if provided
    if chunk_id is not None:
        # Prefix entity IDs with chunk ID for global uniqueness
        entity_id_map = {}  # Old ID -> New ID
        for entity in result["entities"]:
            old_id = entity["id"]
            new_id = f"chunk{chunk_id}_{old_id}"
            entity_id_map[old_id] = new_id
            entity["id"] = new_id
            entity["chunk_id"] = chunk_id

        # Update relation IDs to match new entity IDs
        for relation in result["relations"]:
            relation["source"] = entity_id_map.get(relation["source"], relation["source"])
            relation["target"] = entity_id_map.get(relation["target"], relation["target"])
            relation["chunk_id"] = chunk_id

    return result


def extract_from_chunks(
    chunks: List[Dict[str, any]],
    show_progress: bool = False
) -> Dict[str, any]:
    """
    Extract entities and relations from multiple text chunks using REBEL.

    This function matches the interface of extraction_openai.extract_from_chunks()
    for interoperability.

    Args:
        chunks (List[Dict]): List of chunk dictionaries with 'chunk_id' and 'text' keys
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
        >>> result = extract_from_chunks(chunks, show_progress=True)
        >>> print(f"Extracted {result['entity_count']} entities")
    """
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
            result = extract_triples_rebel(text, chunk_id=chunk_id)

            # Add to results
            all_entities.extend(result["entities"])
            all_relations.extend(result["relations"])

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


def preprocess_text_for_rebel(text: str, max_length: int = 512) -> str:
    """
    Preprocess text for REBEL model input.

    REBEL has a token limit, so we truncate text if needed.
    We also clean up excessive whitespace.

    Args:
        text (str): Raw text
        max_length (int): Maximum character length (approximate). Default: 512

    Returns:
        str: Preprocessed text

    Example:
        >>> long_text = "word " * 1000
        >>> processed = preprocess_text_for_rebel(long_text, max_length=100)
        >>> len(processed) <= 100
        True
    """
    # Clean up whitespace
    text = " ".join(text.split())

    # Truncate if too long (rough approximation, actual tokenization may differ)
    if len(text) > max_length:
        text = text[:max_length]

        # Try to end on a sentence boundary
        last_period = text.rfind('.')
        if last_period > max_length * 0.7:  # If we have at least 70% of text
            text = text[:last_period + 1]

    return text.strip()


def validate_rebel_result(result: Dict[str, List[Dict]]) -> bool:
    """
    Validate that a REBEL extraction result has the expected structure.

    Args:
        result (Dict): Extraction result to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If validation fails

    Example:
        >>> result = extract_triples_rebel(text)
        >>> validate_rebel_result(result)
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

    return True


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("REBEL Knowledge Extraction Example")
    print("=" * 60)

    # Example 1: Simple extraction
    print("\nExample 1: Simple text extraction")
    print("-" * 60)

    text1 = """
    Marie Curie was a physicist and chemist who conducted pioneering research
    on radioactivity. She was the first woman to win a Nobel Prize.
    """

    result1 = extract_triples_rebel(text1.strip(), chunk_id=0)

    print(f"Input text: {text1.strip()[:100]}...")
    print(f"\nExtracted {len(result1['entities'])} entities:")
    for entity in result1['entities']:
        print(f"  - {entity['label']} ({entity['type']}) [ID: {entity['id']}]")

    print(f"\nExtracted {len(result1['relations'])} relations:")
    for relation in result1['relations']:
        source_label = next(e['label'] for e in result1['entities'] if e['id'] == relation['source'])
        target_label = next(e['label'] for e in result1['entities'] if e['id'] == relation['target'])
        print(f"  - {source_label} --[{relation['type']}]--> {target_label}")
        print(f"    Evidence: {relation['evidence_span']}")

    # Example 2: Batch processing
    print("\n" + "=" * 60)
    print("Example 2: Batch chunk processing")
    print("-" * 60)

    chunks = [
        {"chunk_id": 0, "text": "Albert Einstein developed the theory of relativity."},
        {"chunk_id": 1, "text": "He won the Nobel Prize in Physics in 1921."},
        {"chunk_id": 2, "text": "Einstein was born in Germany and later moved to the United States."}
    ]

    result2 = extract_from_chunks(chunks, show_progress=True)

    print(f"\nTotal entities: {result2['entity_count']}")
    print(f"Total relations: {result2['relation_count']}")
    print(f"Chunks processed: {result2['chunk_count']}")

    # Example 3: Compare format with expected schema
    print("\n" + "=" * 60)
    print("Example 3: Validate output format")
    print("-" * 60)

    try:
        validate_rebel_result(result1)
        print("✓ Result structure is valid")
        print("\nSample entity:")
        if result1['entities']:
            print(f"  {result1['entities'][0]}")
        print("\nSample relation:")
        if result1['relations']:
            print(f"  {result1['relations'][0]}")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")

    print("\n" + "=" * 60)
    print("Note: REBEL assigns all entities type='Concept' by default.")
    print("For more specific entity types, use the OpenAI extractor.")
    print("=" * 60)

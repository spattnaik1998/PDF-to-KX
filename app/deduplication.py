"""
Entity deduplication using OpenAI embeddings and clustering.

This module handles deduplicating similar entities by:
1. Computing OpenAI embeddings for entity labels
2. Clustering similar entities using hierarchical clustering
3. Merging duplicate entities into canonical forms
4. Updating relations to use canonical entity IDs

The deduplication process is crucial for creating clean knowledge graphs
where "Einstein", "Albert Einstein", and "A. Einstein" are recognized
as the same entity.

Example usage:
    >>> from app.deduplication import deduplicate_entities
    >>>
    >>> entities = [
    ...     {"id": "e1", "label": "Einstein", "type": "Person"},
    ...     {"id": "e2", "label": "Albert Einstein", "type": "Person"},
    ...     {"id": "e3", "label": "Microsoft", "type": "Organization"}
    ... ]
    >>> relations = [
    ...     {"source": "e1", "target": "e3", "type": "works_at"},
    ...     {"source": "e2", "target": "e3", "type": "employed_by"}
    ... ]
    >>>
    >>> result = deduplicate_entities(entities, relations, threshold=0.85)
    >>> print(f"Reduced from {len(entities)} to {len(result['entities'])} entities")
"""

from typing import List, Dict, Set, Tuple, Optional
from openai import OpenAI
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from app.config import settings


# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


def embed_labels(
    labels: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 100
) -> List[List[float]]:
    """
    Use OpenAI text-embedding-3-large to embed a list of labels.

    This function sends entity labels to OpenAI's embedding API and
    returns vector representations. These embeddings capture semantic
    similarity between labels.

    Args:
        labels (List[str]): List of entity labels to embed
        model (str): OpenAI embedding model. Default: "text-embedding-3-large"
        batch_size (int): Number of texts to embed per API call. Default: 100

    Returns:
        List[List[float]]: List of embedding vectors (each ~3072 dimensions for large model)

    Raises:
        ValueError: If labels is empty or contains only empty strings
        Exception: If OpenAI API call fails

    Example:
        >>> labels = ["Einstein", "Albert Einstein", "Marie Curie"]
        >>> embeddings = embed_labels(labels)
        >>> len(embeddings)
        3
        >>> len(embeddings[0])  # Embedding dimension
        3072
    """
    if not labels:
        raise ValueError("labels cannot be empty")

    # Filter out empty labels
    non_empty_labels = [label.strip() for label in labels if label.strip()]
    if not non_empty_labels:
        raise ValueError("labels contains only empty strings")

    all_embeddings = []

    # Process in batches to avoid API limits
    for i in range(0, len(non_empty_labels), batch_size):
        batch = non_empty_labels[i:i + batch_size]

        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )

            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        except Exception as e:
            raise Exception(f"OpenAI embedding API error: {e}")

    return all_embeddings


def cluster_similar_entities(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    linkage: str = "average"
) -> List[int]:
    """
    Cluster entity embeddings using hierarchical clustering.

    Uses AgglomerativeClustering to group similar entities based on
    cosine similarity of their embeddings.

    Args:
        embeddings (np.ndarray): Entity embeddings, shape (n_entities, embedding_dim)
        threshold (float): Similarity threshold (0-1). Entities with similarity
                          >= threshold are clustered together. Default: 0.85
        linkage (str): Linkage criterion. Options: 'average', 'complete', 'single'.
                      Default: 'average'

    Returns:
        List[int]: Cluster labels for each entity (same index as input embeddings)

    Example:
        >>> embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1]])
        >>> labels = cluster_similar_entities(embeddings, threshold=0.8)
        >>> labels
        [0, 0, 1]  # First two embeddings are similar, third is different
    """
    if len(embeddings) == 0:
        return []

    if len(embeddings) == 1:
        return [0]

    # Convert similarity threshold to distance threshold
    # Cosine similarity ranges from -1 to 1
    # We use (1 - cosine_similarity) as distance
    # So threshold 0.85 similarity -> 0.15 distance
    distance_threshold = 1.0 - threshold

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage=linkage
    )

    cluster_labels = clustering.fit_predict(embeddings)

    return cluster_labels.tolist()


def find_canonical_entity(entity_group: List[Dict]) -> Dict:
    """
    Choose the canonical (representative) entity from a group of similar entities.

    Selection criteria (in order of priority):
    1. Longest label (more complete information)
    2. First alphabetically (for consistency)

    Args:
        entity_group (List[Dict]): List of similar entity dictionaries

    Returns:
        Dict: The chosen canonical entity

    Example:
        >>> entities = [
        ...     {"id": "e1", "label": "Einstein", "type": "Person"},
        ...     {"id": "e2", "label": "Albert Einstein", "type": "Person"}
        ... ]
        >>> canonical = find_canonical_entity(entities)
        >>> canonical["label"]
        'Albert Einstein'
    """
    if not entity_group:
        raise ValueError("entity_group cannot be empty")

    # Sort by length (descending), then alphabetically
    sorted_entities = sorted(
        entity_group,
        key=lambda e: (-len(e["label"]), e["label"])
    )

    return sorted_entities[0]


def deduplicate_entities(
    entities: List[Dict],
    relations: List[Dict],
    threshold: float = None,
    show_progress: bool = False
) -> Dict[str, any]:
    """
    Deduplicate entities and update relations to use canonical entity IDs.

    This is the main function for entity deduplication. It:
    1. Embeds all entity labels using OpenAI
    2. Clusters similar entities
    3. Merges entities within each cluster
    4. Updates relations to reference canonical entities
    5. Removes duplicate relations

    Args:
        entities (List[Dict]): List of entity dicts with 'id', 'label', 'type'
        relations (List[Dict]): List of relation dicts with 'source', 'target', 'type'
        threshold (float, optional): Similarity threshold (0-1).
                                    Defaults to settings.dedup_threshold
        show_progress (bool): Print progress updates

    Returns:
        Dict with keys:
            - entities: List[Dict] - Deduplicated entities
            - relations: List[Dict] - Relations with updated entity IDs
            - entity_mapping: Dict[str, str] - Old entity ID -> Canonical entity ID
            - original_count: int - Original entity count
            - final_count: int - Final entity count after deduplication
            - clusters: int - Number of entity clusters found

    Example:
        >>> entities = [
        ...     {"id": "e1", "label": "Einstein", "type": "Person"},
        ...     {"id": "e2", "label": "Albert Einstein", "type": "Person"}
        ... ]
        >>> relations = [{"source": "e1", "target": "e3", "type": "works_at"}]
        >>> result = deduplicate_entities(entities, relations, threshold=0.85)
        >>> len(result['entities'])
        1  # Two entities merged into one
    """
    if threshold is None:
        threshold = settings.dedup_threshold

    if not entities:
        return {
            "entities": [],
            "relations": relations,
            "entity_mapping": {},
            "original_count": 0,
            "final_count": 0,
            "clusters": 0
        }

    if show_progress:
        print(f"Deduplicating {len(entities)} entities...")

    # Step 1: Embed all entity labels
    if show_progress:
        print("  Computing embeddings...")

    labels = [entity["label"] for entity in entities]
    embeddings = embed_labels(labels)
    embeddings_array = np.array(embeddings)

    # Step 2: Cluster similar entities
    if show_progress:
        print(f"  Clustering with threshold {threshold}...")

    cluster_labels = cluster_similar_entities(embeddings_array, threshold=threshold)

    # Step 3: Group entities by cluster
    clusters = defaultdict(list)
    for entity, cluster_id in zip(entities, cluster_labels):
        clusters[cluster_id].append(entity)

    if show_progress:
        print(f"  Found {len(clusters)} clusters")

    # Step 4: Choose canonical entity for each cluster
    canonical_entities = []
    entity_mapping = {}  # Old ID -> Canonical ID

    for cluster_id, entity_group in clusters.items():
        canonical = find_canonical_entity(entity_group)
        canonical_entities.append(canonical)

        # Map all entities in this cluster to the canonical entity
        for entity in entity_group:
            entity_mapping[entity["id"]] = canonical["id"]

    # Step 5: Update relations with canonical entity IDs
    if show_progress:
        print("  Updating relations...")

    updated_relations = []
    for relation in relations:
        updated_relation = relation.copy()
        updated_relation["source"] = entity_mapping.get(
            relation["source"],
            relation["source"]
        )
        updated_relation["target"] = entity_mapping.get(
            relation["target"],
            relation["target"]
        )
        updated_relations.append(updated_relation)

    # Step 6: Remove duplicate relations
    unique_relations = remove_duplicate_relations(updated_relations)

    if show_progress:
        print(f"  Reduced from {len(entities)} to {len(canonical_entities)} entities")
        print(f"  Reduced from {len(relations)} to {len(unique_relations)} relations")

    return {
        "entities": canonical_entities,
        "relations": unique_relations,
        "entity_mapping": entity_mapping,
        "original_count": len(entities),
        "final_count": len(canonical_entities),
        "clusters": len(clusters)
    }


def remove_duplicate_relations(relations: List[Dict]) -> List[Dict]:
    """
    Remove duplicate relations based on (source, target, type) tuple.

    Keeps the first occurrence of each unique relation.

    Args:
        relations (List[Dict]): List of relation dictionaries

    Returns:
        List[Dict]: Unique relations

    Example:
        >>> relations = [
        ...     {"source": "e1", "target": "e2", "type": "knows"},
        ...     {"source": "e1", "target": "e2", "type": "knows"},
        ...     {"source": "e1", "target": "e3", "type": "works_at"}
        ... ]
        >>> unique = remove_duplicate_relations(relations)
        >>> len(unique)
        2
    """
    seen = set()
    unique_relations = []

    for relation in relations:
        # Create a tuple of (source, target, type) as unique key
        key = (
            relation.get("source"),
            relation.get("target"),
            relation.get("type")
        )

        if key not in seen:
            seen.add(key)
            unique_relations.append(relation)

    return unique_relations


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for embeddings.

    Args:
        embeddings (np.ndarray): Embeddings array, shape (n, embedding_dim)

    Returns:
        np.ndarray: Similarity matrix, shape (n, n)

    Example:
        >>> embeddings = np.array([[1, 0], [0, 1], [1, 0]])
        >>> sim_matrix = compute_similarity_matrix(embeddings)
        >>> sim_matrix[0, 2]  # First and third are identical
        1.0
    """
    return cosine_similarity(embeddings)


def get_deduplication_statistics(
    original_entities: List[Dict],
    deduplicated_entities: List[Dict],
    entity_mapping: Dict[str, str]
) -> Dict[str, any]:
    """
    Get statistics about the deduplication process.

    Args:
        original_entities (List[Dict]): Original entity list
        deduplicated_entities (List[Dict]): Deduplicated entity list
        entity_mapping (Dict[str, str]): Entity ID mapping

    Returns:
        Dict with statistics:
            - original_count: int
            - final_count: int
            - reduction_count: int
            - reduction_percent: float
            - avg_cluster_size: float

    Example:
        >>> stats = get_deduplication_statistics(
        ...     original_entities, deduplicated_entities, entity_mapping
        ... )
        >>> print(f"Reduced by {stats['reduction_percent']:.1f}%")
    """
    original_count = len(original_entities)
    final_count = len(deduplicated_entities)
    reduction_count = original_count - final_count

    # Calculate average cluster size
    cluster_sizes = defaultdict(int)
    for old_id, canonical_id in entity_mapping.items():
        cluster_sizes[canonical_id] += 1

    avg_cluster_size = (
        sum(cluster_sizes.values()) / len(cluster_sizes)
        if cluster_sizes else 1.0
    )

    return {
        "original_count": original_count,
        "final_count": final_count,
        "reduction_count": reduction_count,
        "reduction_percent": (
            (reduction_count / original_count * 100)
            if original_count > 0 else 0
        ),
        "avg_cluster_size": avg_cluster_size
    }


def validate_deduplication_result(result: Dict) -> bool:
    """
    Validate that a deduplication result has the expected structure.

    Args:
        result (Dict): Deduplication result to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If validation fails

    Example:
        >>> result = deduplicate_entities(entities, relations)
        >>> validate_deduplication_result(result)
        True
    """
    required_keys = {
        "entities", "relations", "entity_mapping",
        "original_count", "final_count", "clusters"
    }

    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        raise ValueError(f"Result missing required keys: {missing_keys}")

    # Validate entities
    for entity in result["entities"]:
        if "id" not in entity or "label" not in entity:
            raise ValueError("Entity missing required fields")

    # Validate relations reference valid entities
    entity_ids = {e["id"] for e in result["entities"]}
    for relation in result["relations"]:
        if relation["source"] not in entity_ids:
            raise ValueError(f"Relation references unknown source: {relation['source']}")
        if relation["target"] not in entity_ids:
            raise ValueError(f"Relation references unknown target: {relation['target']}")

    return True


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Entity Deduplication Example")
    print("=" * 60)

    # Example entities with duplicates
    entities = [
        {"id": "e1", "label": "Einstein", "type": "Person"},
        {"id": "e2", "label": "Albert Einstein", "type": "Person"},
        {"id": "e3", "label": "A. Einstein", "type": "Person"},
        {"id": "e4", "label": "Microsoft", "type": "Organization"},
        {"id": "e5", "label": "Microsoft Corporation", "type": "Organization"},
        {"id": "e6", "label": "Marie Curie", "type": "Person"},
        {"id": "e7", "label": "Curie", "type": "Person"},
    ]

    relations = [
        {"source": "e1", "target": "e4", "type": "works_at", "evidence_span": "..."},
        {"source": "e2", "target": "e5", "type": "employed_by", "evidence_span": "..."},
        {"source": "e3", "target": "e4", "type": "works_at", "evidence_span": "..."},
        {"source": "e6", "target": "e7", "type": "same_person", "evidence_span": "..."},
    ]

    print(f"\nOriginal entities: {len(entities)}")
    for entity in entities:
        print(f"  - {entity['label']} ({entity['type']}) [ID: {entity['id']}]")

    print(f"\nOriginal relations: {len(relations)}")
    for relation in relations:
        source_label = next(e['label'] for e in entities if e['id'] == relation['source'])
        target_label = next(e['label'] for e in entities if e['id'] == relation['target'])
        print(f"  - {source_label} --[{relation['type']}]--> {target_label}")

    # Deduplicate
    print("\n" + "-" * 60)
    result = deduplicate_entities(
        entities,
        relations,
        threshold=0.85,
        show_progress=True
    )

    print("\n" + "-" * 60)
    print(f"Deduplicated entities: {len(result['entities'])}")
    for entity in result['entities']:
        print(f"  - {entity['label']} ({entity['type']}) [ID: {entity['id']}]")

    print(f"\nDeduplicated relations: {len(result['relations'])}")
    for relation in result['relations']:
        source_label = next(e['label'] for e in result['entities'] if e['id'] == relation['source'])
        target_label = next(e['label'] for e in result['entities'] if e['id'] == relation['target'])
        print(f"  - {source_label} --[{relation['type']}]--> {target_label}")

    print("\n" + "-" * 60)
    print("Entity Mapping (Old ID -> Canonical ID):")
    for old_id, canonical_id in result['entity_mapping'].items():
        old_label = next(e['label'] for e in entities if e['id'] == old_id)
        canonical_label = next(e['label'] for e in result['entities'] if e['id'] == canonical_id)
        if old_id != canonical_id:
            print(f"  {old_id} ({old_label}) -> {canonical_id} ({canonical_label})")

    print("\n" + "=" * 60)
    stats = get_deduplication_statistics(
        entities,
        result['entities'],
        result['entity_mapping']
    )
    print(f"Reduction: {stats['reduction_count']} entities ({stats['reduction_percent']:.1f}%)")
    print(f"Average cluster size: {stats['avg_cluster_size']:.2f}")
    print("=" * 60)

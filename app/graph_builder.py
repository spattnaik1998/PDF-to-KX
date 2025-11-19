"""
Knowledge graph construction and serialization.

This module builds knowledge graphs from extracted and deduplicated entities
and relations, providing utilities for export, analysis, and visualization.

Example usage:
    >>> from app.graph_builder import build_graph_json, to_networkx
    >>>
    >>> entities = [
    ...     {"id": "e1", "label": "Einstein", "type": "Person"},
    ...     {"id": "e2", "label": "Microsoft", "type": "Organization"}
    ... ]
    >>> relations = [
    ...     {"source": "e1", "target": "e2", "type": "works_at"}
    ... ]
    >>>
    >>> graph_json = build_graph_json(entities, relations)
    >>> print(f"Graph has {len(graph_json['nodes'])} nodes and {len(graph_json['edges'])} edges")
    >>>
    >>> nx_graph = to_networkx(graph_json)
    >>> print(f"NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
"""

from typing import List, Dict, Any, Set
import networkx as nx
import json
from collections import Counter


def build_graph_json(
    entities: List[Dict],
    relations: List[Dict]
) -> Dict[str, Any]:
    """
    Build a JSON-serializable graph object from entities and relations.

    This is the main function for constructing a knowledge graph in JSON format.
    It validates that all edges reference valid nodes and preserves metadata.

    Args:
        entities (List[Dict]): Canonical entities with unique 'id', 'label', 'type'
        relations (List[Dict]): Relations with 'source', 'target', 'type'

    Returns:
        Dict with keys:
            - nodes: List[Dict] - Each node has 'id', 'label', 'type', plus metadata
            - edges: List[Dict] - Each edge has 'source', 'target', 'label', plus metadata

    Example:
        >>> entities = [
        ...     {"id": "e1", "label": "Einstein", "type": "Person"},
        ...     {"id": "e2", "label": "Relativity", "type": "Concept"}
        ... ]
        >>> relations = [
        ...     {"source": "e1", "target": "e2", "type": "developed"}
        ... ]
        >>> graph = build_graph_json(entities, relations)
        >>> graph["nodes"]
        [{'id': 'e1', 'label': 'Einstein', 'type': 'Person'},
         {'id': 'e2', 'label': 'Relativity', 'type': 'Concept'}]
        >>> graph["edges"]
        [{'source': 'e1', 'target': 'e2', 'label': 'developed'}]
    """
    # Build set of valid entity IDs for validation
    valid_entity_ids = {entity["id"] for entity in entities}

    # Convert entities to nodes format
    nodes = []
    for entity in entities:
        node = {
            "id": entity["id"],
            "label": entity.get("label", entity["id"]),
            "type": entity.get("type", "Unknown")
        }

        # Preserve additional metadata
        for key in entity:
            if key not in ["id", "label", "type"]:
                node[key] = entity[key]

        nodes.append(node)

    # Convert relations to edges format, filtering invalid references
    edges = []
    for relation in relations:
        source_id = relation.get("source")
        target_id = relation.get("target")

        # Validate that both source and target exist in entities
        if source_id not in valid_entity_ids:
            continue  # Skip edge with invalid source
        if target_id not in valid_entity_ids:
            continue  # Skip edge with invalid target

        # Avoid self-loops (optional, can be removed if self-loops are meaningful)
        if source_id == target_id:
            continue

        edge = {
            "source": source_id,
            "target": target_id,
            "label": relation.get("type", "related_to")
        }

        # Preserve additional metadata
        for key in relation:
            if key not in ["source", "target", "type"]:
                edge[key] = relation[key]

        edges.append(edge)

    return {
        "nodes": nodes,
        "edges": edges
    }


def to_networkx(graph_json: Dict) -> nx.DiGraph:
    """
    Convert a graph JSON into a directed NetworkX graph.

    This function creates a NetworkX DiGraph from the JSON representation,
    enabling advanced graph analysis and algorithms.

    Args:
        graph_json (Dict): Graph JSON with 'nodes' and 'edges' keys

    Returns:
        nx.DiGraph: Directed NetworkX graph with node and edge attributes

    Example:
        >>> graph_json = build_graph_json(entities, relations)
        >>> G = to_networkx(graph_json)
        >>> import networkx as nx
        >>> print(f"Density: {nx.density(G):.3f}")
        >>> print(f"In-degree of node e1: {G.in_degree('e1')}")
    """
    G = nx.DiGraph()

    # Add nodes with all attributes
    for node in graph_json.get("nodes", []):
        node_id = node["id"]
        # Remove 'id' from attributes to avoid duplication
        node_attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node_id, **node_attrs)

    # Add edges with all attributes
    for edge in graph_json.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        # Remove source/target from attributes
        edge_attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}
        G.add_edge(source, target, **edge_attrs)

    return G


def validate_graph_json(graph_json: Dict) -> bool:
    """
    Validate that a graph JSON has the expected structure.

    Args:
        graph_json (Dict): Graph JSON to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If validation fails

    Example:
        >>> graph = build_graph_json(entities, relations)
        >>> validate_graph_json(graph)
        True
    """
    # Check top-level keys
    if "nodes" not in graph_json or "edges" not in graph_json:
        raise ValueError("Graph JSON must have 'nodes' and 'edges' keys")

    # Validate nodes
    node_ids = set()
    for i, node in enumerate(graph_json["nodes"]):
        if "id" not in node:
            raise ValueError(f"Node {i} missing 'id' field")
        if "label" not in node:
            raise ValueError(f"Node {i} missing 'label' field")
        node_ids.add(node["id"])

    # Check for duplicate node IDs
    if len(node_ids) != len(graph_json["nodes"]):
        raise ValueError("Duplicate node IDs found")

    # Validate edges
    for i, edge in enumerate(graph_json["edges"]):
        if "source" not in edge or "target" not in edge:
            raise ValueError(f"Edge {i} missing 'source' or 'target' field")

        # Check that edge references valid nodes
        if edge["source"] not in node_ids:
            raise ValueError(f"Edge {i} references unknown source: {edge['source']}")
        if edge["target"] not in node_ids:
            raise ValueError(f"Edge {i} references unknown target: {edge['target']}")

    return True


def get_graph_statistics(graph_json: Dict) -> Dict[str, Any]:
    """
    Compute statistics about the graph.

    Args:
        graph_json (Dict): Graph JSON with 'nodes' and 'edges'

    Returns:
        Dict with statistics:
            - num_nodes: int
            - num_edges: int
            - node_types: Dict[str, int] - Count of each node type
            - edge_types: Dict[str, int] - Count of each edge type
            - density: float - Graph density
            - avg_degree: float - Average node degree
            - isolated_nodes: int - Nodes with no edges

    Example:
        >>> stats = get_graph_statistics(graph_json)
        >>> print(f"Graph density: {stats['density']:.3f}")
        >>> print(f"Node types: {stats['node_types']}")
    """
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])

    num_nodes = len(nodes)
    num_edges = len(edges)

    # Count node types
    node_types = Counter(node.get("type", "Unknown") for node in nodes)

    # Count edge types
    edge_types = Counter(edge.get("label", "unknown") for edge in edges)

    # Calculate density
    # Density = actual_edges / possible_edges
    # For directed graph: possible_edges = n * (n - 1)
    density = 0.0
    if num_nodes > 1:
        possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / possible_edges

    # Calculate average degree
    # For directed graph: total degree = 2 * num_edges
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0

    # Find isolated nodes (nodes with no edges)
    nodes_in_edges = set()
    for edge in edges:
        nodes_in_edges.add(edge["source"])
        nodes_in_edges.add(edge["target"])

    all_node_ids = {node["id"] for node in nodes}
    isolated_nodes = len(all_node_ids - nodes_in_edges)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "density": density,
        "avg_degree": avg_degree,
        "isolated_nodes": isolated_nodes
    }


def export_to_json_file(graph_json: Dict, filepath: str, indent: int = 2):
    """
    Export graph JSON to a file.

    Args:
        graph_json (Dict): Graph JSON
        filepath (str): Output file path
        indent (int): JSON indentation level

    Example:
        >>> export_to_json_file(graph_json, "knowledge_graph.json")
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, indent=indent, ensure_ascii=False)


def load_from_json_file(filepath: str) -> Dict:
    """
    Load graph JSON from a file.

    Args:
        filepath (str): Input file path

    Returns:
        Dict: Graph JSON

    Example:
        >>> graph = load_from_json_file("knowledge_graph.json")
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_to_cytoscape(graph_json: Dict) -> Dict[str, Any]:
    """
    Export graph in Cytoscape.js compatible format.

    Cytoscape.js expects format:
    {
      "elements": {
        "nodes": [{"data": {"id": "...", "label": "...", ...}}],
        "edges": [{"data": {"id": "...", "source": "...", "target": "...", ...}}]
      }
    }

    Args:
        graph_json (Dict): Graph JSON

    Returns:
        Dict: Cytoscape.js formatted data

    Example:
        >>> cytoscape_data = export_to_cytoscape(graph_json)
        >>> # Can be directly used with Cytoscape.js in web UI
    """
    cyto_nodes = []
    for node in graph_json.get("nodes", []):
        cyto_nodes.append({
            "data": node.copy()
        })

    cyto_edges = []
    for i, edge in enumerate(graph_json.get("edges", [])):
        edge_data = edge.copy()
        # Add unique edge ID
        edge_data["id"] = f"edge_{i}"
        cyto_edges.append({
            "data": edge_data
        })

    return {
        "elements": {
            "nodes": cyto_nodes,
            "edges": cyto_edges
        }
    }


def filter_graph_by_node_type(
    graph_json: Dict,
    node_types: List[str]
) -> Dict[str, Any]:
    """
    Filter graph to include only specific node types.

    Args:
        graph_json (Dict): Graph JSON
        node_types (List[str]): List of node types to keep

    Returns:
        Dict: Filtered graph JSON

    Example:
        >>> # Keep only Person and Organization nodes
        >>> filtered = filter_graph_by_node_type(graph_json, ["Person", "Organization"])
    """
    # Filter nodes
    filtered_nodes = [
        node for node in graph_json.get("nodes", [])
        if node.get("type") in node_types
    ]

    # Get filtered node IDs
    filtered_node_ids = {node["id"] for node in filtered_nodes}

    # Filter edges to only include those between filtered nodes
    filtered_edges = [
        edge for edge in graph_json.get("edges", [])
        if edge["source"] in filtered_node_ids and edge["target"] in filtered_node_ids
    ]

    return {
        "nodes": filtered_nodes,
        "edges": filtered_edges
    }


def get_node_degree_statistics(graph_json: Dict) -> Dict[str, Any]:
    """
    Compute degree statistics for nodes in the graph.

    Args:
        graph_json (Dict): Graph JSON

    Returns:
        Dict with:
            - in_degree: Dict[str, int] - In-degree for each node
            - out_degree: Dict[str, int] - Out-degree for each node
            - total_degree: Dict[str, int] - Total degree for each node
            - top_in_degree: List[Tuple[str, int]] - Top 10 nodes by in-degree
            - top_out_degree: List[Tuple[str, int]] - Top 10 nodes by out-degree

    Example:
        >>> degree_stats = get_node_degree_statistics(graph_json)
        >>> print("Most referenced nodes:")
        >>> for node_id, degree in degree_stats['top_in_degree'][:5]:
        ...     print(f"  {node_id}: {degree}")
    """
    # Initialize degree dictionaries
    in_degree = {}
    out_degree = {}

    # Initialize all nodes with 0 degree
    for node in graph_json.get("nodes", []):
        node_id = node["id"]
        in_degree[node_id] = 0
        out_degree[node_id] = 0

    # Count degrees from edges
    for edge in graph_json.get("edges", []):
        source = edge["source"]
        target = edge["target"]

        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[target] = in_degree.get(target, 0) + 1

    # Compute total degree
    all_node_ids = {node["id"] for node in graph_json.get("nodes", [])}
    total_degree = {
        node_id: in_degree.get(node_id, 0) + out_degree.get(node_id, 0)
        for node_id in all_node_ids
    }

    # Get top nodes
    top_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    top_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "top_in_degree": top_in_degree,
        "top_out_degree": top_out_degree
    }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge Graph Builder Example")
    print("=" * 60)

    # Example entities and relations
    entities = [
        {"id": "e1", "label": "Albert Einstein", "type": "Person"},
        {"id": "e2", "label": "Theory of Relativity", "type": "Concept"},
        {"id": "e3", "label": "Nobel Prize in Physics", "type": "Award"},
        {"id": "e4", "label": "Princeton University", "type": "Organization"},
        {"id": "e5", "label": "Germany", "type": "Location"},
    ]

    relations = [
        {"source": "e1", "target": "e2", "type": "developed", "evidence_span": "Einstein developed relativity"},
        {"source": "e1", "target": "e3", "type": "won", "evidence_span": "Einstein won the Nobel Prize"},
        {"source": "e1", "target": "e4", "type": "worked_at", "evidence_span": "Einstein worked at Princeton"},
        {"source": "e1", "target": "e5", "type": "born_in", "evidence_span": "Einstein was born in Germany"},
        {"source": "e4", "target": "e5", "type": "located_in", "evidence_span": "Princeton is in the United States"},
    ]

    print("\n1. Building graph JSON...")
    graph_json = build_graph_json(entities, relations)

    print(f"\nGraph JSON structure:")
    print(f"  Nodes: {len(graph_json['nodes'])}")
    print(f"  Edges: {len(graph_json['edges'])}")

    print("\n2. Validating graph...")
    try:
        validate_graph_json(graph_json)
        print("  ✓ Graph is valid")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")

    print("\n3. Computing statistics...")
    stats = get_graph_statistics(graph_json)
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.3f}")
    print(f"  Average degree: {stats['avg_degree']:.2f}")
    print(f"  Isolated nodes: {stats['isolated_nodes']}")
    print(f"\n  Node types:")
    for node_type, count in stats['node_types'].items():
        print(f"    - {node_type}: {count}")
    print(f"\n  Edge types:")
    for edge_type, count in stats['edge_types'].items():
        print(f"    - {edge_type}: {count}")

    print("\n4. Converting to NetworkX...")
    nx_graph = to_networkx(graph_json)
    print(f"  NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
    print(f"  Is directed: {nx_graph.is_directed()}")

    print("\n5. Node degree statistics...")
    degree_stats = get_node_degree_statistics(graph_json)
    print("  Top nodes by in-degree:")
    for node_id, degree in degree_stats['top_in_degree'][:5]:
        node = next(n for n in entities if n['id'] == node_id)
        print(f"    - {node['label']}: {degree}")

    print("\n6. Exporting to Cytoscape format...")
    cytoscape_data = export_to_cytoscape(graph_json)
    print(f"  Cytoscape elements: {len(cytoscape_data['elements']['nodes'])} nodes, {len(cytoscape_data['elements']['edges'])} edges")

    print("\n7. Filtering by node type...")
    person_org_graph = filter_graph_by_node_type(graph_json, ["Person", "Organization"])
    print(f"  Filtered graph (Person & Organization only):")
    print(f"    Nodes: {len(person_org_graph['nodes'])}")
    print(f"    Edges: {len(person_org_graph['edges'])}")

    print("\n" + "=" * 60)
    print("Example JSON output (first node and edge):")
    print(f"\nNode: {json.dumps(graph_json['nodes'][0], indent=2)}")
    print(f"\nEdge: {json.dumps(graph_json['edges'][0], indent=2)}")
    print("=" * 60)

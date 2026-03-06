"""
Graph analytics engine for the Knowledge Graph platform.

Computes centrality metrics, community structure, and generates
human-readable AI insights from extracted graph data.
"""

import networkx as nx
from typing import Dict, List, Any
from collections import Counter


def compute_graph_analytics(graph_json: Dict) -> Dict[str, Any]:
    """
    Compute comprehensive analytics for a knowledge graph.

    Args:
        graph_json: Graph JSON with 'nodes' and 'edges' keys

    Returns:
        Dict containing metrics, distributions, and ranked entity lists
    """
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])

    empty = {
        "num_nodes": 0, "num_edges": 0, "density": 0.0,
        "num_components": 0, "avg_degree": 0.0, "isolated_nodes": 0,
        "entity_types": {}, "relation_types": {},
        "top_entities_by_degree": [], "top_central_entities": [],
    }

    if not nodes:
        return empty

    # Build NetworkX directed graph
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in edges:
        G.add_edge(
            edge["source"], edge["target"],
            label=edge.get("label", "")
        )

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Degree maps
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes()}

    # Top entities ranked by total degree
    top_entities = sorted(
        [
            (
                G.nodes[n].get("label", n),
                total_deg[n],
                G.nodes[n].get("type", "Unknown"),
            )
            for n in G.nodes()
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    # Distribution counts
    entity_types = Counter(node.get("type", "Unknown") for node in nodes)
    relation_types = Counter(edge.get("label", "unknown") for edge in edges)

    # Graph structural metrics
    density = nx.density(G)
    G_undir = G.to_undirected()
    num_components = nx.number_connected_components(G_undir)
    isolated = len([n for n in G.nodes() if G.degree(n) == 0])
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0

    # Degree centrality for top-10 hub nodes
    top_central: List = []
    if num_nodes > 1:
        deg_centrality = nx.degree_centrality(G)
        top_central = sorted(
            deg_centrality.items(), key=lambda x: x[1], reverse=True
        )[:10]
        top_central = [
            (G.nodes[n].get("label", n), round(score, 4))
            for n, score in top_central
        ]

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 4),
        "num_components": num_components,
        "avg_degree": round(avg_degree, 2),
        "isolated_nodes": isolated,
        "entity_types": dict(entity_types),
        "relation_types": dict(relation_types),
        "top_entities_by_degree": top_entities,
        "top_central_entities": top_central,
    }


def generate_insights(analytics: Dict) -> List[str]:
    """
    Generate human-readable insights from graph analytics.

    Args:
        analytics: Output from compute_graph_analytics()

    Returns:
        List of markdown-formatted insight strings
    """
    insights: List[str] = []

    n = analytics.get("num_nodes", 0)
    e = analytics.get("num_edges", 0)
    density = analytics.get("density", 0.0)
    components = analytics.get("num_components", 1)
    entity_types = analytics.get("entity_types", {})
    top_entities = analytics.get("top_entities_by_degree", [])
    isolated = analytics.get("isolated_nodes", 0)
    avg_deg = analytics.get("avg_degree", 0.0)

    if n > 0:
        insights.append(
            f"**{n} unique entities** and **{e} relationships** "
            f"extracted from this document."
        )

    if top_entities:
        label, degree, etype = top_entities[0]
        insights.append(
            f"**{label}** ({etype}) is the most connected entity "
            f"with **{degree} direct relationships**."
        )

    if density > 0:
        if density > 0.3:
            insights.append(
                "The graph is **highly interconnected** — entities have "
                "rich, complex interdependencies throughout the document."
            )
        elif density > 0.1:
            insights.append(
                "The graph has **moderate connectivity**, indicating "
                "a well-structured document with clear topic clusters."
            )
        else:
            insights.append(
                "The graph is **sparsely connected**, typical of broad "
                "documents covering many independent topics."
            )

    if components > 1:
        insights.append(
            f"**{components} distinct knowledge clusters** detected — "
            f"consider cross-referencing these topic groups."
        )

    if entity_types:
        dominant = max(entity_types.items(), key=lambda x: x[1])
        pct = round(dominant[1] / n * 100) if n > 0 else 0
        insights.append(
            f"**{dominant[0]}** is the most prevalent entity type, "
            f"representing **{pct}%** of all entities."
        )

    if avg_deg > 0:
        insights.append(
            f"Entities have an average of **{avg_deg} connections** each."
        )

    if isolated > 0:
        insights.append(
            f"**{isolated} isolated entities** detected — these may "
            f"represent standalone concepts worth reviewing manually."
        )

    return insights

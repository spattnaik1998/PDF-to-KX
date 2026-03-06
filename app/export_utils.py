"""
Multi-format export utilities for knowledge graph data.

Supports JSON, CSV, Excel (multi-sheet), and GraphML formats.
"""

import io
import json
import networkx as nx
from typing import Dict, List, Any


def to_json_string(graph_json: Dict) -> str:
    """Serialize graph JSON to a pretty-printed string."""
    return json.dumps(graph_json, indent=2, ensure_ascii=False)


def to_entities_csv(entities: List[Dict]) -> str:
    """Export entities to CSV, with priority column ordering."""
    try:
        import pandas as pd

        if not entities:
            return "id,label,type\n"

        df = pd.DataFrame(entities)
        priority = ["label", "type", "id", "chunk_id"]
        ordered = [c for c in priority if c in df.columns]
        rest = [c for c in df.columns if c not in priority]
        return df[ordered + rest].to_csv(index=False)
    except ImportError:
        if not entities:
            return ""
        header = ",".join(entities[0].keys())
        rows = [",".join(f'"{str(v)}"' for v in e.values()) for e in entities]
        return "\n".join([header] + rows)


def to_relations_csv(relations: List[Dict]) -> str:
    """Export relations to CSV with human-readable column ordering."""
    try:
        import pandas as pd

        if not relations:
            return "source,target,type,evidence_span\n"

        df = pd.DataFrame(relations)
        priority = ["source", "target", "type", "evidence_span", "chunk_id"]
        ordered = [c for c in priority if c in df.columns]
        rest = [c for c in df.columns if c not in priority]
        return df[ordered + rest].to_csv(index=False)
    except ImportError:
        if not relations:
            return ""
        header = ",".join(relations[0].keys())
        rows = [",".join(f'"{str(v)}"' for v in r.values()) for r in relations]
        return "\n".join([header] + rows)


def to_excel_bytes(
    entities: List[Dict],
    relations: List[Dict],
    stats: Dict[str, Any],
) -> bytes:
    """
    Export a multi-sheet Excel workbook containing entities, relations,
    and graph statistics.

    Requires openpyxl: pip install openpyxl
    """
    import pandas as pd

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if entities:
            df_ent = pd.DataFrame(entities)
            priority = ["label", "type", "id", "chunk_id"]
            ordered = [c for c in priority if c in df_ent.columns]
            rest = [c for c in df_ent.columns if c not in priority]
            df_ent[ordered + rest].to_excel(
                writer, sheet_name="Entities", index=False
            )

        if relations:
            df_rel = pd.DataFrame(relations)
            priority = ["source", "target", "type", "evidence_span"]
            ordered = [c for c in priority if c in df_rel.columns]
            rest = [c for c in df_rel.columns if c not in priority]
            df_rel[ordered + rest].to_excel(
                writer, sheet_name="Relationships", index=False
            )

        # Flatten stats — skip nested dicts (node_types, edge_types)
        flat_stats = [
            {"Metric": k, "Value": v}
            for k, v in stats.items()
            if not isinstance(v, (dict, list))
        ]
        if flat_stats:
            pd.DataFrame(flat_stats).to_excel(
                writer, sheet_name="Statistics", index=False
            )

    return output.getvalue()


def to_graphml_string(graph_json: Dict) -> str:
    """
    Export graph as GraphML for use with Gephi, Cytoscape, or
    other professional graph analysis tools.
    """
    G = nx.DiGraph()
    for node in graph_json.get("nodes", []):
        G.add_node(
            node["id"],
            label=node.get("label", ""),
            node_type=node.get("type", ""),
        )
    for edge in graph_json.get("edges", []):
        G.add_edge(
            edge["source"],
            edge["target"],
            label=edge.get("label", ""),
        )

    buf = io.BytesIO()
    nx.write_graphml(G, buf)
    return buf.getvalue().decode("utf-8")

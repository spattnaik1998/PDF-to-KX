"""
KnowledgeX — AI-Powered Knowledge Graph Platform
Professional Streamlit UI for PDF knowledge graph extraction,
visualization, analytics, and multi-format export.
"""

import json
import os
import sys
import tempfile
from typing import Dict, List, Optional

# Ensure the project root is on sys.path so `app.*` is importable
# regardless of the working directory when Streamlit is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import streamlit.components.v1 as components

# ── Page config must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="KnowledgeX | AI Knowledge Graph Platform",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            "**KnowledgeX v2.0** — Transform PDF documents into interactive "
            "knowledge graphs using AI-powered entity and relationship extraction."
        ),
    },
)

# ── Deferred heavy imports ──────────────────────────────────────────────────
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network

from app.analytics import compute_graph_analytics, generate_insights
from app.chunking import chunk_pages
from app.config import settings
from app.deduplication import deduplicate_entities
from app.export_utils import (
    to_entities_csv,
    to_excel_bytes,
    to_graphml_string,
    to_json_string,
    to_relations_csv,
)
from app.extraction_hf import extract_from_chunks as extract_hf
from app.extraction_openai import extract_from_chunks as extract_openai
from app.graph_builder import build_graph_json, get_graph_statistics
from app.pdf_utils import extract_pages_with_metadata

# ── Constants ───────────────────────────────────────────────────────────────

APP_NAME = "KnowledgeX"
APP_VERSION = "2.0"

TYPE_COLORS: Dict[str, str] = {
    "Person":       "#42A5F5",
    "Organization": "#AB47BC",
    "Location":     "#66BB6A",
    "Concept":      "#FFA726",
    "Event":        "#EF5350",
    "Product":      "#26C6DA",
    "Technology":   "#78909C",
    "Date":         "#FF7043",
    "Award":        "#FFD700",
    "Default":      "#90A4AE",
}

# ── CSS ─────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Global resets ── */
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] {
    background: #0A1628;
}
[data-testid="stSidebar"] {
    background: #0A1F3A !important;
    border-right: 1px solid rgba(30, 111, 255, 0.15);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1F3C;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(30, 111, 255, 0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8DA0B3;
    font-weight: 500;
    padding: 8px 18px;
    font-size: 0.88rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: rgba(30, 111, 255, 0.18) !important;
    color: #42A5F5 !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1E6FFF 0%, #1557CC 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(30, 111, 255, 0.25) !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: linear-gradient(135deg, #3380FF 0%, #1E6FFF 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(30, 111, 255, 0.45) !important;
}
.stButton > button[kind="secondary"] {
    border: 1px solid rgba(30, 111, 255, 0.3) !important;
    color: #8DA0B3 !important;
    border-radius: 8px !important;
    background: transparent !important;
}
.stButton > button[kind="secondary"]:hover:not(:disabled) {
    border-color: #1E6FFF !important;
    color: #42A5F5 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(30, 111, 255, 0.3) !important;
    border-radius: 12px !important;
    background: rgba(30, 111, 255, 0.04) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(30, 111, 255, 0.6) !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(30, 111, 255, 0.1) !important;
    border: 1px solid rgba(30, 111, 255, 0.3) !important;
    border-radius: 8px !important;
    color: #42A5F5 !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(30, 111, 255, 0.2) !important;
    border-color: #1E6FFF !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0D1F3C !important;
    border-radius: 8px !important;
    border: 1px solid rgba(30, 111, 255, 0.12) !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(30, 111, 255, 0.12) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    background: #0A1D38 !important;
}

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: #0D1F3C !important;
    border-radius: 12px !important;
    border: 1px solid rgba(30, 111, 255, 0.2) !important;
}

/* ── Multiselect ── */
[data-baseweb="select"] > div {
    background: #0D1F3C !important;
    border: 1px solid rgba(30, 111, 255, 0.2) !important;
    border-radius: 8px !important;
}

/* ── Text input ── */
.stTextInput input {
    background: #0D1F3C !important;
    border: 1px solid rgba(30, 111, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #E8EDF5 !important;
}
.stTextInput input:focus {
    border-color: #1E6FFF !important;
    box-shadow: 0 0 0 2px rgba(30, 111, 255, 0.15) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(30, 111, 255, 0.12) !important;
}

/* ── Divider ── */
hr { border-color: rgba(30, 111, 255, 0.12) !important; }

/* ── Warning / info ── */
.stAlert { border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A1628; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1E6FFF; }

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: #1E6FFF !important;
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────

_STATE_DEFAULTS = {
    "processed": False,
    "graph_json": None,
    "entities": [],
    "relations": [],
    "stats": {},
    "analytics": {},
    "insights": [],
    "filename": "",
    "num_pages": 0,
    "num_chunks": 0,
    "entity_count": 0,
    "relation_count": 0,
    "processing_error": None,
}


def init_session_state() -> None:
    for key, val in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session_state() -> None:
    for key, val in _STATE_DEFAULTS.items():
        st.session_state[key] = val


# ── Graph visualization ──────────────────────────────────────────────────────

def _node_color(node_type: str) -> str:
    return TYPE_COLORS.get(node_type, TYPE_COLORS["Default"])


def build_pyvis_html(
    graph_json: Dict,
    filter_types: Optional[List[str]] = None,
) -> str:
    """
    Build a self-contained pyvis HTML string for embedding in Streamlit.

    Nodes are sized by degree centrality and colored by entity type.
    """
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])

    if filter_types:
        keep_ids = {n["id"] for n in nodes if n.get("type", "Default") in filter_types}
        nodes = [n for n in nodes if n["id"] in keep_ids]
        edges = [
            e for e in edges
            if e["source"] in keep_ids and e["target"] in keep_ids
        ]

    # Degree count for node sizing
    degree_map: Dict[str, int] = {}
    for edge in edges:
        degree_map[edge["source"]] = degree_map.get(edge["source"], 0) + 1
        degree_map[edge["target"]] = degree_map.get(edge["target"], 0) + 1

    try:
        net = Network(
            height="680px",
            width="100%",
            bgcolor="#0A1628",
            font_color="#E8EDF5",
            directed=True,
            cdn_resources="in_line",
        )
    except TypeError:
        net = Network(
            height="680px",
            width="100%",
            bgcolor="#0A1628",
            font_color="#E8EDF5",
            directed=True,
        )

    for node in nodes:
        nid = node["id"]
        label = node.get("label", nid)
        ntype = node.get("type", "Default")
        color = _node_color(ntype)
        degree = degree_map.get(nid, 0)
        size = max(14, min(55, 12 + degree * 4))

        display_label = label[:28] + ("…" if len(label) > 28 else "")
        tooltip = (
            f"<div style='font-family:sans-serif;padding:10px 12px;"
            f"background:#132035;border-radius:8px;"
            f"border:1px solid rgba(30,111,255,0.35)'>"
            f"<b style='color:{color};font-size:1.05em'>{label}</b><br>"
            f"<span style='color:#8DA0B3;font-size:0.85em'>Type: {ntype}</span><br>"
            f"<span style='color:#8DA0B3;font-size:0.85em'>Connections: {degree}</span>"
            f"</div>"
        )

        net.add_node(
            nid,
            label=display_label,
            title=tooltip,
            color={
                "background": color,
                "border": "rgba(255,255,255,0.15)",
                "highlight": {"background": color, "border": "#ffffff"},
                "hover": {"background": color, "border": "#ffffff80"},
            },
            size=size,
            font={"size": 11, "color": "#E8EDF5"},
            borderWidth=1,
            shadow={"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 8},
        )

    for edge in edges:
        elabel = edge.get("label", "")
        net.add_edge(
            edge["source"],
            edge["target"],
            title=elabel,
            label=elabel[:22] if elabel else "",
            color={"color": "#1E3A5F", "highlight": "#1E6FFF", "hover": "#42A5F5"},
            arrows={"to": {"enabled": True, "scaleFactor": 0.65}},
            smooth={"type": "curvedCW", "roundness": 0.15},
            font={"size": 9, "color": "#6B8BAE", "align": "middle"},
            width=1,
        )

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -6500,
          "centralGravity": 0.2,
          "springLength": 190,
          "springConstant": 0.04,
          "damping": 0.1,
          "avoidOverlap": 0.15
        },
        "maxVelocity": 50,
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": { "enabled": true, "iterations": 250 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true,
        "navigationButtons": true,
        "keyboard": { "enabled": true }
      }
    }
    """)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    )
    tmp.close()
    net.save_graph(tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as fh:
        html = fh.read()
    os.unlink(tmp.name)
    return html


# ── Tab renderers ────────────────────────────────────────────────────────────

def render_graph_tab(graph_json: Dict, analytics: Dict) -> None:
    nodes = graph_json.get("nodes", [])
    if not nodes:
        st.info("No entities were extracted. Try adjusting chunk size or deduplication threshold.")
        return

    if len(nodes) > 400:
        st.warning(
            f"Large graph detected ({len(nodes)} nodes). "
            "Filter entity types below to improve rendering performance."
        )

    all_types = sorted({n.get("type", "Default") for n in nodes})

    col_filter, col_opts = st.columns([4, 1])
    with col_filter:
        selected_types = st.multiselect(
            "Filter by entity type",
            options=all_types,
            default=all_types,
            key="graph_type_filter",
        )
    with col_opts:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        show_legend = st.checkbox("Legend", value=True, key="show_legend")

    if not selected_types:
        st.warning("Select at least one entity type to display the graph.")
        return

    active_filter = (
        selected_types if set(selected_types) != set(all_types) else None
    )
    filtered_count = (
        sum(1 for n in nodes if n.get("type", "Default") in selected_types)
        if active_filter else len(nodes)
    )
    st.caption(f"Displaying **{filtered_count}** of **{len(nodes)}** entities")

    with st.spinner("Rendering graph…"):
        graph_html = build_pyvis_html(graph_json, filter_types=active_filter)

    components.html(graph_html, height=710, scrolling=False)

    if show_legend:
        present_types = (
            selected_types if active_filter else all_types
        )
        legend_items = "".join(
            f'<div style="display:flex;align-items:center;gap:7px">'
            f'<div style="width:11px;height:11px;border-radius:50%;'
            f'background:{_node_color(t)};flex-shrink:0"></div>'
            f'<span style="color:#C5D0DE;font-size:0.8em">{t}</span>'
            f'</div>'
            for t in present_types
        )
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;'
            f'padding:12px 16px;background:#0D1F3C;border-radius:8px;'
            f'border:1px solid rgba(30,111,255,0.12)">{legend_items}</div>',
            unsafe_allow_html=True,
        )


def render_analytics_tab(
    analytics: Dict, insights: List[str], filename: str
) -> None:

    # ── KPI strip ──────────────────────────────────────────────────────────
    n = analytics.get("num_nodes", 0)
    e = analytics.get("num_edges", 0)
    ntypes = len(analytics.get("entity_types", {}))
    density = analytics.get("density", 0.0)

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);
                    gap:14px;margin-bottom:24px">
          <div style="background:linear-gradient(135deg,#132035,#1a2f4a);
                      border:1px solid rgba(66,165,245,0.25);border-radius:14px;
                      padding:22px;text-align:center">
            <div style="font-size:2.4em;font-weight:800;color:#42A5F5;
                        line-height:1">{n}</div>
            <div style="font-size:0.72em;color:#8DA0B3;letter-spacing:0.12em;
                        margin-top:6px">ENTITIES</div>
          </div>
          <div style="background:linear-gradient(135deg,#132035,#1a2f4a);
                      border:1px solid rgba(171,71,188,0.25);border-radius:14px;
                      padding:22px;text-align:center">
            <div style="font-size:2.4em;font-weight:800;color:#AB47BC;
                        line-height:1">{e}</div>
            <div style="font-size:0.72em;color:#8DA0B3;letter-spacing:0.12em;
                        margin-top:6px">RELATIONSHIPS</div>
          </div>
          <div style="background:linear-gradient(135deg,#132035,#1a2f4a);
                      border:1px solid rgba(102,187,106,0.25);border-radius:14px;
                      padding:22px;text-align:center">
            <div style="font-size:2.4em;font-weight:800;color:#66BB6A;
                        line-height:1">{ntypes}</div>
            <div style="font-size:0.72em;color:#8DA0B3;letter-spacing:0.12em;
                        margin-top:6px">ENTITY TYPES</div>
          </div>
          <div style="background:linear-gradient(135deg,#132035,#1a2f4a);
                      border:1px solid rgba(255,167,38,0.25);border-radius:14px;
                      padding:22px;text-align:center">
            <div style="font-size:2.4em;font-weight:800;color:#FFA726;
                        line-height:1">{density:.3f}</div>
            <div style="font-size:0.72em;color:#8DA0B3;letter-spacing:0.12em;
                        margin-top:6px">GRAPH DENSITY</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── AI Insights ────────────────────────────────────────────────────────
    if insights:
        st.markdown("#### AI Insights")
        bullets = "".join(
            f'<div style="color:#C5D0DE;padding:5px 0 5px 4px;font-size:0.9em">'
            f'<span style="color:#1E6FFF;margin-right:8px">▸</span>{i}</div>'
            for i in insights
        )
        st.markdown(
            f'<div style="background:#0D1F3C;border-left:3px solid #1E6FFF;'
            f'border-radius:0 10px 10px 0;padding:14px 20px;margin-bottom:24px">'
            f'{bullets}</div>',
            unsafe_allow_html=True,
        )

    # ── Charts row 1 ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    entity_types = analytics.get("entity_types", {})
    relation_types = analytics.get("relation_types", {})

    with col1:
        if entity_types:
            st.markdown("##### Entity Type Distribution")
            colors = [_node_color(t) for t in entity_types]
            fig = go.Figure(
                go.Pie(
                    labels=list(entity_types.keys()),
                    values=list(entity_types.values()),
                    hole=0.58,
                    marker=dict(
                        colors=colors,
                        line=dict(color="#0A1628", width=2),
                    ),
                    textfont=dict(color="#E8EDF5", size=12),
                    hovertemplate="<b>%{label}</b><br>%{value} entities · %{percent}<extra></extra>",
                )
            )
            total = sum(entity_types.values())
            fig.add_annotation(
                text=f"<b>{total}</b><br><span style='font-size:0.8em'>Total</span>",
                x=0.5, y=0.5,
                font=dict(size=16, color="#E8EDF5"),
                showarrow=False,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=8, r=8, t=8, b=8),
                legend=dict(
                    font=dict(size=11, color="#C5D0DE"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        if relation_types:
            st.markdown("##### Top Relationship Types")
            top_rels = sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]
            rl, rv = zip(*top_rels) if top_rels else ([], [])
            fig2 = go.Figure(
                go.Bar(
                    x=list(rv),
                    y=list(rl),
                    orientation="h",
                    marker=dict(
                        color=list(rv),
                        colorscale=[[0, "#1557CC"], [0.5, "#1E6FFF"], [1, "#42A5F5"]],
                        showscale=False,
                        line=dict(width=0),
                    ),
                    text=list(rv),
                    textposition="outside",
                    textfont=dict(color="#8DA0B3", size=11),
                    hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
                )
            )
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=8, r=60, t=8, b=8),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Top entities chart ─────────────────────────────────────────────────
    top_entities = analytics.get("top_entities_by_degree", [])
    if top_entities:
        st.markdown("##### Most Connected Entities")
        top_n = top_entities[:12]
        labels = [e[0][:35] for e in top_n]
        values = [e[1] for e in top_n]
        types = [e[2] for e in top_n]
        colors_bar = [_node_color(t) for t in types]

        fig3 = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=colors_bar, line=dict(width=0)),
                text=values,
                textposition="outside",
                textfont=dict(color="#8DA0B3", size=11),
                hovertemplate="<b>%{y}</b><br>Connections: %{x}<extra></extra>",
            )
        )
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(l=8, r=60, t=8, b=8),
            xaxis=dict(showgrid=False, showticklabels=False, title=None),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Secondary metrics row ──────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    c3, c4, c5, c6 = st.columns(4)
    secondary = [
        (c3, "COMPONENTS",        analytics.get("num_components", 1), "#42A5F5"),
        (c4, "AVG. CONNECTIONS",  analytics.get("avg_degree", 0.0),   "#AB47BC"),
        (c5, "ISOLATED ENTITIES", analytics.get("isolated_nodes", 0), "#EF5350"),
        (c6, "PAGES PROCESSED",  st.session_state.get("num_pages", 0), "#66BB6A"),
    ]
    for col, label, value, color in secondary:
        with col:
            st.markdown(
                f'<div style="background:#0D1F3C;border-radius:10px;padding:16px;'
                f'border:1px solid rgba(30,111,255,0.12);text-align:center">'
                f'<div style="color:#8DA0B3;font-size:0.72em;letter-spacing:0.1em;'
                f'margin-bottom:6px">{label}</div>'
                f'<div style="color:{color};font-size:1.6em;font-weight:700">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_data_tab(entities: List[Dict], relations: List[Dict]) -> None:
    inner_tabs = st.tabs(["Entities", "Relationships"])

    with inner_tabs[0]:
        if not entities:
            st.info("No entities to display.")
        else:
            df_ent = pd.DataFrame(entities)
            priority = ["label", "type", "id", "chunk_id"]
            ordered = [c for c in priority if c in df_ent.columns]
            rest = [c for c in df_ent.columns if c not in priority]
            df_ent = df_ent[ordered + rest]

            search = st.text_input(
                "Search entities", placeholder="Filter by name, type…",
                key="ent_search"
            )
            if search:
                mask = df_ent.apply(
                    lambda row: row.astype(str)
                    .str.contains(search, case=False, na=False)
                    .any(),
                    axis=1,
                )
                df_ent = df_ent[mask]

            st.caption(f"Showing **{len(df_ent)}** of **{len(entities)}** entities")
            st.dataframe(
                df_ent,
                use_container_width=True,
                height=420,
                column_config={
                    "label": st.column_config.TextColumn("Entity", width="medium"),
                    "type":  st.column_config.TextColumn("Type",   width="small"),
                    "id":    st.column_config.TextColumn("ID",     width="medium"),
                },
            )

    with inner_tabs[1]:
        if not relations:
            st.info("No relationships to display.")
        else:
            entity_label_map = {e["id"]: e.get("label", e["id"]) for e in entities}
            display_rows = [
                {
                    "Source":       entity_label_map.get(r["source"], r["source"]),
                    "Relationship": r.get("type", ""),
                    "Target":       entity_label_map.get(r["target"], r["target"]),
                    "Evidence":     r.get("evidence_span", ""),
                }
                for r in relations
            ]
            df_rel = pd.DataFrame(display_rows)

            search_r = st.text_input(
                "Search relationships", placeholder="Filter by entity or type…",
                key="rel_search"
            )
            if search_r:
                mask = df_rel.apply(
                    lambda row: row.astype(str)
                    .str.contains(search_r, case=False, na=False)
                    .any(),
                    axis=1,
                )
                df_rel = df_rel[mask]

            st.caption(f"Showing **{len(df_rel)}** of **{len(relations)}** relationships")
            st.dataframe(
                df_rel,
                use_container_width=True,
                height=420,
                column_config={
                    "Source":       st.column_config.TextColumn("Source", width="medium"),
                    "Relationship": st.column_config.TextColumn("Relationship", width="medium"),
                    "Target":       st.column_config.TextColumn("Target", width="medium"),
                    "Evidence":     st.column_config.TextColumn("Evidence", width="large"),
                },
            )


def _export_card(title: str, description: str) -> None:
    st.markdown(
        f'<div style="background:#0D1F3C;border-radius:12px;padding:18px 20px;'
        f'margin-bottom:12px;border:1px solid rgba(30,111,255,0.14)">'
        f'<div style="font-size:1em;font-weight:600;color:#E8EDF5;margin-bottom:5px">'
        f'{title}</div>'
        f'<div style="color:#8DA0B3;font-size:0.82em;margin-bottom:14px;line-height:1.5">'
        f'{description}</div></div>',
        unsafe_allow_html=True,
    )


def render_export_tab(
    graph_json: Dict,
    entities: List[Dict],
    relations: List[Dict],
    stats: Dict,
) -> None:
    st.markdown(
        '<p style="color:#8DA0B3;margin-bottom:24px">Download your knowledge graph '
        "in multiple formats for use in downstream analysis, BI tools, or graph "
        "databases.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        _export_card(
            "📄 Full Graph — JSON",
            "Complete graph with nodes and edges. Compatible with "
            "JavaScript graph libraries, databases, and custom pipelines.",
        )
        st.download_button(
            "⬇ Download JSON",
            data=to_json_string(graph_json).encode("utf-8"),
            file_name="knowledge_graph.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        _export_card(
            "👤 Entities — CSV",
            "All extracted entities with IDs, labels, types, and source chunk. "
            "Open directly in Excel, Google Sheets, or any data tool.",
        )
        st.download_button(
            "⬇ Download Entities CSV",
            data=to_entities_csv(entities).encode("utf-8"),
            file_name="entities.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        _export_card(
            "🔬 Graph — GraphML",
            "Industry-standard format for Gephi, Cytoscape, and Neo4j. "
            "Ideal for network analysis and professional graph databases.",
        )
        try:
            gml = to_graphml_string(graph_json)
            st.download_button(
                "⬇ Download GraphML",
                data=gml.encode("utf-8"),
                file_name="knowledge_graph.graphml",
                mime="application/xml",
                use_container_width=True,
            )
        except Exception:
            st.button(
                "⬇ Download GraphML",
                disabled=True,
                use_container_width=True,
                help="GraphML export requires networkx",
            )

    with col2:
        _export_card(
            "🔗 Relationships — CSV",
            "All extracted relationships with source, target, relationship type, "
            "and supporting evidence spans from the original text.",
        )
        st.download_button(
            "⬇ Download Relations CSV",
            data=to_relations_csv(relations).encode("utf-8"),
            file_name="relations.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        _export_card(
            "📊 Full Report — Excel",
            "Multi-sheet Excel workbook containing entities, relationships, "
            "and graph statistics. Share directly with stakeholders.",
        )
        try:
            xlsx = to_excel_bytes(entities, relations, stats)
            st.download_button(
                "⬇ Download Excel Report",
                data=xlsx,
                file_name="knowledge_graph_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.button(
                "⬇ Download Excel Report",
                disabled=True,
                use_container_width=True,
                help="Excel export requires openpyxl: pip install openpyxl",
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        _export_card(
            "📈 Graph Statistics — JSON",
            "Structured metrics including node count, edge count, density, "
            "component count, and entity/relationship type distributions.",
        )
        st.download_button(
            "⬇ Download Statistics",
            data=json.dumps(stats, indent=2).encode("utf-8"),
            file_name="graph_statistics.json",
            mime="application/json",
            use_container_width=True,
        )


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_file,
    engine: str,
    chunk_size: int,
    chunk_overlap: int,
    dedup_threshold: float,
) -> None:
    """Execute the full extraction pipeline and store results in session state."""
    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            temp_path = tmp.name

        with st.status("Processing your document…", expanded=True) as status:

            st.write("📄 Extracting text from PDF…")
            pages = extract_pages_with_metadata(temp_path)
            st.write(f"   ✓ {len(pages)} page{'s' if len(pages) != 1 else ''} extracted")

            st.write("✂️ Chunking text for AI processing…")
            chunks = chunk_pages(pages, max_chars=chunk_size, overlap=chunk_overlap)
            if not chunks:
                raise ValueError(
                    "No text chunks created — the PDF may contain only images or "
                    "scanned pages without embedded text."
                )
            st.write(f"   ✓ {len(chunks)} chunks created")

            engine_label = "GPT-4 (OpenAI)" if engine == "openai" else "REBEL (Local)"
            st.write(f"🤖 Extracting entities & relationships ({engine_label})…")
            if engine == "openai":
                if not settings.openai_api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is not configured. "
                        "Add it to your .env file or switch to the REBEL engine."
                    )
                extraction = extract_openai(chunks, show_progress=False)
            else:
                extraction = extract_hf(chunks, show_progress=False)

            ec = extraction["entity_count"]
            rc = extraction["relation_count"]
            st.write(f"   ✓ {ec} entities · {rc} relationships found")

            if ec == 0:
                raise ValueError(
                    "No entities were extracted. The document may be too short, "
                    "too technical, or the extraction engine may need tuning."
                )

            st.write("🔍 Deduplicating entities…")
            dedup = deduplicate_entities(
                extraction["entities"],
                extraction["relations"],
                threshold=dedup_threshold,
                show_progress=False,
            )
            st.write(
                f"   ✓ Consolidated to {dedup['final_count']} unique entities "
                f"· {len(dedup['relations'])} relationships"
            )

            st.write("🕸️ Building knowledge graph…")
            graph_json = build_graph_json(dedup["entities"], dedup["relations"])
            stats = get_graph_statistics(graph_json)

            st.write("📊 Computing analytics & insights…")
            analytics = compute_graph_analytics(graph_json)
            insights = generate_insights(analytics)

            status.update(label="✅ Knowledge graph ready!", state="complete")

        # Persist results
        st.session_state.processed = True
        st.session_state.graph_json = graph_json
        st.session_state.entities = dedup["entities"]
        st.session_state.relations = dedup["relations"]
        st.session_state.stats = stats
        st.session_state.analytics = analytics
        st.session_state.insights = insights
        st.session_state.filename = pdf_file.name
        st.session_state.num_pages = len(pages)
        st.session_state.num_chunks = len(chunks)
        st.session_state.entity_count = dedup["final_count"]
        st.session_state.relation_count = len(dedup["relations"])
        st.session_state.processing_error = None

    except Exception as exc:
        st.session_state.processing_error = str(exc)
        raise

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ── Page sections ─────────────────────────────────────────────────────────────

def render_welcome() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:52px 0 36px">
          <div style="font-size:3.2em;font-weight:800;
                      background:linear-gradient(135deg,#1E6FFF 0%,#42A5F5 60%,#AB47BC 100%);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      letter-spacing:-0.025em;line-height:1.1">
            Transform Documents<br>Into Intelligence
          </div>
          <div style="color:#8DA0B3;font-size:1.1em;margin-top:18px;
                      max-width:560px;margin-left:auto;margin-right:auto;line-height:1.6">
            Upload any PDF and watch as AI extracts entities, discovers relationships,
            and builds an interactive knowledge graph — ready to explore, analyze, and export.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:repeat(3,1fr);
                    gap:18px;margin:8px 0 36px">
          <div style="background:linear-gradient(145deg,#101E35,#132035);
                      border:1px solid rgba(30,111,255,0.2);border-radius:16px;
                      padding:28px 22px;text-align:center">
            <div style="font-size:2em;margin-bottom:12px">🤖</div>
            <div style="font-size:1em;font-weight:600;color:#E8EDF5;margin-bottom:8px">
              Dual AI Engines
            </div>
            <div style="color:#7A90A8;font-size:0.85em;line-height:1.55">
              GPT-4 for typed entities with rich context, or the local REBEL
              model for air-gapped, zero-cost processing.
            </div>
          </div>
          <div style="background:linear-gradient(145deg,#101E35,#132035);
                      border:1px solid rgba(171,71,188,0.2);border-radius:16px;
                      padding:28px 22px;text-align:center">
            <div style="font-size:2em;margin-bottom:12px">🕸️</div>
            <div style="font-size:1em;font-weight:600;color:#E8EDF5;margin-bottom:8px">
              Interactive Graph
            </div>
            <div style="color:#7A90A8;font-size:0.85em;line-height:1.55">
              Physics-based visualization with hover tooltips, entity type
              filtering, zoom, and pan controls.
            </div>
          </div>
          <div style="background:linear-gradient(145deg,#101E35,#132035);
                      border:1px solid rgba(102,187,106,0.2);border-radius:16px;
                      padding:28px 22px;text-align:center">
            <div style="font-size:2em;margin-bottom:12px">📊</div>
            <div style="font-size:1em;font-weight:600;color:#E8EDF5;margin-bottom:8px">
              Analytics Dashboard
            </div>
            <div style="color:#7A90A8;font-size:0.85em;line-height:1.55">
              AI-generated insights, entity distribution charts, centrality
              analysis, and multi-format export.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # How it works
    st.markdown(
        """
        <div style="margin:0 0 28px">
          <div style="font-size:1.05em;font-weight:600;color:#8DA0B3;
                      text-align:center;letter-spacing:0.08em;margin-bottom:20px">
            HOW IT WORKS
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px">
            <div style="text-align:center;padding:14px">
              <div style="width:38px;height:38px;background:rgba(30,111,255,0.12);
                          border:2px solid #1E6FFF;border-radius:50%;
                          display:flex;align-items:center;justify-content:center;
                          margin:0 auto 10px;font-weight:700;color:#42A5F5;font-size:0.9em">1</div>
              <div style="color:#E8EDF5;font-size:0.88em;font-weight:600">Upload PDF</div>
              <div style="color:#7A90A8;font-size:0.78em;margin-top:4px;line-height:1.4">
                Any PDF: reports, papers, contracts
              </div>
            </div>
            <div style="text-align:center;padding:14px">
              <div style="width:38px;height:38px;background:rgba(171,71,188,0.12);
                          border:2px solid #AB47BC;border-radius:50%;
                          display:flex;align-items:center;justify-content:center;
                          margin:0 auto 10px;font-weight:700;color:#AB47BC;font-size:0.9em">2</div>
              <div style="color:#E8EDF5;font-size:0.88em;font-weight:600">AI Extraction</div>
              <div style="color:#7A90A8;font-size:0.78em;margin-top:4px;line-height:1.4">
                GPT-4 or REBEL identifies entities & relations
              </div>
            </div>
            <div style="text-align:center;padding:14px">
              <div style="width:38px;height:38px;background:rgba(102,187,106,0.12);
                          border:2px solid #66BB6A;border-radius:50%;
                          display:flex;align-items:center;justify-content:center;
                          margin:0 auto 10px;font-weight:700;color:#66BB6A;font-size:0.9em">3</div>
              <div style="color:#E8EDF5;font-size:0.88em;font-weight:600">Deduplication</div>
              <div style="color:#7A90A8;font-size:0.78em;margin-top:4px;line-height:1.4">
                Embeddings merge similar entities into one
              </div>
            </div>
            <div style="text-align:center;padding:14px">
              <div style="width:38px;height:38px;background:rgba(255,167,38,0.12);
                          border:2px solid #FFA726;border-radius:50%;
                          display:flex;align-items:center;justify-content:center;
                          margin:0 auto 10px;font-weight:700;color:#FFA726;font-size:0.9em">4</div>
              <div style="color:#E8EDF5;font-size:0.88em;font-weight:600">Explore & Export</div>
              <div style="color:#7A90A8;font-size:0.78em;margin-top:4px;line-height:1.4">
                Visualize, analyze, and download in 5 formats
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,rgba(30,111,255,0.07),
                    rgba(171,71,188,0.07));
                    border:1px solid rgba(30,111,255,0.25);border-radius:12px;
                    padding:18px 24px;text-align:center">
          <div style="color:#42A5F5;font-weight:600;margin-bottom:5px;font-size:0.95em">
            👈 &nbsp; Upload a PDF in the sidebar to get started
          </div>
          <div style="color:#7A90A8;font-size:0.82em">
            Supports academic papers, business reports, legal documents, technical
            specifications, and more
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_header() -> None:
    fname = st.session_state.filename or "Document"
    pages = st.session_state.num_pages
    chunks = st.session_state.num_chunks
    ents = st.session_state.entity_count
    rels = st.session_state.relation_count

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;padding:10px 0 16px;
                    border-bottom:1px solid rgba(30,111,255,0.12);margin-bottom:8px">
          <div style="flex:1">
            <span style="font-size:1em;font-weight:700;color:#E8EDF5">📄 {fname}</span>
            <span style="color:#506070;margin:0 8px">·</span>
            <span style="color:#7A90A8;font-size:0.84em">
              {pages} page{'s' if pages != 1 else ''}
              &nbsp;·&nbsp; {chunks} chunks
              &nbsp;·&nbsp;
              <span style="color:#42A5F5">{ents} entities</span>
              &nbsp;·&nbsp;
              <span style="color:#AB47BC">{rels} relationships</span>
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """
    Renders the sidebar and returns pipeline inputs.

    Returns:
        (uploaded_file, engine, chunk_size, chunk_overlap,
         dedup_threshold, process_clicked)
    """
    with st.sidebar:
        # ── Branding ───────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center;padding:12px 0 22px">
              <div style="font-size:1.7em;font-weight:800;
                          background:linear-gradient(135deg,#1E6FFF,#42A5F5);
                          -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                🕸️ KnowledgeX
              </div>
              <div style="color:#3D5A80;font-size:0.7em;margin-top:3px;
                          letter-spacing:0.12em">
                AI KNOWLEDGE GRAPH PLATFORM
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Upload ─────────────────────────────────────────────────────
        st.markdown(
            '<div style="color:#C5D0DE;font-size:0.88em;font-weight:600;'
            'margin-bottom:8px">📂 Upload Document</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "PDF file",
            type=["pdf"],
            label_visibility="collapsed",
            help="Upload a PDF document to extract its knowledge graph",
        )

        if uploaded_file:
            st.markdown(
                f'<div style="background:rgba(0,200,150,0.08);'
                f'border:1px solid rgba(0,200,150,0.28);border-radius:8px;'
                f'padding:9px 12px;margin:6px 0">'
                f'<span style="color:#00C896;font-size:0.82em">✓ {uploaded_file.name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Config ─────────────────────────────────────────────────────
        with st.expander("⚙️ Configuration", expanded=True):
            openai_ok = bool(settings.openai_api_key)
            if not openai_ok:
                st.warning(
                    "OpenAI API key not found. Only the local REBEL engine "
                    "is available. Add `OPENAI_API_KEY` to your `.env` file.",
                    icon="⚠️",
                )

            engine_options = ["openai", "rebel"] if openai_ok else ["rebel"]
            engine = st.selectbox(
                "Extraction Engine",
                options=engine_options,
                format_func=lambda x: (
                    "🤖 GPT-4  (OpenAI)" if x == "openai" else "🦾 REBEL  (Local)"
                ),
                help=(
                    "GPT-4 extracts typed entities (Person, Org, Location…) with "
                    "evidence spans. REBEL runs locally — no API key or cost."
                ),
            )

            chunk_size = st.slider(
                "Chunk Size (chars)",
                min_value=200,
                max_value=2000,
                value=min(settings.chunk_size, 2000),
                step=100,
                help="Characters per text chunk fed to the AI. Larger = richer "
                     "context but higher API cost and latency.",
            )

            max_overlap = min(300, chunk_size - 50)
            default_overlap = min(settings.chunk_overlap, max_overlap)
            chunk_overlap = st.slider(
                "Chunk Overlap (chars)",
                min_value=0,
                max_value=max_overlap,
                value=default_overlap,
                step=25,
                help="Characters shared between adjacent chunks to preserve "
                     "cross-boundary entity context.",
            )

            dedup_threshold = st.slider(
                "Deduplication Threshold",
                min_value=0.5,
                max_value=1.0,
                value=settings.dedup_threshold,
                step=0.05,
                help="Cosine similarity threshold for merging entity variants. "
                     "Higher = stricter (fewer merges). 0.85 is a good default.",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Action button ───────────────────────────────────────────────
        process_clicked = st.button(
            "🚀 Extract Knowledge Graph",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
            help="Upload a PDF first" if uploaded_file is None else "Start extraction",
        )
        if uploaded_file is None:
            st.caption("Upload a PDF document above to enable extraction.")

        # ── Reset ───────────────────────────────────────────────────────
        if st.session_state.get("processed"):
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset", use_container_width=True, type="secondary"):
                reset_session_state()
                st.rerun()

        # ── Footer ──────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="position:fixed;bottom:16px;left:0;right:0;
                        width:var(--sidebar-width,260px);
                        text-align:center;color:#283A52;font-size:0.7em;
                        padding:0 16px">
              KnowledgeX v2.0 &nbsp;·&nbsp; PDF → Knowledge Graph
            </div>
            """,
            unsafe_allow_html=True,
        )

    return uploaded_file, engine, chunk_size, chunk_overlap, dedup_threshold, process_clicked


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    init_session_state()
    inject_css()

    (
        uploaded_file,
        engine,
        chunk_size,
        chunk_overlap,
        dedup_threshold,
        process_clicked,
    ) = render_sidebar()

    # ── Handle extraction trigger ────────────────────────────────────────
    if process_clicked and uploaded_file is not None:
        # Clear any previous error / results before new run
        st.session_state.processing_error = None
        st.session_state.processed = False
        try:
            run_pipeline(uploaded_file, engine, chunk_size, chunk_overlap, dedup_threshold)
        except Exception:
            pass  # error stored in session_state.processing_error
        st.rerun()

    # ── Error state ──────────────────────────────────────────────────────
    if st.session_state.get("processing_error"):
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.error(
            f"**Extraction failed:** {st.session_state.processing_error}",
            icon="❌",
        )
        st.info(
            "**Suggestions:** Check your API key, try a different engine, "
            "increase chunk size, or use a text-based (non-scanned) PDF.",
            icon="💡",
        )
        return

    # ── Results state ────────────────────────────────────────────────────
    if st.session_state.get("processed") and st.session_state.get("graph_json"):
        render_results_header()
        tab_graph, tab_analytics, tab_data, tab_export = st.tabs(
            [
                "🕸️  Knowledge Graph",
                "📊  Analytics",
                "🔎  Data Explorer",
                "📦  Export",
            ]
        )
        with tab_graph:
            render_graph_tab(
                st.session_state.graph_json,
                st.session_state.analytics,
            )
        with tab_analytics:
            render_analytics_tab(
                st.session_state.analytics,
                st.session_state.insights,
                st.session_state.filename,
            )
        with tab_data:
            render_data_tab(
                st.session_state.entities,
                st.session_state.relations,
            )
        with tab_export:
            render_export_tab(
                st.session_state.graph_json,
                st.session_state.entities,
                st.session_state.relations,
                st.session_state.stats,
            )
        return

    # ── Welcome state ────────────────────────────────────────────────────
    render_welcome()


if __name__ == "__main__":
    main()

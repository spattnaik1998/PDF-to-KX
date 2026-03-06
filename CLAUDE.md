# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env with your API keys
```

### Run API Server
```bash
uvicorn app.api:app --reload
# API docs available at http://localhost:8000/docs
```

### Run Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

### Test via curl
```bash
curl -X POST http://localhost:8000/pdf-to-kg \
    -F "file=@document.pdf" \
    -F "engine=openai"
```

## New Modules (v2.0)

- **`app/analytics.py`** — `compute_graph_analytics(graph_json)` builds a NetworkX graph and returns degree stats, entity/relation type distributions, centrality rankings, and component count. `generate_insights(analytics)` returns markdown bullet strings.
- **`app/export_utils.py`** — `to_json_string`, `to_entities_csv`, `to_relations_csv`, `to_excel_bytes` (requires openpyxl), `to_graphml_string`.
- **`.streamlit/config.toml`** — Dark corporate theme (`#0A1628` bg, `#1E6FFF` primary).
- **`ui/streamlit_app.py`** — Complete professional UI (replaces the stub). See UI Architecture below.

## Architecture

This is a PDF-to-Knowledge-Graph pipeline with two separate frontends (FastAPI REST API + Streamlit UI) sharing a common backend in `app/`.

### Pipeline Flow
```
PDF Upload → Text Extraction → Chunking → Entity/Relation Extraction → Deduplication → Graph Construction → JSON/Visualization
```

### Backend Modules (`app/`)

- **`config.py`**: Pydantic `Settings` class loads all env vars. Global `settings` singleton used throughout. Constants `DEFAULT_MODEL_OPENAI` (`gpt-4-turbo-preview`) and `DEFAULT_MODEL_HF_REBEL` (`Babelscape/rebel-large`) defined here.

- **`pdf_utils.py`**: PDF text extraction via PyMuPDF. Key functions: `extract_text_from_pdf`, `extract_pages_with_metadata` (returns list of `{page_number, text}` dicts), `validate_pdf`.

- **`chunking.py`**: Sliding window chunking. `chunk_pages()` is the primary function used by the API (takes page-metadata dicts, returns chunks with `chunk_id`, `page_number`, `text`). `simple_chunk()` is the underlying implementation. Each chunk is chunked per page independently.

- **`extraction_openai.py`**: Uses OpenAI's structured output (`client.beta.chat.completions.parse`) with Pydantic models (`Entity`, `Relation`, `KnowledgeGraph`) to guarantee JSON schema compliance. `extract_from_chunks()` makes entity IDs globally unique by prefixing `chunk{id}_`. Includes exponential backoff retry logic.

- **`extraction_hf.py`**: Alternative extraction using the HuggingFace REBEL model (`Babelscape/rebel-large`) — runs locally, no API key needed.

- **`deduplication.py`**: Uses OpenAI `text-embedding-3-large` to embed entity labels, then `AgglomerativeClustering` (cosine distance) to cluster near-duplicates. `find_canonical_entity()` picks the longest label as the canonical form. Updates all relation source/target IDs to canonical IDs, then deduplicates relations by `(source, target, type)` key.

- **`graph_builder.py`**: Converts entity/relation lists to `{nodes, edges}` JSON. `to_networkx()` converts to `nx.DiGraph`. Also supports Cytoscape.js export format and node type filtering. Self-loops are dropped by default.

- **`api.py`**: FastAPI app with CORS enabled. Main endpoint `POST /pdf-to-kg` runs the full 5-step pipeline. Also exposes `POST /extract-text` (text only), `GET /config`, `GET /health`, `GET /supported-engines`.

### Frontend (`ui/`)

- **`streamlit_app.py`**: Fully implemented professional UI. Layout: sidebar (upload + config) + main area with 4 tabs: **Knowledge Graph** (pyvis physics graph), **Analytics** (plotly charts + AI insights), **Data Explorer** (searchable entity/relation DataFrames), **Export Hub** (JSON, CSV, Excel, GraphML). Uses `st.session_state` to persist results across reruns. Pipeline runs in main area via `st.status()` for live progress. Brand name: **KnowledgeX**.

#### UI State Machine
```
Welcome page → [upload + click Extract] → run_pipeline() → st.rerun()
→ Results (4 tabs)  ← reset button → Welcome page
```

#### UI Internals
- CSS injected via `st.markdown` — targets `[data-testid="stSidebar"]`, tab selectors, button variants, file uploader
- `build_pyvis_html()` writes a temp HTML file, reads it back, then deletes it — use `cdn_resources='in_line'` so the graph works in Streamlit's iframe
- Node size = `max(14, min(55, 12 + degree * 4))`; node color from `TYPE_COLORS` dict keyed by entity type
- Charts use `template="plotly_dark"` with `paper_bgcolor="rgba(0,0,0,0)"` for transparent backgrounds

### Key Design Decisions

- **Dual extraction engines**: OpenAI (cloud, structured output) vs REBEL (local, no API cost). Selected per-request via `engine` form param.
- **Entity ID scoping**: Raw entity IDs from the LLM are local to each chunk. They're made globally unique with `chunk{id}_` prefix before deduplication maps them to canonical IDs.
- **Deduplication uses OpenAI embeddings** (not sentence-transformers despite being in requirements). The `.env.example` references `EMBEDDING_MODEL` for sentence-transformers but `deduplication.py` hardcodes `text-embedding-3-large`.
- **Temporary files**: Uploaded PDFs are written to OS temp dir and deleted in `finally` blocks.

## Configuration

All configuration via `.env`. Key variables:
- `OPENAI_API_KEY` — required for `openai` engine and deduplication
- `EXTRACTION_ENGINE` — `"openai"` (default) or `"rebel"`
- `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50` — character-based chunking
- `DEDUP_THRESHOLD=0.85` — cosine similarity threshold for entity merging
- `HF_TOKEN` — optional, for gated HuggingFace models

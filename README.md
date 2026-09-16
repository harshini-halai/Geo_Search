# AERO-SENTINEL // Geo-Semantic Intelligence Platform

An offline-first, full-stack geospatial surveillance and semantic retrieval platform. The system ingests raw satellite GeoTIFF imagery, generates latent vision embeddings using OpenCLIP, indexes spatial patches into Qdrant vector storage, and provides a decoupled MVC web operations console for real-time semantic querying, cluster telemetry, and human-in-the-loop (HITL) anomaly verification.

---

## Key Highlights & Core Features

- **Automated GeoTIFF Ingestion & Tiling Pipeline:** Dynamically slices high-resolution satellite imagery into standardized 256x256 tiles with configurable overlap and extracts multi-dimensional embeddings.
- **Dual-Store Synchronization:**
    _**Qdrant (Vector Store):** 512-dimensional vector indexing using HNSW for sub-second semantic and cross-modal cosine similarity search.
    _ **SQLite / `aiosqlite` (Audit Ledger):** Write-Ahead Logging (WAL) persistent relational store tracking metadata, unsupervised clusters, anomaly scores, and analyst triage statuses.
- **Cross-Modal Semantic Search:** Query satellite imagery using natural language prompts (e.g., `"dense canopy forest"`, `"heavy excavation"`, `"river bends and water"`) via cross-attention ViT embeddings.
- **Unsupervised Geospatial Intelligence:** KMeans spatial clustering alongside Isolation Forest anomaly detection for automated land-cover categorization and surface change monitoring.
- **Production Decoupled MVC Architecture:** Clean separation of backend FastAPI endpoints and an offline-capable frontend console utilizing Tailwind CSS, Chart.js telemetry, and modular client scripts.
- **Human-in-the-Loop (HITL) Triage Console:** Analysts can review suspicious patches, inspect full-resolution imagery in an optical HUD crosshair modal, and sign off with verified or dismissed tags.

---

## Architectural Layout

```text
geo-backend/
├── app/
│   ├── api/                  # REST endpoints (ingest, search, change, analyst, ml)
│   ├── controllers/          # Web controller orchestration & template routing
│   ├── core/                 # App configuration & environment settings
│   ├── ml/                   # OpenCLIP embedder & change detection modules
│   ├── models/               # SQLite schemas & aiosqlite audit ledger methods
│   └── services/             # Qdrant client store & dynamic GeoTIFF tiler
├── data/                     # Persistent local SQLite & vector stores
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css     # Custom HUD styling & animations
│   │   └── js/
│   │       ├── chart.js      # Chart.js vector cluster telemetry
│   │       ├── modal.js      # Optical patch inspection HUD logic
│   │       └── search.js     # Semantic retrieval UI client
│   └── templates/
│       └── dashboard.html    # Operations surveillance console
├── storage/
│   ├── raw_geotiff/          # Source GeoTIFF imagery
│   └── tiles/                # Extracted 256x256 visual tiles
├── main.py                   # Application entrypoint & static mount registry
└── requirements.txt

```

## System Architecture & Workflow

### 1. Ingestion & Embedding Pipeline
- Splits large raster imagery into standard 256x256 tiles under `/storage/tiles/`.
- Computes 512-D spatial/visual embeddings and indexes them in a local Qdrant collection (`satellite_tiles`).

### 2. Anomaly Detection & Clustering
- **Isolation Forest**: Evaluates feature vectors to flag structural and land-use outliers against a defined anomaly threshold (e.g., 0.550).
- **K-Means (k=3)**: Groups indexed tiles into distinct semantic land-use categories (e.g., Forest, Water/Estuary, Urban/Industrial).

### 3. Human-in-the-Loop (HITL) Audit System
- Fast, local persistence using SQLite (`geo_system.db`) running in WAL mode for concurrent writes.
- Web dashboard allows analysts to visually triage flagged tiles, view tactical HUD overlays, and persist review status (`pending`, `verified`, `false_positive`).

### Quick Start (Local Setup)
1. Install dependencies:
   
   pip install -r requirements.txt

## Data Layer & Pipeline

### 1. Raster Ingestion & Synthetic Imagery
- **Image Specifications**: Synthetic multispectral optical satellite tiles generated at standard $256 \times 256$ resolution across RGB channels.
- **Storage Path**: Standardized tile storage localized under `storage/tiles/<tile_id>.png`.
- **Target Land Classes**:
  - `Dense Forest Canopy` (Baseline vegetation canopy)
  - `River Delta & Estuary` (Hydrological surface water)
  - `Industrial Logistics Zone` (Commercial infrastructure)
  - `Rapid Deforestation Scar` (High-priority forest loss anomaly)
  - `High-Density Residential` (Urban settlement)
  - `Offshore Harbor Vessel Lane` (Maritime transit corridor)

### 2. Relational Audit Metadata (SQLite)
- **Database Path**: `data/geo_system.db`
- **Concurrency Mode**: Configured with `PRAGMA journal_mode=WAL` (Write-Ahead Logging) for non-blocking concurrent reads during analyst triage actions.
- **Schema (`tile_audit`)**:
  - `id` (TEXT PRIMARY KEY): Unique identifier (e.g., `tile_alpha_01`).
  - `image_path` (TEXT): Disk path to the tile image.
  - `primary_tag` (TEXT): Ground truth or classified land-use description.
  - `cluster_id` (INTEGER): Unsupervised grouping cluster assignment ($k=0, 1, 2$).
  - `anomaly_score` (REAL): Outlier score derived via Isolation Forest ($0.00$ to $1.00$).
  - `status` (TEXT): HITL verification stage (`pending`, `verified`, `dismissed`).

### 3. Vector Database (Qdrant)
- **Collection Name**: `satellite_tiles`
- **Vector Dimension**: 512-D normalized embeddings representing spatial-visual representations.
- **Distance Metric**: `Cosine` similarity.
- **Payload Schema**:
  ```json
  {
    "tile_id": "tile_alpha_01",
    "image_path": "/storage/tiles/tile_alpha_01.png",
    "tag": "Dense Forest Canopy",
    "date": "2026-03-01",
    "sensor": "Sentinel-2",
    "bbox": {"xmin": 0, "ymin": 0, "xmax": 256, "ymax": 256}
  }

### 4. Data Seeding Workflow
To reset or generate a fresh batch of synthetic tiles, audit entries, and Qdrant index embeddings, run:

python sample_data.py
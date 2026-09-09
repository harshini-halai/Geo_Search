# Geo-Semantic Backend (Offline-First)

FastAPI backend for offline semantic search and change detection over satellite GeoTIFF imagery.

## What this project does

- Ingests GeoTIFF images and indexes image tiles.
- Supports semantic text queries and image-based search.
- Detects multi-temporal change candidates between two image sources.
- Maintains an analyst review queue for reviewed change results.

## Local setup

```bash
# Clone repository
git clone https://github.com/harshini-halai/Geo_Search.git
cd Geo_Search

# Create and activate a virtual environment
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Useful URLs

- API docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

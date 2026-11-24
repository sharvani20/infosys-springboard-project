#!/usr/bin/env bash
# Render build step for the backend (Python web service).
set -euo pipefail

pip install --upgrade pip
pip install -r requirements.txt

# Pre-build the search index (and download the small ONNX embedding model)
# at build time, so the running service starts instantly and uses little RAM.
# If this step fails, the API still starts: it falls back to keyword search
# and builds the index itself in the background.
python -m backend.rag.build_index || echo "WARNING: index pre-build failed; the API will build it on startup."

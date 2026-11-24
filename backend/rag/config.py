import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load local .env first so RETRIEVAL_MODE etc. work in development too.
# (On Render the values come from real environment variables.)
load_dotenv(PROJECT_ROOT / ".env")

UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"

# Generated search index (chunks.json + embeddings.npy). Built at deploy time
# by `python -m backend.rag.build_index` and re-built automatically if missing.
DATA_DIR = Path(
    os.getenv("INDEX_DIR", str(PROJECT_ROOT / "backend" / "data"))
)

# Where the small ONNX embedding model is cached. Kept inside the project so a
# model downloaded during the Render build is still there at runtime.
MODEL_CACHE_DIR = Path(
    os.getenv(
        "MODEL_CACHE_DIR",
        str(PROJECT_ROOT / "backend" / ".model_cache")
    )
)

# Same model family as before, but run through ONNX (no PyTorch).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# "hybrid"  -> semantic (embeddings) + keyword search   (default)
# "keyword" -> keyword search only; never loads an embedding model.
#              Use this as a safety valve if memory is still tight.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 6

# Extra keyword weight for phrases that matter a lot for policy questions.
BOOST_PHRASES = {
    "pre-existing disease": 5,
    "36 months of continuous coverage": 5,
}


def semantic_enabled():
    return RETRIEVAL_MODE != "keyword"


for _directory in (UPLOAD_DIR, DATA_DIR):
    try:
        _directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

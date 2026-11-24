import threading

import numpy as np

from backend.rag.config import EMBEDDING_MODEL, MODEL_CACHE_DIR


_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the ONNX embedding model on first use (not at import/startup)."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                # Imported lazily so keyword-only mode never needs it.
                from fastembed import TextEmbedding

                print("Loading embedding model...")
                _model = TextEmbedding(
                    model_name=EMBEDDING_MODEL,
                    cache_dir=str(MODEL_CACHE_DIR),
                    threads=1
                )

    return _model


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def generate_embeddings(texts, batch_size=16):
    """Return an (n, dim) float32 array of L2-normalised embeddings."""

    if not texts:
        return np.zeros((0, 0), dtype="float32")

    model = get_model()

    vectors = list(model.embed(list(texts), batch_size=batch_size))

    return _normalize(np.asarray(vectors, dtype="float32"))


def generate_query_embedding(query):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    return generate_embeddings([query.strip()])[0].tolist()

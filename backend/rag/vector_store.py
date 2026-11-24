"""
A tiny file-backed vector store.

The knowledge base is one policy PDF (~100 chunks), so a ChromaDB server-style
database is unnecessary. Chunks live in `chunks.json` and their L2-normalised
embeddings in `embeddings.npy`; cosine search is one numpy matrix product.
Memory use is a few hundred KB.
"""

import json
import os
import threading

import numpy as np

from backend.rag.config import DATA_DIR, TOP_K


CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

_lock = threading.RLock()
_state = None   # {"chunks": [dict], "emb": ndarray | None}


# ---------- persistence ----------

def _load():
    global _state

    with _lock:
        if _state is not None:
            return _state

        chunks = []
        embeddings = None

        if CHUNKS_FILE.exists():
            try:
                chunks = json.loads(
                    CHUNKS_FILE.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                chunks = []

        if chunks and EMBEDDINGS_FILE.exists():
            try:
                loaded = np.load(EMBEDDINGS_FILE)

                if loaded.ndim == 2 and loaded.shape[0] == len(chunks):
                    embeddings = loaded.astype("float32")
            except (OSError, ValueError):
                embeddings = None

        _state = {"chunks": chunks, "emb": embeddings}
        return _state


def _atomic_write(path, writer, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    with open(temp_path, mode) as handle:
        writer(handle)

    os.replace(temp_path, path)


def _save(state):
    try:
        _atomic_write(
            CHUNKS_FILE,
            lambda f: f.write(
                json.dumps(state["chunks"], ensure_ascii=False)
                .encode("utf-8")
            ),
            "wb"
        )

        if state["emb"] is not None and len(state["chunks"]) > 0:
            _atomic_write(
                EMBEDDINGS_FILE,
                lambda f: np.save(f, state["emb"]),
                "wb"
            )
        elif EMBEDDINGS_FILE.exists():
            EMBEDDINGS_FILE.unlink()

    except OSError as error:
        # Read-only or full disk: keep working from memory.
        print("Could not persist search index:", error)


def _normalize(matrix):
    matrix = np.asarray(matrix, dtype="float32")

    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2-D array.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return matrix / norms


# ---------- public API ----------

def count():
    with _lock:
        return len(_load()["chunks"])


def has_embeddings():
    with _lock:
        state = _load()
        return state["emb"] is not None and len(state["chunks"]) > 0


def get_all_documents():
    """Every stored chunk (text + metadata), used for keyword search."""
    with _lock:
        return [dict(chunk) for chunk in _load()["chunks"]]


def delete_documents_by_source(source):
    if not source:
        return 0

    with _lock:
        state = _load()

        keep = [
            chunk["source"] != str(source)
            for chunk in state["chunks"]
        ]
        removed = keep.count(False)

        if removed == 0:
            return 0

        if state["emb"] is not None:
            state["emb"] = state["emb"][np.array(keep, dtype=bool)]

        state["chunks"] = [
            chunk for chunk, flag in zip(state["chunks"], keep) if flag
        ]

        if not state["chunks"]:
            state["emb"] = None

        _save(state)
        return removed


def add_documents(chunks, embeddings=None):
    """
    Add chunks. `embeddings` may be None to add keyword-searchable chunks
    only; semantic search then stays off until `set_embeddings` is called.
    """

    if embeddings is not None and len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match.")

    if not chunks:
        return 0

    new_chunks = [
        {
            "text": chunk["text"],
            "source": str(chunk["source"]),
            "page": int(chunk["page"]),
            "chunk_index": int(chunk["chunk_index"])
        }
        for chunk in chunks
    ]

    with _lock:
        state = _load()

        can_keep_embeddings = (
            embeddings is not None
            and (state["emb"] is not None or not state["chunks"])
        )

        if can_keep_embeddings:
            new_matrix = _normalize(embeddings)

            if state["emb"] is None:
                combined = new_matrix
            else:
                combined = np.vstack([state["emb"], new_matrix])
        else:
            combined = None

        state["chunks"] = state["chunks"] + new_chunks
        state["emb"] = combined

        _save(state)

    return len(new_chunks)


def set_embeddings(embeddings):
    """Attach embeddings for all stored chunks (row i <-> chunk i)."""

    with _lock:
        state = _load()
        matrix = _normalize(embeddings)

        if matrix.shape[0] != len(state["chunks"]):
            return False   # chunks changed while embedding; caller may retry

        state["emb"] = matrix
        _save(state)
        return True


def search_documents(query_embedding, top_k=TOP_K):
    with _lock:
        state = _load()
        embeddings = state["emb"]
        chunks = state["chunks"]

        if embeddings is None or not chunks:
            return []

        query = np.asarray(query_embedding, dtype="float32").ravel()

        if query.shape[0] != embeddings.shape[1]:
            raise ValueError("Query embedding has the wrong dimension.")

        norm = np.linalg.norm(query)

        if norm > 0:
            query = query / norm

        similarity = embeddings @ query
        best = np.argsort(-similarity)[:min(top_k, len(chunks))]

        evidence = []

        for index in best:
            chunk = chunks[int(index)]

            evidence.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "distance": float(1.0 - similarity[int(index)])
            })

        return evidence

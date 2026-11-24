import pytest

from backend.rag import vector_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the vector store at an empty temp folder for each test."""
    monkeypatch.setattr(vector_store, "CHUNKS_FILE", tmp_path / "chunks.json")
    monkeypatch.setattr(
        vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy"
    )
    monkeypatch.setattr(vector_store, "_state", None)
    return vector_store

import pytest

from backend.rag import retriever


def test_retrieve_evidence_combines_semantic_and_keyword(monkeypatch):
    semantic = {
        "text": "Applicants must be at least 21 years old.",
        "source": "sample_banking.txt",
        "page": 1,
        "chunk_index": 0,
        "distance": 0.1
    }
    keyword_only = {
        "text": "The waiting period is 36 months of continuous coverage.",
        "source": "sample_banking.txt",
        "page": 2,
        "chunk_index": 0
    }

    monkeypatch.setattr(retriever, "semantic_enabled", lambda: True)
    monkeypatch.setattr(retriever, "has_embeddings", lambda: True)
    monkeypatch.setattr(
        retriever, "generate_query_embedding", lambda q: [0.1, 0.2, 0.3]
    )
    monkeypatch.setattr(
        retriever, "search_documents", lambda emb, top_k=3: [semantic]
    )
    monkeypatch.setattr(
        retriever, "get_all_documents", lambda: [semantic, keyword_only]
    )

    results = retriever.retrieve_evidence(
        "What is the waiting period for coverage?",
        top_k=3
    )

    assert len(results) == 2
    # keyword hit with the boosted phrase ranks first
    assert "36 months" in results[0]["text"]
    assert any("21 years old" in r["text"] for r in results)


def test_keyword_mode_never_touches_embeddings(monkeypatch):
    chunk = {
        "text": "Cancer is covered after the waiting period.",
        "source": "p.txt",
        "page": 1,
        "chunk_index": 0
    }

    def boom(*args, **kwargs):
        raise AssertionError("embedding model must not be used")

    monkeypatch.setattr(retriever, "semantic_enabled", lambda: False)
    monkeypatch.setattr(retriever, "generate_query_embedding", boom)
    monkeypatch.setattr(retriever, "search_documents", boom)
    monkeypatch.setattr(retriever, "get_all_documents", lambda: [chunk])

    results = retriever.retrieve_evidence("Is cancer covered?")

    assert len(results) == 1
    assert results[0]["keyword_score"] > 0


def test_semantic_failure_falls_back_to_keywords(monkeypatch):
    chunk = {
        "text": "Cashless treatment is available at network hospitals.",
        "source": "p.txt",
        "page": 3,
        "chunk_index": 1
    }

    def broken(query):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(retriever, "semantic_enabled", lambda: True)
    monkeypatch.setattr(retriever, "has_embeddings", lambda: True)
    monkeypatch.setattr(retriever, "generate_query_embedding", broken)
    monkeypatch.setattr(retriever, "get_all_documents", lambda: [chunk])

    results = retriever.retrieve_evidence("cashless hospitals")

    assert len(results) == 1


def test_retrieve_empty_query():
    with pytest.raises(ValueError):
        retriever.retrieve_evidence("")


def test_retrieve_whitespace_query():
    with pytest.raises(ValueError):
        retriever.retrieve_evidence("   ")

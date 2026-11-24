def _chunk(text, source="sample.txt", page=1, index=0):
    return {
        "text": text,
        "source": source,
        "page": page,
        "chunk_index": index
    }


def _vec(position, dim=384):
    vector = [0.0] * dim
    vector[position] = 1.0
    return vector


def test_add_and_search_documents(store):
    chunks = [
        _chunk("Applicants must be at least 21 years old.", index=0),
        _chunk("Loans are repaid monthly.", index=1),
    ]

    added = store.add_documents(chunks, [_vec(0), _vec(1)])

    assert added == 2
    assert store.count() == 2
    assert store.has_embeddings()

    results = store.search_documents(_vec(0), top_k=2)

    assert results[0]["text"].startswith("Applicants must")
    assert results[0]["page"] == 1
    assert results[0]["distance"] < results[1]["distance"]


def test_add_empty_documents(store):
    assert store.add_documents([], []) == 0


def test_mismatched_chunks_and_embeddings(store):
    try:
        store.add_documents([_chunk("Sample text")], [])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched inputs")


def test_delete_by_source_keeps_other_documents(store):
    store.add_documents(
        [_chunk("alpha", source="a.txt"), _chunk("beta", source="b.txt")],
        [_vec(0), _vec(1)]
    )

    assert store.delete_documents_by_source("a.txt") == 1
    assert store.count() == 1

    results = store.search_documents(_vec(1), top_k=5)

    assert [r["source"] for r in results] == ["b.txt"]


def test_index_persists_to_disk(store, monkeypatch):
    store.add_documents([_chunk("persisted text")], [_vec(2)])

    # simulate a restart: forget in-memory state, reload from files
    monkeypatch.setattr(store, "_state", None)

    assert store.count() == 1
    assert store.has_embeddings()
    assert store.search_documents(_vec(2))[0]["text"] == "persisted text"


def test_keyword_only_chunks_then_embeddings_attached(store):
    store.add_documents([_chunk("no vectors yet")], None)

    assert store.count() == 1
    assert not store.has_embeddings()
    assert store.search_documents(_vec(0)) == []

    assert store.set_embeddings([_vec(0)]) is True
    assert store.has_embeddings()


def test_set_embeddings_rejects_wrong_row_count(store):
    store.add_documents([_chunk("one")], None)

    assert store.set_embeddings([_vec(0), _vec(1)]) is False

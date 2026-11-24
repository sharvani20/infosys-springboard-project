import re

from backend.rag.config import TOP_K, BOOST_PHRASES, semantic_enabled
from backend.rag.embeddings import generate_query_embedding
from backend.rag.vector_store import (
    search_documents,
    get_all_documents,
    has_embeddings
)


STOPWORDS = {
    "what", "is", "the", "for", "under",
    "a", "an", "of", "in", "to", "and"
}


def retrieve_evidence(query, top_k=TOP_K):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    clean_query = query.strip()

    # 1. Semantic search (skipped in keyword mode or if no embeddings yet)
    semantic_results = []

    if semantic_enabled() and has_embeddings():
        try:
            query_embedding = generate_query_embedding(clean_query)

            semantic_results = search_documents(
                query_embedding,
                top_k=max(top_k, 30)
            )

        except Exception as error:
            # Never let a missing/failed embedding model take the API down:
            # fall back to keyword search for this request.
            print(
                "Semantic search unavailable, using keywords only:",
                error
            )

    # 2. Keyword-based search across stored chunks
    query_words = set(
        re.findall(r"[a-z0-9]+", clean_query.lower())
    ) - STOPWORDS

    keyword_results = []

    for chunk in get_all_documents():

        text_lower = chunk["text"].lower()

        score = sum(
            1 for word in query_words
            if word in text_lower
        )

        # Give extra weight to important policy phrases
        for phrase, weight in BOOST_PHRASES.items():
            if phrase in text_lower:
                score += weight

        if score > 0:
            keyword_results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "distance": None,
                "keyword_score": score
            })

    # 3. Combine semantic and keyword results
    combined = {}

    for item in semantic_results:
        key = (
            item["source"],
            item["page"],
            item["chunk_index"]
        )
        combined[key] = item

    for item in keyword_results:
        key = (
            item["source"],
            item["page"],
            item["chunk_index"]
        )

        if key in combined:
            combined[key]["keyword_score"] = item["keyword_score"]
        else:
            combined[key] = item

    # 4. Rank keyword matches first, then semantic matches
    results = list(combined.values())

    results.sort(
        key=lambda item: (
            item.get("keyword_score", 0),
            -(item.get("distance") or 0)
        ),
        reverse=True
    )

    return results[:max(top_k, 30)]

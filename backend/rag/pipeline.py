import threading
from pathlib import Path
from uuid import uuid4

from backend.rag.config import TOP_K, semantic_enabled
from backend.rag.document_loader import load_document
from backend.rag.chunker import chunk_document
from backend.rag.embeddings import generate_embeddings
from backend.rag.vector_store import (
    add_documents,
    delete_documents_by_source,
    count,
    has_embeddings,
    get_all_documents,
    set_embeddings
)
from backend.rag.retriever import retrieve_evidence


def ingest_document(file_path, with_embeddings=None):
    """
    Load -> chunk -> (embed) -> store.

    with_embeddings=None means "follow RETRIEVAL_MODE".
    """

    if with_embeddings is None:
        with_embeddings = semantic_enabled()

    # Step 1: Load document
    document = load_document(file_path)

    if not document["text"].strip():
        return {
            "document_id": None,
            "chunks_added": 0,
            "source": document["source"],
            "message": "No readable text found."
        }

    # Step 2: Generate document ID
    document_id = str(uuid4())

    # Step 3: Create chunks
    chunks = chunk_document(document)

    if not chunks:
        return {
            "document_id": None,
            "chunks_added": 0,
            "source": document["source"],
            "message": "No chunks were generated."
        }

    # Step 4: Embed BEFORE deleting old chunks (so a failure keeps old data)
    embeddings = None

    if with_embeddings:
        embeddings = generate_embeddings(
            [chunk["text"] for chunk in chunks]
        )

    # Step 5: Replace old chunks of this document only
    delete_documents_by_source(document["source"])

    # Step 6: Store new chunks
    chunks_added = add_documents(chunks, embeddings)

    return {
        "document_id": document_id,
        "chunks_added": chunks_added,
        "source": document["source"],
        "message": "Document successfully indexed."
    }


def _embed_in_background():
    try:
        chunks = get_all_documents()
        embeddings = generate_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        if set_embeddings(embeddings):
            print("Embeddings ready for", len(chunks), "chunks.")
        else:
            print("Index changed during embedding; skipped.")

    except Exception as error:
        print(
            "Embedding failed; continuing with keyword search:",
            error
        )


def ensure_knowledge_base(pdf_path):
    """
    Called once at API startup. Must be fast and memory-safe:

    * normal case: the index was pre-built during the deploy -> nothing to do
    * fallback: index missing -> chunk the PDF right now (keyword search works
      immediately) and compute embeddings in a background thread.
    """

    pdf_path = Path(pdf_path)

    if count() == 0:
        if not pdf_path.exists():
            print("Policy PDF not found:", pdf_path)
            return

        print("Search index missing - building from", pdf_path.name)

        result = ingest_document(pdf_path, with_embeddings=False)

        print("Policy ingestion completed:", result)

    else:
        print("Knowledge base ready. Chunks:", count())

    if semantic_enabled() and count() > 0 and not has_embeddings():
        threading.Thread(
            target=_embed_in_background,
            daemon=True
        ).start()


def answer_with_evidence(question, top_k=TOP_K):
    evidence = retrieve_evidence(
        query=question,
        top_k=top_k
    )

    return {
        "question": question,
        "evidence": evidence,
        "evidence_count": len(evidence)
    }

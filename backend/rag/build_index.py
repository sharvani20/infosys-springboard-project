"""
Pre-build the search index. Run during the deploy (see build.sh):

    python -m backend.rag.build_index

Doing this at build time means the running web service never has to load an
embedding model just to start up.
"""

from backend.rag.config import UPLOAD_DIR, semantic_enabled
from backend.rag.pipeline import ingest_document
from backend.rag.vector_store import count, has_embeddings


POLICY_PDF = UPLOAD_DIR / "hdfc_policy.pdf"


def main():
    use_embeddings = semantic_enabled()

    print(
        "Building index from", POLICY_PDF.name,
        "(semantic embeddings: %s)" % ("on" if use_embeddings else "off")
    )

    result = ingest_document(POLICY_PDF, with_embeddings=use_embeddings)

    print(result)
    print(
        "Chunks stored:", count(),
        "| embeddings present:", has_embeddings()
    )


if __name__ == "__main__":
    main()

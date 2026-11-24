"""Print the stored chunks of one PDF page (default: page 11).

Usage:  python inspect_chunks.py [page_number]
"""

import sys

from backend.rag.vector_store import count, get_all_documents

page = int(sys.argv[1]) if len(sys.argv) > 1 else 11

print("\nTotal stored chunks:", count())

chunks = sorted(
    (c for c in get_all_documents() if c["page"] == page),
    key=lambda c: c["chunk_index"]
)

print(f"\nChunks found on PDF page {page}:", len(chunks))

for chunk in chunks:
    print("\n" + "=" * 70)
    print("CHUNK INDEX:", chunk["chunk_index"])
    print("=" * 70)
    print(chunk["text"])

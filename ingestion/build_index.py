from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.chunking import chunk_pdf, load_pdfs
from ingestion.vectorstore import COLLECTION_NAME, chroma_dir, get_embedder, open_collection, raw_docs_dir


def build_index(
    source_dir: Path,
    persist_dir: Path,
    collection_name: str = COLLECTION_NAME,
    rebuild: bool = False,
    batch_size: int = 64,
    chunk_size: int = 900,
    overlap: int = 150,
) -> int:
    """Chunk every PDF in source_dir, embed the chunks and upsert them. Returns the collection size."""
    source_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = load_pdfs(source_dir)
    if not pdf_paths:
        print(f"No PDFs found in {source_dir}. Add files or run `python -m scripts.generate_pdfs`.")
        return 0

    all_chunks: List[Dict] = []
    for p in tqdm(pdf_paths, desc="PDFs"):
        all_chunks.extend(chunk_pdf(p, chunk_size=chunk_size, overlap=overlap))
    if not all_chunks:
        print("No text chunks extracted; check your PDFs.")
        return 0

    embedder = get_embedder()
    print(f"Embedding {len(all_chunks)} chunks from {len(pdf_paths)} PDFs with {embedder.name}")
    collection = open_collection(embedder, persist_dir, collection_name, rebuild=rebuild)

    for start in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding + writing"):
        batch = all_chunks[start : start + batch_size]
        docs = [c["text"] for c in batch]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=docs,
            metadatas=[c["metadata"] for c in batch],
            embeddings=embedder.embed(docs),
        )

    count = collection.count()
    print(f"Done. Collection '{collection_name}' now has {count} records.")
    return count


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build the Chroma index from PDFs.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate the collection before ingesting")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Chroma collection name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/write batch size")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size (characters)")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap (characters)")
    args = parser.parse_args()

    build_index(
        source_dir=raw_docs_dir(),
        persist_dir=chroma_dir(),
        collection_name=args.collection,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

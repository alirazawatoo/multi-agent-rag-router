from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
import sys

# Robust import so this file works both as a module and as a script
try:  # If executed with `python -m ingestion.build_index`
    if __package__:
        from .chunking import load_pdfs, chunk_pdf  # type: ignore
    else:
        raise ImportError
except Exception:
    # If executed as `python ingestion/build_index.py`, ensure project root is on sys.path
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.append(str(_root))
    from ingestion.chunking import load_pdfs, chunk_pdf  # type: ignore


DEFAULT_COLLECTION = "pwb_docs"


def get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(
    source_dir: Path,
    persist_dir: Path,
    model_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    rebuild: bool = False,
    batch_size: int = 64,
    chunk_size: int = 900,
    overlap: int = 150,
):
    source_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = load_pdfs(source_dir)
    if not pdf_paths:
        print(f"No PDFs found in {source_dir}. Add files or run scripts/generate_pdfs.py")
        return

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Chunking {len(pdf_paths)} PDFs ...")
    all_chunks: List[Dict] = []
    for p in tqdm(pdf_paths, desc="PDFs"):
        chunks = chunk_pdf(p, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No text chunks extracted; check your PDFs.")
        return

    print(f"Total chunks: {len(all_chunks)}")

    print(f"Initializing Chroma persistent client at {persist_dir}")
    client = chromadb.PersistentClient(path=str(persist_dir))

    if rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
        collection = client.create_collection(collection_name)
    else:
        # get_or_create ensures idempotent behavior across runs
        collection = client.get_or_create_collection(collection_name)

    print(f"Indexing into collection: {collection_name}")

    # Prepare docs for embedding
    docs = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    metas = [c["metadata"] for c in all_chunks]

    # Embed and write in batches
    total = len(docs)
    pbar = tqdm(total=total, desc="Embedding + Writing")

    for idxs in (range(i, min(i + batch_size, total)) for i in range(0, total, batch_size)):
        idxs = list(idxs)
        batch_docs = [docs[i] for i in idxs]
        batch_ids = [ids[i] for i in idxs]
        batch_metas = [metas[i] for i in idxs]

        embeds = model.encode(batch_docs, normalize_embeddings=True).tolist()

        # Prefer upsert if available; fallback to add/update combo
        if hasattr(collection, "upsert"):
            collection.upsert(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
                embeddings=embeds,
            )
        else:
            try:
                collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas,
                    embeddings=embeds,
                )
            except Exception:
                collection.update(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas,
                    embeddings=embeds,
                )

        pbar.update(len(batch_docs))

    pbar.close()

    try:
        count = collection.count()
    except Exception:
        count = "unknown"

    print(f"Done. Collection '{collection_name}' now has {count} records.")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build Chroma index from PDFs using sentence-transformers.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate the collection before ingesting")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/write batch size")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size (characters)")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap (characters)")

    args = parser.parse_args()

    chroma_dir = Path(get_env("CHROMA_DB_DIR", "data/chroma"))
    raw_docs_dir = Path(get_env("RAW_DOCS_DIR", "data/raw"))
    embedding_model = get_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    build_index(
        source_dir=raw_docs_dir,
        persist_dir=chroma_dir,
        model_name=embedding_model,
        collection_name=args.collection,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

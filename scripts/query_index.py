from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

DEFAULT_COLLECTION = "pwb_docs"


def get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def query_index(query: str, k: int = 4, collection_name: str = DEFAULT_COLLECTION):
    chroma_dir = Path(get_env("CHROMA_DB_DIR", "data/chroma"))
    model_name = get_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(collection_name)

    q_emb = model.encode([query], normalize_embeddings=True).tolist()

    res = collection.query(query_embeddings=q_emb, n_results=k, include=["distances", "metadatas", "documents", "embeddings"])  # type: ignore

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    print("\nTop matches:")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        src = meta.get("filename", meta.get("source", ""))
        page = meta.get("page", "?")
        title = meta.get("title", "")
        print(f"[{i}] {title} (p{page}) — {src} | distance={dist:.4f}")
        snippet = doc.replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        print(f"    {snippet}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Query the Chroma vector DB and view citations.")
    parser.add_argument("--q", required=True, type=str, help="Query text")
    parser.add_argument("--k", type=int, default=4, help="Number of results")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Chroma collection name")

    args = parser.parse_args()

    query_index(args.q, k=args.k, collection_name=args.collection)

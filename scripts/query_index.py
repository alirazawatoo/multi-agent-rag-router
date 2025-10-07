from __future__ import annotations

import argparse

from dotenv import load_dotenv

from ingestion.vectorstore import COLLECTION_NAME, get_embedder, open_collection


def query_index(query: str, k: int = 4, collection_name: str = COLLECTION_NAME) -> None:
    embedder = get_embedder()
    print(f"Embedding backend: {embedder.name}")
    collection = open_collection(embedder, name=collection_name)
    if not collection.count():
        print("The index is empty. Run `python -m ingestion.build_index` first.")
        return

    res = collection.query(
        query_embeddings=embedder.embed([query]),
        n_results=min(k, collection.count()),
        include=["distances", "metadatas", "documents"],
    )

    print("\nTop matches:")
    for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
        src = meta.get("filename", meta.get("source", ""))
        print(f"[{i}] {meta.get('title', '')} (p{meta.get('page', '?')}) — {src} | distance={dist:.4f}")
        snippet = doc.replace("\n", " ")
        print(f"    {snippet[:300] + '...' if len(snippet) > 300 else snippet}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Query the Chroma vector DB and view citations.")
    parser.add_argument("--q", required=True, type=str, help="Query text")
    parser.add_argument("--k", type=int, default=4, help="Number of results")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Chroma collection name")
    args = parser.parse_args()

    query_index(args.q, k=args.k, collection_name=args.collection)

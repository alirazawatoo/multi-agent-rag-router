"""Index settings, embedding backends and the Chroma collection shared by every entry point.

Two embedding backends are available:

* ``hash`` (default): a deterministic token-hashing embedder. It needs no model download and
  no PyTorch, so tests and the Docker demo run offline in a small image. Retrieval is lexical
  (shared words), not semantic.
* ``sentence-transformers``: the semantic model named by ``EMBEDDING_MODEL``. Install
  ``requirements-embeddings.txt`` first; the model is downloaded on first use.

An index has to be queried with the backend that built it, so the backend name is stored in
the collection metadata and checked whenever the collection is opened.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Protocol

COLLECTION_NAME = "pwb_docs"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HASH_DIMENSIONS = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Very common words carry no retrieval signal for a bag-of-words embedder.
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or should "
    "the this to what when where which who why with you your".split()
)


class Embedder(Protocol):
    name: str

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class HashingEmbedder:
    def __init__(self, dimensions: int = HASH_DIMENSIONS):
        self.dimensions = dimensions
        self.name = f"hash-{dimensions}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_BACKEND=sentence-transformers needs `pip install -r requirements-embeddings.txt`"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def chroma_dir() -> Path:
    return Path(os.environ.get("CHROMA_DB_DIR", "data/chroma"))


def raw_docs_dir() -> Path:
    return Path(os.environ.get("RAW_DOCS_DIR", "data/raw"))


def get_embedder() -> Embedder:
    backend = os.environ.get("EMBEDDING_BACKEND", "hash").strip().lower()
    model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return _cached_embedder(backend, model)


@lru_cache(maxsize=4)
def _cached_embedder(backend: str, model: str) -> Embedder:
    if backend == "hash":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model)
    raise ValueError(f"Unknown EMBEDDING_BACKEND {backend!r}; use 'hash' or 'sentence-transformers'")


def open_collection(
    embedder: Embedder,
    persist_dir: Optional[Path] = None,
    name: str = COLLECTION_NAME,
    rebuild: bool = False,
):
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir or chroma_dir()))
    if rebuild:
        client.get_or_create_collection(name)
        client.delete_collection(name)
    # Passing metadata to get_or_create_collection would overwrite an existing collection's tag.
    collection = client.get_or_create_collection(name)
    built_with = (collection.metadata or {}).get("embedding")
    if built_with is None and collection.count() == 0:
        collection.modify(metadata={"embedding": embedder.name})
    elif built_with and built_with != embedder.name:
        raise RuntimeError(
            f"Collection '{name}' was built with {built_with} but EMBEDDING_BACKEND gives {embedder.name}. "
            "Rebuild it with `python -m ingestion.build_index --rebuild`."
        )
    return collection

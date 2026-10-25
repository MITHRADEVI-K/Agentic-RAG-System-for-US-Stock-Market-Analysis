"""Vector store setup.

Uses ChromaDB's built-in all-MiniLM-L6-v2 embedding model (runs locally via
ONNX, free, no API key, no PyTorch install needed).
"""
from __future__ import annotations

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME

_client = None


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

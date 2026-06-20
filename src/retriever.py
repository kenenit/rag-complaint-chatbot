"""
retriever.py — Semantic search over the vector store.
"""

from typing import List, Dict, Any, Literal
from sentence_transformers import SentenceTransformer
import chromadb
import faiss
import numpy as np
import pickle
import os

from src.embedder import EMBEDDING_MODEL


class ChromaRetriever:
    """Retriever backed by ChromaDB."""

    def __init__(self, persist_dir: str = "vector_store/chroma", collection_name: str = "complaints"):
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_collection(collection_name)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"ChromaRetriever ready — {self.collection.count():,} chunks indexed")

    def retrieve(self, query: str, k: int = 5, product_filter: str = None) -> List[Dict[str, Any]]:
        """Embed query and return top-k chunks."""
        query_vec = self.model.encode([query], normalize_embeddings=True).tolist()
        where = {"product_category": product_filter} if product_filter else None
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({"text": doc, "metadata": meta, "score": float(dist)})
        return chunks


class FAISSRetriever:
    """Retriever backed by FAISS."""

    def __init__(self, persist_dir: str = "vector_store/faiss"):
        self.index = faiss.read_index(os.path.join(persist_dir, "index.faiss"))
        with open(os.path.join(persist_dir, "chunks_metadata.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"FAISSRetriever ready — {self.index.ntotal:,} vectors indexed")

    def retrieve(self, query: str, k: int = 5, product_filter: str = None) -> List[Dict[str, Any]]:
        """Embed query and return top-k chunks."""
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(query_vec, k * 3)  # over-fetch to allow filtering
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if product_filter and chunk.get("product_category") != product_filter:
                continue
            results.append({"text": chunk["text"], "metadata": chunk, "score": float(dist)})
            if len(results) == k:
                break
        return results

"""
embedder.py — Generate embeddings and build/persist the vector store.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
import pickle
import os
from tqdm import tqdm


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence transformer model."""
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate embeddings for all chunk texts."""
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts):,} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def build_chroma_store(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    persist_dir: str = "vector_store/chroma",
    collection_name: str = "complaints",
) -> chromadb.Collection:
    """Create and persist a ChromaDB collection."""
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    print(f"ChromaDB collection '{collection_name}' saved to {persist_dir}")
    return collection


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_faiss_store(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    persist_dir: str = "vector_store/faiss",
) -> None:
    """Create and persist a FAISS index + metadata."""
    os.makedirs(persist_dir, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product (cosine for normalised vecs)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(persist_dir, "index.faiss"))
    with open(os.path.join(persist_dir, "chunks_metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"FAISS index ({index.ntotal} vectors) saved to {persist_dir}")

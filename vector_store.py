import os
import json
from typing import Any, Dict, List

import faiss
import numpy as np


class VectorStore:
    """Simple wrapper around a FAISS index with JSON metadata."""

    def __init__(self, index_path: str, meta_path: str, embed_dim: int) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.embed_dim = embed_dim
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load index and metadata from disk, creating new store if missing."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            self.metadata = json.loads(open(self.meta_path).read())
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embed_dim))
            self.metadata = []

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        open(self.meta_path, "w").write(json.dumps(self.metadata, indent=2))

    def add(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        """Add ``vector`` with ``meta`` information to the store."""
        if self.index is None:
            raise ValueError("VectorStore not initialised")
        vec_id = len(self.metadata)
        self.index.add_with_ids(np.expand_dims(vector, 0), np.array([vec_id]))
        self.metadata.append(meta)
        self.save()

    def query(self, vector: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Return ``k`` nearest neighbours for ``vector``."""
        if self.index is None:
            raise ValueError("VectorStore not initialised")
        D, I = self.index.search(np.expand_dims(vector, 0), k)
        return [self.metadata[i] for i in I[0] if i != -1]

    def list(self) -> List[Dict[str, Any]]:
        """Return all stored metadata."""
        return self.metadata

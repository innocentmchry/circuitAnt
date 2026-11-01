import math
import time
from typing import List, Optional, Tuple

import numpy as np


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class EmbeddingModel:
    """Wrapper for sentence embedding models with caching and retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._cache: dict[str, List[float]] = {}

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, text: str) -> Tuple[List[float], float]:
        """Returns (embedding, elapsed_seconds)."""
        if text in self._cache:
            return self._cache[text], 0.0

        self._ensure_model()
        t0 = time.perf_counter()
        emb = self._model.encode(text, normalize_embeddings=True)
        elapsed = time.perf_counter() - t0
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        self._cache[text] = emb_list
        return emb_list, elapsed

    def retrieve_top_k(
        self,
        query_embedding: List[float],
        candidates: List[Tuple[str, List[float]]],  # list of (id, embedding)
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, int]]:
        """Returns list of (id, cosine_sim, rank)."""
        scored = []
        for cand_id, cand_emb in candidates:
            sim = cosine_similarity(query_embedding, cand_emb)
            if sim >= threshold:
                scored.append((cand_id, float(sim)))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return [(cand_id, sim, rank + 1) for rank, (cand_id, sim) in enumerate(top)]

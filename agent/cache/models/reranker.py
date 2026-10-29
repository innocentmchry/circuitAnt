import time
from typing import List, Optional, Tuple


class CrossEncoderReranker:
    """Wrapper for Cross-Encoder models to score (query, candidate_text) pairs."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)

    def score_pairs(self, query: str, candidate_texts: List[str]) -> Tuple[List[float], float]:
        """Returns (scores, elapsed_seconds)."""
        if not candidate_texts:
            return [], 0.0

        self._ensure_model()
        pairs = [(query, cand) for cand in candidate_texts]
        t0 = time.perf_counter()
        raw_scores = self._model.predict(pairs)
        elapsed = time.perf_counter() - t0

        if hasattr(raw_scores, "tolist"):
            scores = raw_scores.tolist()
        elif isinstance(raw_scores, (list, tuple)):
            scores = [float(s) for s in raw_scores]
        else:
            scores = [float(raw_scores)]
        return scores, elapsed

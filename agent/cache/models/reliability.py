import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ReliabilityModel:
    """Lightweight vCache-inspired reliability calibration layer.
    
    Uses logistic regression or threshold calibration on agreement observations
    to predict the probability that a retrieved/adapted tool-call plan matches
    the LLM reference plan.
    """

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self.weights: Optional[List[float]] = None
        self.bias: float = 0.0
        self.is_fitted: bool = False

    def extract_features(
        self,
        cosine_sim: float,
        cross_encoder_score: Optional[float],
        rank: int,
        param_count: int,
        structural_passed: bool,
    ) -> List[float]:
        # Cross encoder score is normalized via sigmoid if available, else 0.5 default
        ce_val = 1.0 / (1.0 + math.exp(-cross_encoder_score)) if cross_encoder_score is not None else 0.5
        rank_feature = 1.0 / float(rank)
        struct_val = 1.0 if structural_passed else 0.0
        param_val = min(1.0, float(param_count) / 5.0)
        return [cosine_sim, ce_val, rank_feature, struct_val, param_val]

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        """Fit logistic regression weights on agreement observations."""
        if len(X) < 5 or len(set(y)) < 2:
            # Fallback heuristic weights if training data is too small or single class
            self.weights = [1.5, 2.0, 1.0, 1.0, 0.5]
            self.bias = -2.5
            self.is_fitted = True
            return

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(class_weight="balanced")
            clf.fit(X, y)
            self.weights = clf.coef_[0].tolist()
            self.bias = float(clf.intercept_[0])
            self.is_fitted = True
        except Exception:
            self.weights = [1.5, 2.0, 1.0, 1.0, 0.5]
            self.bias = -2.5
            self.is_fitted = True

    def predict_probability(self, features: List[float]) -> Tuple[float, float]:
        """Returns (probability, elapsed_seconds)."""
        t0 = time.perf_counter()
        if not self.is_fitted or self.weights is None:
            # Default scoring if not yet fitted: weighted average of similarity and rank
            sim, ce, r, struct, p = features
            prob = 0.4 * sim + 0.3 * ce + 0.15 * r + 0.15 * struct
            elapsed = time.perf_counter() - t0
            return float(min(1.0, max(0.0, prob))), elapsed

        z = self.bias + sum(w * x for w, x in zip(self.weights, features))
        prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
        elapsed = time.perf_counter() - t0
        return float(prob), elapsed

    def is_reliable(self, probability: float) -> bool:
        return probability >= self.confidence_threshold

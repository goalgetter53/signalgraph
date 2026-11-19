"""Pair scorers: L2-regularised logistic regression and Fellegi-Sunter.

Both map a feature vector to P(match). Both are small enough to implement
directly on numpy, which keeps the trained model a readable JSON file of
weights instead of a pickled estimator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from signalgraph.resolution.features import FEATURE_NAMES

FloatArray = NDArray[np.float64]


class PairScorer(Protocol):
    def predict_proba(self, x: FloatArray) -> FloatArray: ...


def _sigmoid(z: FloatArray) -> FloatArray:
    out: FloatArray = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    return out


class LogisticMatcher:
    """Logistic regression fitted with Newton-Raphson (IRLS) and an L2 penalty."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: FloatArray = np.zeros(len(FEATURE_NAMES))
        self.intercept_: float = 0.0

    def fit(self, x: FloatArray, y: NDArray[np.int_]) -> LogisticMatcher:
        n, d = x.shape
        design = np.hstack([np.ones((n, 1)), x])
        w = np.zeros(d + 1)
        penalty = np.eye(d + 1) * self.l2
        penalty[0, 0] = 0.0  # never shrink the intercept
        yf = y.astype(np.float64)
        for _ in range(self.max_iter):
            p = _sigmoid(design @ w)
            grad = design.T @ (p - yf) + penalty @ w
            hess = (design * (p * (1 - p))[:, None]).T @ design + penalty
            step = np.linalg.solve(hess + 1e-9 * np.eye(d + 1), grad)
            w -= step
            if float(np.max(np.abs(step))) < self.tol:
                break
        self.intercept_ = float(w[0])
        self.coef_ = w[1:].copy()
        return self

    def decision_function(self, x: FloatArray) -> FloatArray:
        out: FloatArray = x @ self.coef_ + self.intercept_
        return out

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return _sigmoid(self.decision_function(x))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "logistic",
            "features": list(FEATURE_NAMES),
            "intercept": self.intercept_,
            "coef": dict(zip(FEATURE_NAMES, map(float, self.coef_), strict=True)),
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticMatcher:
        if list(data["features"]) != list(FEATURE_NAMES):
            raise ValueError("model feature list does not match this code version")
        model = cls(l2=float(data.get("l2", 1.0)))
        model.intercept_ = float(data["intercept"])
        model.coef_ = np.array([float(data["coef"][f]) for f in FEATURE_NAMES])
        return model


# Fellegi-Sunter comparison levels per feature: thresholds -> level index.
FS_LEVELS: dict[str, tuple[float, ...]] = {
    "jaro_winkler": (0.97, 0.92, 0.85),
    "tfidf_cosine": (0.9, 0.6, 0.3),
    "token_containment": (0.99, 0.5),
    "residual_similarity": (0.95, 0.8),
    "domain": (),
    "country": (),
}


def _fs_codes(x: FloatArray) -> NDArray[np.int_]:
    """Discretise the feature matrix into comparison-vector level codes."""
    cols = []
    for j, name in enumerate(
        ("jaro_winkler", "tfidf_cosine", "token_containment", "residual_similarity")
    ):
        thresholds = FS_LEVELS[name]
        level = np.full(x.shape[0], len(thresholds), dtype=np.int_)
        for k in reversed(range(len(thresholds))):
            level[x[:, j] >= thresholds[k]] = k
        cols.append(level)
    # domain: 0 match, 1 mismatch, 2 unknown ; same for country
    cols.append(x[:, 4].astype(np.int_))  # number mismatch: 0 / 1
    cols.append(np.where(x[:, 5] > 0, 0, np.where(x[:, 6] > 0, 1, 2)))
    cols.append(np.where(x[:, 7] > 0, 0, np.where(x[:, 8] > 0, 1, 2)))
    return np.stack(cols, axis=1)


class FellegiSunterMatcher:
    """Supervised Fellegi-Sunter: m/u level probabilities estimated from labels.

    Assumes conditional independence of comparison fields given match status
    (the classic FS assumption). Laplace smoothing keeps unseen levels finite.
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.log_m: list[list[float]] = []
        self.log_u: list[list[float]] = []
        self.prior_log_odds = 0.0

    def fit(self, x: FloatArray, y: NDArray[np.int_]) -> FellegiSunterMatcher:
        codes = _fs_codes(x)
        self.log_m, self.log_u = [], []
        for j in range(codes.shape[1]):
            n_levels = int(codes[:, j].max()) + 1
            n_levels = max(n_levels, 4)
            m_counts = np.bincount(codes[y == 1, j], minlength=n_levels) + self.smoothing
            u_counts = np.bincount(codes[y == 0, j], minlength=n_levels) + self.smoothing
            self.log_m.append(list(np.log(m_counts / m_counts.sum())))
            self.log_u.append(list(np.log(u_counts / u_counts.sum())))
        pos = float(y.sum())
        self.prior_log_odds = math.log((pos + 1) / (len(y) - pos + 1))
        return self

    def match_weight(self, x: FloatArray) -> FloatArray:
        codes = _fs_codes(x)
        total = np.zeros(x.shape[0])
        for j in range(codes.shape[1]):
            lm = np.asarray(self.log_m[j])
            lu = np.asarray(self.log_u[j])
            idx = np.minimum(codes[:, j], len(lm) - 1)
            total += lm[idx] - lu[idx]
        return total

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return _sigmoid(self.match_weight(x) + self.prior_log_odds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "fellegi_sunter",
            "log_m": self.log_m,
            "log_u": self.log_u,
            "prior_log_odds": self.prior_log_odds,
        }


DEFAULT_MODEL_PATH = Path(__file__).with_name("er_model.json")


def load_default_model(path: Path = DEFAULT_MODEL_PATH) -> LogisticMatcher:
    return LogisticMatcher.from_dict(json.loads(path.read_text(encoding="utf-8")))

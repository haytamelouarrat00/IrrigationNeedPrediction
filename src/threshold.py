"""Optimize per-class probability weights to maximize balanced accuracy.

Decision rule: argmax_c ( P(c|x) * w_c ).  We fix w_0 = 1 and search w_1, w_2 > 0.

Uses a vectorized recall/balanced-accuracy implementation (bincount-based) instead
of sklearn.balanced_accuracy_score, which is ~50x faster on 630k rows and lets us
nest these calls inside ensemble blend optimization.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def fast_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    correct = np.bincount(y_true[y_true == y_pred], minlength=n_classes)
    total = np.bincount(y_true, minlength=n_classes)
    return float(np.mean(correct / np.maximum(total, 1)))


def _ba(weights: np.ndarray, proba: np.ndarray, y: np.ndarray) -> float:
    w = np.concatenate([[1.0], np.asarray(weights, dtype=float)])
    pred = np.argmax(proba * w, axis=1)
    return fast_balanced_accuracy(y, pred)


def grid_search_weights(
    proba: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Coarse grid search over (w1, w2) for 3-class problems."""
    if grid is None:
        grid = np.geomspace(0.2, 50.0, num=25)
    best_w = np.array([1.0, 1.0])
    best_score = _ba(best_w, proba, y)
    for w1 in grid:
        for w2 in grid:
            score = _ba(np.array([w1, w2]), proba, y)
            if score > best_score:
                best_score = score
                best_w = np.array([w1, w2])
    return best_w, best_score


def refine_weights(
    proba: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Local refinement around an initial weight vector using Nelder-Mead."""
    def neg(w):
        return -_ba(np.clip(w, 1e-3, None), proba, y)
    res = minimize(neg, init, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
    w = np.clip(res.x, 1e-3, None)
    return w, _ba(w, proba, y)


def tune_weights(proba: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Two-stage: coarse grid + local refinement."""
    coarse, _ = grid_search_weights(proba, y)
    refined, score = refine_weights(proba, y, coarse)
    full = np.concatenate([[1.0], refined])
    return full, score


def apply_weights(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(proba * weights, axis=1)

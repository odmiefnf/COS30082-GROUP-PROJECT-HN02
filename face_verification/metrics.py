"""NumPy-only binary verification metrics.

Every score accepted here must be oriented so that a larger value means the
images are more likely to show the same identity (label 1). For a distance,
pass its negative. Thresholds selected on evaluation pairs describe those
validation pairs; they are not estimates of performance on an unseen test set.
"""

from __future__ import annotations

import numpy as np


def _validate_binary_scores(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Validate a nonempty, finite, one-dimensional binary scoring problem."""
    try:
        labels = np.asarray(y_true, dtype=np.float64)
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels and scores must be numeric one-dimensional arrays.") from exc
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("Labels and scores must both be one-dimensional arrays.")
    if labels.size == 0 or labels.shape != values.shape:
        raise ValueError("Labels and scores must have the same nonzero length.")
    if not np.isfinite(labels).all() or not np.isfinite(values).all():
        raise ValueError("Labels and scores must contain only finite values.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Verification labels must be 0 (different) or 1 (same).")
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("ROC/AUC requires at least one label-0 and one label-1 pair.")
    return labels.astype(np.int64), values


def compute_roc(y_true, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr, thresholds)`` for the rule ``score >= threshold``.

    Equal scores enter the accepted set together. The initial threshold is
    positive infinity, with FPR=TPR=0; the final point accepts every pair and
    has FPR=TPR=1. No interpolation or score rounding occurs before ranking.
    """
    labels, values = _validate_binary_scores(y_true, scores)
    order = np.argsort(-values, kind="mergesort")
    ranked_scores = values[order]
    ranked_labels = labels[order]
    # Only evaluate the end of each equal-score group, avoiding tie-order bias.
    group_ends = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), labels.size - 1]
    true_positives = np.cumsum(ranked_labels, dtype=np.int64)[group_ends]
    false_positives = (group_ends + 1) - true_positives
    positives = int(labels.sum())
    negatives = labels.size - positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    thresholds = np.r_[np.inf, ranked_scores[group_ends]]
    return fpr, tpr, thresholds


def auc(fpr, tpr) -> float:
    """Integrate an ROC curve with the trapezoidal rule."""
    try:
        x = np.asarray(fpr, dtype=np.float64)
        y = np.asarray(tpr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("FPR and TPR must be numeric one-dimensional arrays.") from exc
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("FPR and TPR must have the same length of at least two.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("FPR and TPR must contain only finite values.")
    if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
        raise ValueError("ROC coordinates must be ordered by nondecreasing FPR and TPR.")
    if np.any((x < 0) | (x > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("FPR and TPR must lie in [0, 1].")
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) * 0.5))


def compute_auc(y_true, scores) -> float:
    """Compute ROC AUC from labels and continuous, higher-is-same scores."""
    fpr, tpr, _ = compute_roc(y_true, scores)
    return auc(fpr, tpr)


def select_validation_threshold(y_true, scores) -> dict[str, float | int | str]:
    """Choose a finite validation threshold maximizing Youden's ``TPR-FPR``.

    Ties choose the highest score threshold. The threshold and all operating
    point statistics are calibrated on these same pairs, not an unseen test.
    """
    labels, values = _validate_binary_scores(y_true, scores)
    fpr, tpr, thresholds = compute_roc(labels, values)
    finite_indices = np.flatnonzero(np.isfinite(thresholds))
    best = int(finite_indices[np.argmax((tpr - fpr)[finite_indices])])
    threshold = float(thresholds[best])
    accepted = values >= threshold
    tp = int(np.sum(accepted & (labels == 1)))
    fp = int(np.sum(accepted & (labels == 0)))
    tn = int(np.sum(~accepted & (labels == 0)))
    fn = int(np.sum(~accepted & (labels == 1)))
    return {
        "score_threshold": threshold,
        "accept_rule": "score >= threshold",
        "selection_method": "maximum Youden J (TPR-FPR), highest finite threshold on ties",
        "scope": "validation-calibrated / in-sample; not an unbiased test estimate",
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
        "youden_j": float(tpr[best] - fpr[best]),
        "accuracy": float((tp + tn) / labels.size),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


# Familiar alias for callers accustomed to the conventional ROC function name.
roc_curve = compute_roc


__all__ = ["compute_roc", "roc_curve", "auc", "compute_auc", "select_validation_threshold"]

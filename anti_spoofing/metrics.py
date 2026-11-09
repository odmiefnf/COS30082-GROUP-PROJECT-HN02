"""Binary presentation-attack metrics with canonical 0=spoof, 1=live labels."""

from __future__ import annotations

import math

import numpy as np

from face_verification.metrics import compute_auc, select_validation_threshold


def _validate(labels, live_scores) -> tuple[np.ndarray, np.ndarray]:
    try:
        y = np.asarray(labels, dtype=np.float64)
        scores = np.asarray(live_scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels and live scores must be numeric arrays.") from exc
    if y.ndim != 1 or scores.ndim != 1 or y.shape != scores.shape or y.size == 0:
        raise ValueError("Labels and live scores must be nonempty 1D arrays of the same length.")
    if not np.isfinite(y).all() or not np.isfinite(scores).all():
        raise ValueError("Labels and live scores must be finite.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Canonical liveness labels must be 0=spoof or 1=live.")
    if not (np.any(y == 0) and np.any(y == 1)):
        raise ValueError("Liveness evaluation requires both spoof (0) and live (1) samples.")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Live scores must be probabilities in [0, 1].")
    return y.astype(np.int64), scores


def liveness_metrics(labels, live_scores, threshold: float) -> dict:
    """Compute accuracy, APCER, BPCER, ACER and live-positive ROC AUC.

    Accept a face as live when ``P(live) >= threshold``. APCER is the fraction
    of spoof samples accepted as live; BPCER is the fraction of live samples
    rejected as spoof. ACER is their arithmetic mean. Both classes are required.
    """
    y, scores = _validate(labels, live_scores)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
        raise ValueError("The live threshold must be a finite number in [0, 1].")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The live threshold must be a finite number in [0, 1].")
    live = scores >= threshold
    tp = int(np.sum(live & (y == 1)))
    fp = int(np.sum(live & (y == 0)))
    tn = int(np.sum(~live & (y == 0)))
    fn = int(np.sum(~live & (y == 1)))
    apcer = fp / (fp + tn)
    bpcer = fn / (fn + tp)
    return {
        "accuracy": float((tp + tn) / y.size),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float((apcer + bpcer) / 2),
        "AUC": compute_auc(y, scores),
        "threshold": threshold,
        "accept_rule": "live_score >= threshold",
        "sample_count": int(y.size),
        "live_count": int(tp + fn),
        "spoof_count": int(tn + fp),
        "confusion": {
            "true_live": tp,
            "spoof_accepted_live": fp,
            "true_spoof": tn,
            "live_rejected_spoof": fn,
        },
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_order": {"rows": ["spoof", "live"], "columns": ["spoof", "live"]},
        "score_definition": "softmax(logits)[live_label=1]; higher means more likely live",
    }


def calibrate_threshold(labels, live_scores) -> dict:
    """Select a finite threshold on VALIDATION scores only, minimizing ACER.

    Maximizing Youden's TPR-FPR is equivalent to minimizing ACER. Equal
    objective values select the highest candidate score threshold. Callers
    must supply validation scores; evaluation must reuse the saved threshold.
    """
    y, scores = _validate(labels, live_scores)
    selected = select_validation_threshold(y, scores)
    threshold = float(selected["score_threshold"])
    return {
        "threshold": threshold,
        "threshold_source": "validation",
        "threshold_calibrated": True,
        "selection_method": selected["selection_method"],
        "scope": "validation-calibrated / in-sample; not an unbiased test estimate",
        "accept_rule": "live_score >= threshold",
        "validation_metrics": liveness_metrics(y, scores, threshold),
    }


__all__ = ["liveness_metrics", "calibrate_threshold"]

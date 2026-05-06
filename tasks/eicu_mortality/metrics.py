"""Mortality classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def mortality_scores(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Standard ICU mortality eval suite."""
    out: dict[str, float] = {}
    if len(np.unique(labels)) < 2:
        # Degenerate split — return NaNs so callers can detect it.
        return {"auroc": float("nan"), "auprc": float("nan"), "max_f1": float("nan")}

    out["auroc"] = float(roc_auc_score(labels, probs))
    out["auprc"] = float(average_precision_score(labels, probs))

    p, r, _ = precision_recall_curve(labels, probs)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    out["max_f1"] = float(np.max(f1))

    # Sensitivity at 90% specificity — the standard clinically useful operating
    # point for ICU early-warning systems.
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    target_fpr = 0.10
    idx = int(np.argmax(fpr >= target_fpr))
    out["sens_at_spec_90"] = float(tpr[idx]) if fpr[idx] >= target_fpr else float("nan")

    out["pos_rate"] = float(labels.mean())
    return out

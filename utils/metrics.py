"""Evaluation metrics for anomaly detection performance.

This module computes various classification metrics for evaluating
anomaly detection models, including:
- Average Precision (AUPR - Area Under Precision-Recall curve)
- Maximum F1 score across all thresholds
- Maximum precision and recall at various threshold levels
- AUC-ROC (Area Under Receiver Operating Characteristic curve)

Modified from code provided by Mark Tenzer on June 20, 2025
"""

from typing import Optional, Dict
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_scores(
    Y_test: np.ndarray, 
    Y_test_pred: np.ndarray, 
    weight: Optional[np.ndarray] = None, 
    print_result: bool = False, 
    max_size: int | None = None
) -> Dict[str, float]:
    labels = Y_test
    scores = Y_test_pred

    # Sample a subset if too large
    if max_size is not None and len(labels) > max_size:
        print(f"Sampling {max_size} points from {len(labels)} for score computation...")
        idx = np.random.choice(len(labels), size=max_size, replace=False)
        labels = labels[idx]
        scores = scores[idx]
        if weight is not None:
            weight = weight[idx]

    # Compute average precision (AP) with weights
    ap = average_precision_score(labels, scores, sample_weight=weight)

    # Compute AUROC; undefined when only one class is present
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(labels, scores, sample_weight=weight)

    # Compute precision-recall curve and max F1 with weights
    precision, recall, thresholds = precision_recall_curve(
        labels, scores, sample_weight=weight
    )
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(
        numerator, denom, out=np.zeros_like(denom), where=(denom != 0)
    )

    max_f1 = np.max(f1_scores)

    max_precision = 0.0
    max_recall = 0.0
    for thres in np.arange(0.01, 1.01, 0.01):
        preds = (scores >= thres).astype(int)
        precision_thres = precision_score(
            labels, preds, sample_weight=weight, zero_division=0
        )
        if precision_thres > max_precision:
            max_precision = precision_thres

        recall_thres = recall_score(
            labels, preds, sample_weight=weight, zero_division=0
        )
        if recall_thres > max_recall:
            max_recall = recall_thres

    result = {
        "AP": ap,
        "AUROC": auroc,
        "Max F1": max_f1,
    }
    for k, v in result.items():
        # if np.array, convert to float
        if isinstance(v, np.ndarray):
            result[k] = float(v)

    if print_result:
        for k, v in result.items():
            print(f"{k}: {v:.2%}" if not (isinstance(v, float) and np.isnan(v)) else f"{k}: N/A")

    return result

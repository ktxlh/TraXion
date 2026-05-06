"""Ranking metrics for next-POI recommendation.

All functions expect *scores* (higher is better) and integer ground-truth
POI indices.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Fraction of samples where the true POI is within the top-k."""
    return float((ranks < k).mean())


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """NDCG@k with binary relevance (only one relevant item per query)."""
    within_k = ranks < k
    # DCG = 1 / log2(rank + 2)  (rank is 0-indexed)
    dcg = np.where(within_k, 1.0 / np.log2(ranks + 2.0), 0.0)
    # Ideal DCG = 1 / log2(2) = 1.0  (best item at rank 0)
    return float(dcg.mean())


def mrr(ranks: np.ndarray) -> float:
    """Mean Reciprocal Rank."""
    return float((1.0 / (ranks + 1.0)).mean())


def compute_rec_scores(
    ranks: np.ndarray,
    ks: list[int] = (1, 5, 10, 20),
    print_result: bool = True,
) -> dict[str, float]:
    """Compute standard next-POI recommendation metrics.

    Args:
        ranks: 1-D array of 0-indexed ranks of the ground-truth POI for each query.
        ks: list of cutoffs for Recall@K and NDCG@K.
        print_result: whether to print a summary.

    Returns:
        dict with keys like ``recall@1``, ``ndcg@10``, ``mrr``.
    """
    ranks = _to_numpy(ranks).astype(np.float64)
    result: dict[str, float] = {}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(ranks, k)
        result[f"ndcg@{k}"] = ndcg_at_k(ranks, k)
    result["mrr"] = mrr(ranks)

    if print_result:
        parts = [f"{k}: {v:.4f}" for k, v in result.items()]
        print("  ".join(parts))

    return result

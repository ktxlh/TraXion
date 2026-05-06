"""Evaluation metrics for social link prediction.

Two complementary evaluations:

1. Classification: AUC / AP on a held-out mix of (test positives) + (test non-edges),
   sized 1:1 per `prepare_social_splits.py`.

2. Ranking: for each test positive (u, v), sample ``n_negatives`` non-edges
   anchored at u, rank v against them, report Hits@k and MRR averaged over all
   test positives. This is the standard "sampled HR@K" protocol from OSN link
   prediction.

Both consume the same `score_fn(pairs: np.ndarray[N,2]) -> np.ndarray[N]` callable.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(
    test_pos_pairs: np.ndarray,
    test_neg_pairs: np.ndarray,
    score_fn,
) -> dict[str, float]:
    y_pos = np.ones(len(test_pos_pairs), dtype=np.int64)
    y_neg = np.zeros(len(test_neg_pairs), dtype=np.int64)
    pairs = np.concatenate([test_pos_pairs, test_neg_pairs], axis=0)
    scores = score_fn(pairs)
    labels = np.concatenate([y_pos, y_neg])
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def ranking_metrics(
    test_pos_pairs: np.ndarray,
    all_edges: set[tuple[int, int]],
    n_agents: int,
    score_fn,
    n_negatives: int = 100,
    k: int = 10,
    seed: int = 7,
) -> dict[str, float]:
    """Per-anchor sampled ranking. Returns Hits@k and MRR."""
    rng = np.random.default_rng(seed)
    hits, rrs = [], []
    for u, v in test_pos_pairs:
        u, v = int(u), int(v)
        negs: list[int] = []
        while len(negs) < n_negatives:
            cand = rng.integers(0, n_agents, n_negatives * 2)
            for c in cand:
                c = int(c)
                if c == u or c == v:
                    continue
                lo, hi = (u, c) if u < c else (c, u)
                if (lo, hi) in all_edges:
                    continue
                negs.append(c)
                if len(negs) >= n_negatives:
                    break
        pairs = np.array([[u, v]] + [[u, c] for c in negs], dtype=np.int64)
        scores = score_fn(pairs)
        rank = 1 + int((scores[1:] > scores[0]).sum())
        hits.append(1.0 if rank <= k else 0.0)
        rrs.append(1.0 / rank)
    return {"hits_at_10": float(np.mean(hits)), "mrr": float(np.mean(rrs))}

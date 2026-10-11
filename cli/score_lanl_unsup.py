"""Score LANL without fine-tuning: rank fusion of the two pre-training heads.

Reads test.npz / protos.npz written by cli/evaluate_lanl_unsup.py and scores each
test event as

    rank(sigmoid(noise-detection logit)) + rank(-cos(visit_emb, prototype[user]))

with ranks normalized to [0, 1] over the test events. Reports event-level AP,
AUROC and Max F1, and user-level metrics (max-pooled over each user's events).

Usage:
    python -m cli.score_lanl_unsup --dump_dir logs/lanl_unsup_dumps/<tag> [--out json_path]
"""

import argparse
import json
import os
from typing import Dict

import numpy as np


# --------------------------- metrics ---------------------------

def _max_f1(labels, scores):
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(labels, scores)
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.maximum(denom, 1e-12), 0.0)
    return float(np.max(f1))


def event_metrics(labels, scores) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {
        "event/AP": float(average_precision_score(labels, scores)),
        "event/AUROC": float(roc_auc_score(labels, scores)),
        "event/MaxF1": _max_f1(labels, scores),
    }


def user_metrics(agents, labels, scores) -> Dict[str, float]:
    """Max-pool by agent."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    uniq, inv = np.unique(agents, return_inverse=True)
    max_score = np.full(len(uniq), -np.inf, dtype=float)
    max_label = np.zeros(len(uniq), dtype=int)
    np.maximum.at(max_score, inv, scores)
    np.maximum.at(max_label, inv, labels.astype(int))
    return {
        "user/AP": float(average_precision_score(max_label, max_score)),
        "user/AUROC": float(roc_auc_score(max_label, max_score)),
        "user/MaxF1": _max_f1(max_label, max_score),
    }


# --------------------------- scoring ---------------------------

def proto_self_score(visit_embs: np.ndarray, agents: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """-cos(visit_emb, prototype of the event's own user); high = unlike the user."""
    v = visit_embs / (np.linalg.norm(visit_embs, axis=1, keepdims=True) + 1e-12)
    p = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    cos_self = np.einsum("nd,nd->n", v, p[agents])
    return (-cos_self).astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rank_fusion_scores(bce_logits: np.ndarray, visit_embs: np.ndarray, agents: np.ndarray,
                       protos: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    bce = _sigmoid(bce_logits)
    proto = proto_self_score(visit_embs, agents, protos)
    return rankdata(bce) / len(bce) + rankdata(proto) / len(proto)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    test = np.load(os.path.join(args.dump_dir, "test.npz"))
    protos = np.load(os.path.join(args.dump_dir, "protos.npz"))["protos"]
    agents = test["agents"]
    labels = test["labels"].astype(np.float32)
    print(f"Test events: {len(agents):,}, anomalies: {int(labels.sum())}")

    scores = rank_fusion_scores(test["bce_logits"], test["visit_embs"], agents, protos)
    result = {**event_metrics(labels, scores), **user_metrics(agents, labels, scores)}
    for k, v in result.items():
        print(f"  {k:14s} {v:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

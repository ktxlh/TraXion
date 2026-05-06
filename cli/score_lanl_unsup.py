"""Offline scorer for LANL unsupervised dumps from evaluate_lanl_unsup.py.

Reads train.npz / test.npz / protos.npz from a dump directory and reports
event-level and user-level metrics under several test-time scoring recipes:

    A. bce_raw        : current pretraining BCE noise score (sigmoid of logit)
    B. proto_self     : -cos(visit_emb, proto[user])
    C. proto_imp      : max_{j != user} cos(visit_emb, proto[j]) - cos(visit_emb, proto[user])
    D. z_bce          : per-user z(bce_raw)        - calibration only
    E. z_bce_imp_sum  : z_user(bce_raw) + z_user(proto_imp)
    F. z_bce_self_sum : z_user(bce_raw) + z_user(proto_self)

Per-user z stats are estimated from the user's own training-event scores.
Cold-start (test-only) users fall back to global stats.

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


# --------------------------- scoring helpers ---------------------------

def cosine_against_protos(visit_embs: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Return (N, n_agents) cosine similarity matrix."""
    v = visit_embs / (np.linalg.norm(visit_embs, axis=1, keepdims=True) + 1e-12)
    p = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    return v @ p.T


def proto_scores(visit_embs: np.ndarray, agents: np.ndarray, protos: np.ndarray):
    cos = cosine_against_protos(visit_embs, protos)  # (N, A)
    n = cos.shape[0]
    cos_self = cos[np.arange(n), agents]
    # mask out own slot
    cos_masked = cos.copy()
    cos_masked[np.arange(n), agents] = -np.inf
    cos_other = cos_masked.max(axis=1)
    s_self = -cos_self                 # high = doesn't match own user
    s_imp = cos_other - cos_self       # high = better matches another user
    return s_self.astype(np.float32), s_imp.astype(np.float32)


def per_user_z(scores: np.ndarray, agents: np.ndarray,
               train_scores: np.ndarray, train_agents: np.ndarray,
               min_count: int = 8):
    """Z-score test scores by per-user training distribution.

    Cold-start (no train events for that user) → fall back to global z.
    Users with < min_count train events → use a shrinkage estimator that
    blends per-user mean/std with global mean/std.
    """
    # Global stats from train.
    g_mean = float(train_scores.mean())
    g_std = float(train_scores.std() + 1e-6)

    # Per-user train stats.
    train_agents = train_agents.astype(np.int64)
    uniq_tr, inv_tr = np.unique(train_agents, return_inverse=True)
    sums = np.bincount(inv_tr, weights=train_scores, minlength=len(uniq_tr))
    sqs = np.bincount(inv_tr, weights=train_scores ** 2, minlength=len(uniq_tr))
    cnts = np.bincount(inv_tr, minlength=len(uniq_tr))
    means = sums / np.maximum(cnts, 1)
    var = np.maximum(sqs / np.maximum(cnts, 1) - means ** 2, 0.0)
    stds = np.sqrt(var) + 1e-6

    user_to_mean = dict(zip(uniq_tr.tolist(), means.tolist()))
    user_to_std = dict(zip(uniq_tr.tolist(), stds.tolist()))
    user_to_cnt = dict(zip(uniq_tr.tolist(), cnts.tolist()))

    out = np.empty_like(scores)
    for i, a in enumerate(agents.tolist()):
        c = user_to_cnt.get(a, 0)
        if c == 0:
            out[i] = (scores[i] - g_mean) / g_std
        elif c < min_count:
            # Shrinkage toward global based on count
            w = c / float(min_count)
            mu = w * user_to_mean[a] + (1 - w) * g_mean
            sd = w * user_to_std[a] + (1 - w) * g_std
            out[i] = (scores[i] - mu) / max(sd, 1e-6)
        else:
            out[i] = (scores[i] - user_to_mean[a]) / user_to_std[a]
    return out.astype(np.float32)


# --------------------------- main ---------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--min_count", type=int, default=8,
                   help="Min train events per user for per-user z-norm; smaller users shrink toward global.")
    args = p.parse_args()

    test = np.load(os.path.join(args.dump_dir, "test.npz"))
    train = np.load(os.path.join(args.dump_dir, "train.npz"))
    protos = np.load(os.path.join(args.dump_dir, "protos.npz"))["protos"]

    te_agents = test["agents"]
    te_labels = test["labels"].astype(np.float32)
    te_bce_logits = test["bce_logits"]
    te_visit = test["visit_embs"]

    tr_agents = train["agents"]
    tr_bce_logits = train["bce_logits"]
    tr_visit = train["visit_embs"]

    print(f"Test events: {len(te_agents):,}, anomalies: {int(te_labels.sum())}")
    print(f"Train events (for per-user stats): {len(tr_agents):,}")
    print(f"Prototype bank: {protos.shape}")

    # --- raw scores ---
    te_bce_raw = _sigmoid(te_bce_logits)
    tr_bce_raw = _sigmoid(tr_bce_logits)

    # --- prototype-based scores ---
    print("Computing prototype scores ...")
    te_self, te_imp = proto_scores(te_visit, te_agents, protos)
    tr_self, tr_imp = proto_scores(tr_visit, tr_agents, protos)

    # --- per-user z-norm (using train) ---
    print("Computing per-user z-norm ...")
    te_z_bce = per_user_z(te_bce_raw, te_agents, tr_bce_raw, tr_agents, args.min_count)
    te_z_self = per_user_z(te_self, te_agents, tr_self, tr_agents, args.min_count)
    te_z_imp = per_user_z(te_imp, te_agents, tr_imp, tr_agents, args.min_count)

    # global z (standardize each score globally, no per-user splits)
    def gz(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    te_gz_bce = gz(te_bce_raw)
    te_gz_self = gz(te_self)
    te_gz_imp = gz(te_imp)

    # rank fusion: convert to ranks in [0, 1] and combine
    from scipy.stats import rankdata
    te_r_bce = rankdata(te_bce_raw) / len(te_bce_raw)
    te_r_self = rankdata(te_self) / len(te_self)
    te_r_imp = rankdata(te_imp) / len(te_imp)

    # --- recipes ---
    recipes = {
        # singletons
        "A_bce_raw":             te_bce_raw,
        "B_proto_self":          te_self,
        "C_proto_imp":           te_imp,
        # per-user z (uses train stats)
        "D_z_bce":               te_z_bce,
        "E_z_bce_p_z_imp":       te_z_bce + te_z_imp,
        "F_z_bce_p_z_self":      te_z_bce + te_z_self,
        "G_z_self_p_z_imp":      te_z_self + te_z_imp,
        "H_all3_zsum":           te_z_bce + te_z_self + te_z_imp,
        # global-z sum (no per-user split)
        "I_gz_bce_p_gz_self":    te_gz_bce + te_gz_self,
        "J_gz_bce_p_gz_imp":     te_gz_bce + te_gz_imp,
        "K_gz_all3":             te_gz_bce + te_gz_self + te_gz_imp,
        # rank-fusion
        "L_r_bce_p_r_self":      te_r_bce + te_r_self,
        "M_r_bce_p_r_imp":       te_r_bce + te_r_imp,
        "N_r_all3":              te_r_bce + te_r_self + te_r_imp,
        # weighted (BCE-leaning)
        "O_2bce_p_self":         2*te_gz_bce + te_gz_self,
        "P_3bce_p_self":         3*te_gz_bce + te_gz_self,
        # mean-only per-user calibration of BCE (drop std)
        "Q_bce_minus_user_mean": te_bce_raw - per_user_z(te_bce_raw, te_agents, tr_bce_raw, tr_agents, args.min_count) * 0,
    }
    # Q is wrong above; replace with explicit centering recipe
    # per-user mean from train
    tr_uniq, tr_inv = np.unique(tr_agents, return_inverse=True)
    tr_sum = np.bincount(tr_inv, weights=tr_bce_raw, minlength=len(tr_uniq))
    tr_cnt = np.bincount(tr_inv, minlength=len(tr_uniq))
    tr_mean = tr_sum / np.maximum(tr_cnt, 1)
    user_mean = dict(zip(tr_uniq.tolist(), tr_mean.tolist()))
    g_mean = float(tr_bce_raw.mean())
    te_user_mean = np.array([user_mean.get(a, g_mean) for a in te_agents.tolist()], dtype=np.float32)
    recipes["Q_bce_minus_user_mean"] = te_bce_raw - te_user_mean
    recipes["R_bce_minus_user_mean_p_self"] = (te_bce_raw - te_user_mean) + 0.5 * te_self
    recipes["S_bce_p_self_alpha0.3"] = te_gz_bce + 0.3 * te_gz_self
    recipes["T_bce_p_self_alpha0.5"] = te_gz_bce + 0.5 * te_gz_self
    recipes["U_bce_p_self_alpha0.7"] = te_gz_bce + 0.7 * te_gz_self
    # finer sweep — bce-leaning
    recipes["V_bce_p_self_alpha0.1"] = te_gz_bce + 0.1 * te_gz_self
    recipes["W_bce_p_self_alpha0.2"] = te_gz_bce + 0.2 * te_gz_self
    # raw-bce + scaled proto (keep raw bce as anchor)
    bce_std = float(te_bce_raw.std() + 1e-8)
    self_std = float(te_self.std() + 1e-8)
    recipes["X_raw_bce_p_self_norm0.05"] = te_bce_raw + 0.05 * (te_self * (bce_std / self_std))
    recipes["Y_raw_bce_p_self_norm0.10"] = te_bce_raw + 0.10 * (te_self * (bce_std / self_std))
    recipes["Z_raw_bce_p_self_norm0.20"] = te_bce_raw + 0.20 * (te_self * (bce_std / self_std))

    out = {}
    for name, s in recipes.items():
        ev = event_metrics(te_labels, s)
        us = user_metrics(te_agents, te_labels, s)
        merged = {**ev, **us}
        out[name] = merged
        print(f"\n[{name}]")
        for k, v in merged.items():
            print(f"  {k:14s} {v:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

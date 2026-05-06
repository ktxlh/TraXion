"""Evaluate an ensemble of (trained social model) + (handcrafted LR baseline).

Diagnostic: measures how much of the baseline-gap comes from the pair_head
dropping the handcrafted signal vs. the embedding being uninformative.

For each alpha in [0, 1]:
    prob = alpha * sigmoid(LR_logit) + (1-alpha) * sigmoid(model_logit)

Model and LR scores are precomputed once per eval pair, then recombined per
alpha (cheap). Negatives for ranking are sampled once with a fixed seed so
Hits@10/MRR are directly comparable across alphas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_dataset_config
from tasks.social.baseline import _sample_train_nonedges
from tasks.social.dataset import SocialTrajectoryDataset, social_collate_fn
from tasks.social.model import SocialModel
from tasks.social.pair_features import FEATURE_DIM as PAIR_FEATURE_DIM
from tasks.social.pair_features import PairFeatureComputer
from utils.file_io import load_dataset_metadata, load_pois, load_stays
from utils.misc import split_train_to_train_val
from utils.training_data import (
    ensure_category_columns,
    normalize_and_validate_agents,
    prepare_stays,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import os as _os
SPLITS_ROOT = Path(_os.environ.get("TRAXION_DATA_DIR", "./data")) / "social_splits"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _compute_model_scores(pairs, model, embs, pfc, device, chunk=16384):
    out = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pairs), chunk):
            p = pairs[s : s + chunk]
            e_i = embs[p[:, 0]]
            e_j = embs[p[:, 1]]
            if pfc is not None and model.pair_feature_dim > 0:
                pf = torch.from_numpy(pfc.features_batch(p)).to(device=device, dtype=e_i.dtype)
            else:
                pf = None
            out[s : s + chunk] = model.score_pairs(e_i, e_j, pf).cpu().numpy()
    return out


def _sample_nonedges(n_agents, forbidden, n, seed):
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        u = int(rng.integers(0, n_agents)); v = int(rng.integers(0, n_agents))
        if u == v: continue
        lo, hi = (u, v) if u < v else (v, u)
        if (lo, hi) in forbidden or (lo, hi) in seen: continue
        seen.add((lo, hi)); out.append((lo, hi))
    return np.asarray(out, dtype=np.int64)


def _sample_ranking_negatives(positives, forbidden, n_agents, n_negatives=100, seed=7):
    """For each positive (u, v), sample n_negatives non-edges anchored at u.

    Returns (P, n_negatives) int64 array of negative partners."""
    rng = np.random.default_rng(seed)
    out = np.empty((len(positives), n_negatives), dtype=np.int64)
    for row, (u, v) in enumerate(positives.tolist()):
        u = int(u); v = int(v); negs = []
        while len(negs) < n_negatives:
            cand = rng.integers(0, n_agents, n_negatives * 2)
            for c in cand:
                c = int(c)
                if c == u or c == v: continue
                lo, hi = (u, c) if u < c else (c, u)
                if (lo, hi) in forbidden: continue
                negs.append(c)
                if len(negs) >= n_negatives: break
        out[row] = negs
    return out


def _ranking_metrics_from_scores(pos_scores, neg_scores, k=10):
    """pos_scores: (P,). neg_scores: (P, n_neg). Returns dict."""
    rank = 1 + (neg_scores > pos_scores[:, None]).sum(axis=1)
    hits = (rank <= k).mean()
    mrr = (1.0 / rank).mean()
    return {"hits_at_10": float(hits), "mrr": float(mrr)}


def run(args):
    cfg = get_dataset_config(args.dataset)
    meta = load_dataset_metadata(cfg)
    _, categories = load_pois(cfg)

    ck = torch.load(args.social_checkpoint, map_location=args.device, weights_only=False)
    ck_args = ck.get("args", {})

    class _A:
        def __init__(self, d): self.__dict__.update(d)
    a = _A(ck_args); a.device = args.device
    a.n_categories = len(categories); a.n_agents = int(meta["n_agents"])

    raw_train = load_stays("train", cfg)
    stays = prepare_stays(raw_train, min_time=meta["min_time"])
    stays = ensure_category_columns(stays, categories)
    tr, vl = split_train_to_train_val(stays)
    tr, vl = normalize_and_validate_agents(tr, vl)
    all_stays = pd.concat([tr, vl], ignore_index=True)

    train_e = pd.read_parquet(SPLITS_ROOT / args.dataset / "train_edges.parquet")
    val_e = pd.read_parquet(SPLITS_ROOT / args.dataset / "val_edges.parquet")
    test_e = pd.read_parquet(SPLITS_ROOT / args.dataset / "test_edges.parquet")
    test_ne = pd.read_parquet(SPLITS_ROOT / args.dataset / "test_nonedges.parquet")
    n_agents = int(pd.read_json(SPLITS_ROOT / args.dataset / "meta.json", typ="series")["n_agents"])
    all_observed = set(
        map(tuple, pd.concat([train_e, val_e, test_e])[["u", "v"]].values.tolist())
    )

    pair_feature_dim = PAIR_FEATURE_DIM if not getattr(a, "disable_pair_features", False) else 0
    pfc = PairFeatureComputer(train_stays=tr, train_edges=train_e)

    model = SocialModel(a, pair_feature_dim=pair_feature_dim).to(args.device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    ds = SocialTrajectoryDataset(
        stays=all_stays, categories=categories,
        clique_size=int(a.clique_size), n_agents=n_agents,
        max_seq_len=getattr(a, "max_seq_len", None),
        overlap_min_frac=getattr(a, "overlap_min_frac", 0.0),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False,
        collate_fn=social_collate_fn, num_workers=4,
    )
    embs = torch.zeros(n_agents, model.emb_dim, device=args.device)
    print("Computing agent embeddings ...", flush=True)
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            e = model.encode(
                batch["location"], batch["start"], batch["stop"],
                batch["category"], batch["agent"],
                batch["padding_mask"], batch["neighbor_padding_mask"],
            )
            embs[batch["agent_ids"].to(args.device)] = e

    # Fit LR baseline
    print("Fitting LR baseline ...", flush=True)
    train_ne = _sample_train_nonedges(n_agents, all_observed, len(train_e), seed=123)
    X_tr = np.vstack([pfc.features_batch(train_e[["u", "v"]].values),
                      pfc.features_batch(train_ne[["u", "v"]].values)])
    y_tr = np.concatenate([np.ones(len(train_e)), np.zeros(len(train_ne))])
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X_tr), y_tr)

    def lr_logit(pairs):
        pf = scaler.transform(pfc.features_batch(pairs))
        # Log-odds (clf.decision_function returns log-odds for binary)
        return clf.decision_function(pf).astype(np.float32)

    # --- Val pairs ---
    val_pos = val_e[["u", "v"]].values.astype(np.int64)
    val_neg = _sample_nonedges(n_agents, all_observed, len(val_pos), seed=321)
    # --- Test pairs ---
    test_pos = test_e[["u", "v"]].values.astype(np.int64)
    test_neg = test_ne[["u", "v"]].values.astype(np.int64)

    # Scores for classification metrics
    print("Scoring val/test pairs ...", flush=True)
    def both_scores(pairs):
        m = _compute_model_scores(pairs, model, embs, pfc, args.device)  # logits
        l = lr_logit(pairs)
        return m, l

    m_vp, l_vp = both_scores(val_pos)
    m_vn, l_vn = both_scores(val_neg)
    m_tp, l_tp = both_scores(test_pos)
    m_tn, l_tn = both_scores(test_neg)

    # Ranking pairs: for each test positive, 100 sampled negatives anchored at u.
    print("Sampling ranking negatives ...", flush=True)
    val_rank_negs = _sample_ranking_negatives(val_pos, all_observed, n_agents, 100, seed=7)
    test_rank_negs = _sample_ranking_negatives(test_pos, all_observed, n_agents, 100, seed=7)

    # Flatten to (P, 101, 2) then (P*101, 2) for scoring.
    def rank_pair_array(positives, neg_partners):
        P = len(positives); K = neg_partners.shape[1]
        out = np.empty((P * (K + 1), 2), dtype=np.int64)
        out[0::K+1] = positives
        # Fill negatives
        anchor = positives[:, 0:1]  # use u as anchor
        for k in range(K):
            out[k+1::K+1, 0] = anchor[:, 0]
            out[k+1::K+1, 1] = neg_partners[:, k]
        return out, P, K

    print("Scoring ranking pairs (val) ...", flush=True)
    vr_pairs, P_v, K = rank_pair_array(val_pos, val_rank_negs)
    m_vr, l_vr = both_scores(vr_pairs)
    m_vr = m_vr.reshape(P_v, K + 1); l_vr = l_vr.reshape(P_v, K + 1)

    print("Scoring ranking pairs (test) ...", flush=True)
    tr_pairs, P_t, _ = rank_pair_array(test_pos, test_rank_negs)
    m_tr, l_tr = both_scores(tr_pairs)
    m_tr = m_tr.reshape(P_t, K + 1); l_tr = l_tr.reshape(P_t, K + 1)

    print()
    print(f"{'alpha':<6} {'val_AUC':<8} {'val_AP':<8} {'val_H10':<8} {'val_MRR':<8} | "
          f"{'test_AUC':<8} {'test_AP':<8} {'test_H10':<8} {'test_MRR':<8}")
    print("-" * 110)
    best_alpha = 0.0; best_val_auc = -1.0
    rows = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Classification metrics
        def combine(m, l):
            return alpha * sigmoid(l) + (1 - alpha) * sigmoid(m)
        v_scores = np.concatenate([combine(m_vp, l_vp), combine(m_vn, l_vn)])
        v_labels = np.concatenate([np.ones_like(m_vp), np.zeros_like(m_vn)])
        t_scores = np.concatenate([combine(m_tp, l_tp), combine(m_tn, l_tn)])
        t_labels = np.concatenate([np.ones_like(m_tp), np.zeros_like(m_tn)])
        v_auc = roc_auc_score(v_labels, v_scores)
        v_ap = average_precision_score(v_labels, v_scores)
        t_auc = roc_auc_score(t_labels, t_scores)
        t_ap = average_precision_score(t_labels, t_scores)

        # Ranking metrics (val + test)
        v_all = combine(m_vr, l_vr)  # (P_v, K+1), col 0 = positive
        v_rnk = _ranking_metrics_from_scores(v_all[:, 0], v_all[:, 1:])
        t_all = combine(m_tr, l_tr)
        t_rnk = _ranking_metrics_from_scores(t_all[:, 0], t_all[:, 1:])

        print(f"{alpha:<6.2f} {v_auc:<8.4f} {v_ap:<8.4f} {v_rnk['hits_at_10']:<8.4f} {v_rnk['mrr']:<8.4f} | "
              f"{t_auc:<8.4f} {t_ap:<8.4f} {t_rnk['hits_at_10']:<8.4f} {t_rnk['mrr']:<8.4f}")
        rows[alpha] = (v_auc, v_ap, v_rnk, t_auc, t_ap, t_rnk)
        if v_auc > best_val_auc:
            best_val_auc = v_auc; best_alpha = alpha
    _, _, _, ta, tap, trnk = rows[best_alpha]
    print()
    print(f"Best alpha by val AUC: {best_alpha}")
    print(f"TEST: AUC={ta:.4f} AP={tap:.4f} Hits@10={trnk['hits_at_10']:.4f} MRR={trnk['mrr']:.4f}")
    print(f"AUC & AP & Hits@10 & MRR:  {ta:.4f} & {tap:.4f} & {trnk['hits_at_10']:.4f} & {trnk['mrr']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--social_checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

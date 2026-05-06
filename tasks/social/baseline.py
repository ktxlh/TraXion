"""Handcrafted-feature logistic-regression baseline for social link prediction.

Reports AUC / AP / Hits@10 / MRR per dataset. Learned model must beat these.

Usage::

    python -m tasks.social.baseline --dataset foursquare-tokyo
    python -m tasks.social.baseline --dataset gowalla-stockholm-v1
    python -m tasks.social.baseline --dataset gowalla-austin-v1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import get_dataset_config
from tasks.social.pair_features import FEATURE_NAMES, PairFeatureComputer
from utils.file_io import load_dataset_metadata, load_stays
from utils.misc import split_train_to_train_val
from utils.training_data import normalize_and_validate_agents, prepare_stays


SPLITS_ROOT = Path(os.environ.get("TRAXION_DATA_DIR", "./data")) / "social_splits"


def _sample_train_nonedges(n_agents: int, all_edges: set[tuple[int, int]], k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(out) < k:
        batch = max(2 * (k - len(out)), 1024)
        u = rng.integers(0, n_agents, batch)
        v = rng.integers(0, n_agents, batch)
        m = u != v
        u, v = u[m], v[m]
        lo = np.minimum(u, v); hi = np.maximum(u, v)
        for a, b in zip(lo.tolist(), hi.tolist()):
            key = (a, b)
            if key in all_edges or key in seen:
                continue
            seen.add(key); out.append(key)
            if len(out) >= k:
                break
    return pd.DataFrame(out, columns=["u", "v"], dtype=np.int64)


def _sampled_ranking_metrics(
    test_pos: pd.DataFrame,
    all_edges: set[tuple[int, int]],
    n_agents: int,
    score_fn,
    n_negatives: int = 100,
    k: int = 10,
    seed: int = 7,
) -> tuple[float, float]:
    """For each test positive (u, v), sample ``n_negatives`` non-edges anchored at u,
    rank v against them, compute Hits@k and MRR. Averaged over test positives."""
    rng = np.random.default_rng(seed)
    hits, rrs = [], []
    for u, v in test_pos[["u", "v"]].itertuples(index=False):
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
        rank = 1 + int((scores[1:] > scores[0]).sum())  # ties broken pessimistically
        hits.append(1.0 if rank <= k else 0.0)
        rrs.append(1.0 / rank)
    return float(np.mean(hits)), float(np.mean(rrs))


def run(dataset: str) -> None:
    out_dir = SPLITS_ROOT / dataset
    train_e = pd.read_parquet(out_dir / "train_edges.parquet")
    val_e = pd.read_parquet(out_dir / "val_edges.parquet")
    test_e = pd.read_parquet(out_dir / "test_edges.parquet")
    test_ne = pd.read_parquet(out_dir / "test_nonedges.parquet")
    meta = pd.read_json(out_dir / "meta.json", typ="series").to_dict()
    n_agents = int(meta["n_agents"])

    cfg = get_dataset_config(dataset)
    md = load_dataset_metadata(cfg)
    stays = prepare_stays(load_stays("train", cfg), min_time=md["min_time"])
    tr, vl = split_train_to_train_val(stays)
    tr, _ = normalize_and_validate_agents(tr, vl)

    print(f"[{dataset}] building pair feature computer ...")
    pfc = PairFeatureComputer(train_stays=tr, train_edges=train_e)

    # All observed edges (train+val+test) are "forbidden" for training-negative sampling.
    all_edges = set(map(tuple, pd.concat([train_e, val_e, test_e])[["u", "v"]].values.tolist()))

    # Build training set: train_edges (label=1) + equal random non-edges (label=0).
    train_ne = _sample_train_nonedges(n_agents, all_edges, len(train_e), seed=123)

    print(f"[{dataset}] computing features for {len(train_e):,} pos + {len(train_ne):,} neg training pairs ...")
    X_tr = np.vstack([pfc.features_batch(train_e[["u", "v"]].values),
                      pfc.features_batch(train_ne[["u", "v"]].values)])
    y_tr = np.concatenate([np.ones(len(train_e)), np.zeros(len(train_ne))])

    print(f"[{dataset}] computing features for {len(test_e):,} pos + {len(test_ne):,} neg test pairs ...")
    X_te = np.vstack([pfc.features_batch(test_e[["u", "v"]].values),
                      pfc.features_batch(test_ne[["u", "v"]].values)])
    y_te = np.concatenate([np.ones(len(test_e)), np.zeros(len(test_ne))])

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr_s, y_tr)

    from sklearn.metrics import average_precision_score, roc_auc_score
    p_te = clf.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    ap = average_precision_score(y_te, p_te)

    def score_fn(pairs: np.ndarray) -> np.ndarray:
        feats = pfc.features_batch(pairs)
        return clf.predict_proba(scaler.transform(feats))[:, 1]

    hits10, mrr = _sampled_ranking_metrics(
        test_e, all_edges=all_edges, n_agents=n_agents,
        score_fn=score_fn, n_negatives=100, k=10,
    )

    print()
    print(f"[{dataset}] === Handcrafted-only baseline ===")
    print(f"  AUC      = {auc:.4f}")
    print(f"  AP       = {ap:.4f}")
    print(f"  Hits@10  = {hits10:.4f}")
    print(f"  MRR      = {mrr:.4f}")
    print(f"  coefs:   " + "  ".join(f"{n}={c:+.3f}" for n, c in zip(FEATURE_NAMES, clf.coef_[0])))
    print(f"  AUC & AP & Hits@10 & MRR:  {auc:.4f} & {ap:.4f} & {hits10:.4f} & {mrr:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()

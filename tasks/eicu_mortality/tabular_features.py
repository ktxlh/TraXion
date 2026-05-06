"""Per-stay tabular features (demographics + admit context + event aggregates).

Combined with the pooled visit embedding inside MortalityModel, this lets the
fine-tune head see exactly the same admin signal that the XGBoost full-feature
baseline uses (~0.74 AUROC on test). Without this, CLIQUE only sees the event
sequence and starts ~20 AUROC points behind on this small dataset.

Returns a `(feature_dict, feature_names, scaler)` tuple where `feature_dict`
is `dict[patientunitstayid -> np.ndarray]`. The scaler is fit on the train+val
slice and applied to all rows so eval values are comparable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _age_to_num(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, str):
        s = s.strip()
        if s == "> 89" or s == ">89":
            return 90.0
    try:
        return float(s)
    except Exception:
        return np.nan


def _norm_apache(d):
    if not isinstance(d, str):
        return "unknown"
    d = d.lower().strip()
    d = re.sub(r"\s+", " ", d)
    return d if d else "unknown"


def _onehot(series: pd.Series, vocab: list[str], prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=series.index)
    for v in vocab:
        col = f"{prefix}_{re.sub('[^A-Za-z0-9]+', '_', str(v))[:24]}"
        out[col] = (series == v).astype(np.float32)
    out[f"{prefix}_OTHER"] = (~series.isin(vocab)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------

@dataclass
class TabularFeatureBundle:
    feature_dim: int
    feature_names: list[str]
    by_stay: dict[int, np.ndarray]
    mean: np.ndarray
    std: np.ndarray


def build_tabular_features(
    stay_labels: pd.DataFrame,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    apache_top_k: int = 15,
) -> TabularFeatureBundle:
    """Build a (D,) feature vector per stay; z-score using train+val statistics."""
    sl = stay_labels.copy()
    sl["patientunitstayid"] = sl["patientunitstayid"].astype(np.int64)

    # ---- demographics ----
    sl["age_num"] = sl["age"].apply(_age_to_num)
    sl["age_log"] = np.log1p(sl["age_num"].fillna(0.0))
    sl["male"] = (sl["gender"] == "Male").astype(np.float32)
    sl["female"] = (sl["gender"] == "Female").astype(np.float32)
    sl["height"] = pd.to_numeric(sl.get("admissionheight"), errors="coerce")
    sl["weight_admit"] = pd.to_numeric(sl.get("admissionweight"), errors="coerce")
    if "dischargeweight" in sl.columns:
        sl["weight_disch"] = pd.to_numeric(sl["dischargeweight"], errors="coerce")
    else:
        sl["weight_disch"] = sl["weight_admit"]
    sl["bmi"] = sl["weight_admit"] / ((sl["height"] / 100.0) ** 2)
    sl["unit_visit_num"] = pd.to_numeric(sl.get("unitvisitnumber"), errors="coerce").fillna(1.0)
    sl["log_unit_visit"] = np.log1p(sl["unit_visit_num"])

    # Categorical one-hots: top-K vocab from train+val rows only.
    train_mask = sl["split"].isin(["train", "val"])
    eth_top = sl.loc[train_mask, "ethnicity"].value_counts().head(6).index.tolist()
    unit_top = sl.loc[train_mask, "unittype"].value_counts().head(8).index.tolist()
    if "unitstaytype" in sl.columns:
        stay_top = sl.loc[train_mask, "unitstaytype"].value_counts().head(4).index.tolist()
    else:
        sl["unitstaytype"] = "unknown"
        stay_top = ["unknown"]
    apache_norm = sl["apacheadmissiondx"].apply(_norm_apache)
    apache_top = apache_norm.loc[train_mask].value_counts().head(apache_top_k).index.tolist()

    eth_oh = _onehot(sl["ethnicity"], eth_top, "eth")
    unit_oh = _onehot(sl["unittype"], unit_top, "unit")
    stay_oh = _onehot(sl["unitstaytype"], stay_top, "stay")
    apache_oh = _onehot(apache_norm, apache_top, "apx")

    # ---- event aggregates ----
    all_events = pd.concat([train_events, test_events], ignore_index=True)
    counts = (
        all_events.pivot_table(
            index="patientunitstayid", columns="venue_type",
            values="agent", aggfunc="count", fill_value=0,
        )
        .rename(columns=lambda c: f"count_{c}")
    )
    distinct = (
        all_events.groupby("patientunitstayid")["poi_id"].nunique().rename("n_distinct")
    )
    total = all_events.groupby("patientunitstayid").size().rename("n_events")
    starts = all_events["start"]
    if pd.api.types.is_datetime64_any_dtype(starts):
        span_series = (
            all_events.assign(_start_h=starts.astype("int64") / 3.6e12)  # ns -> hours
            .groupby("patientunitstayid")["_start_h"]
            .agg(lambda s: float(s.max() - s.min()))
        )
    else:
        # `start` is already float (hours since min_time, after prepare_stays).
        span_series = (
            all_events.groupby("patientunitstayid")["start"]
            .agg(lambda s: float(s.max() - s.min()))
        )
    span = span_series.rename("time_span_h")
    agg = pd.concat([counts, distinct, total, span], axis=1).fillna(0.0)
    agg.index = agg.index.astype(np.int64)
    # Log-transform heavy-tailed counts.
    for col in agg.columns:
        agg[col] = np.log1p(agg[col].astype(np.float64))

    sl_idx = sl.set_index("patientunitstayid")
    feats = pd.concat(
        [
            sl_idx[["age_num", "age_log", "male", "female",
                    "height", "weight_admit", "weight_disch", "bmi",
                    "unit_visit_num", "log_unit_visit"]],
            eth_oh.set_index(sl_idx.index),
            unit_oh.set_index(sl_idx.index),
            stay_oh.set_index(sl_idx.index),
            apache_oh.set_index(sl_idx.index),
        ],
        axis=1,
    )
    feats = feats.join(agg, how="left").fillna(0.0)
    feats = feats.replace([np.inf, -np.inf], 0.0)

    # z-score using train+val statistics
    train_idx = sl.loc[train_mask, "patientunitstayid"].astype(np.int64).values
    train_feats = feats.loc[train_idx].astype(np.float32).values
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    arr = ((feats.astype(np.float32).values - mean) / std).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    by_stay = {int(idx): arr[i] for i, idx in enumerate(feats.index.tolist())}
    return TabularFeatureBundle(
        feature_dim=arr.shape[1],
        feature_names=list(feats.columns),
        by_stay=by_stay,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

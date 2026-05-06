"""Metrics for next-visit prediction.

Re-uses ranking helpers from ``tasks.poi_rec.metrics`` for H@k.

T@t (fraction of next-timestamp predictions within ``t`` minutes of ground
truth, conditioned on the true next location): we report it for several
``t`` values so a single one can be picked afterwards.

We use a *point* prediction strategy: the predicted time delta is the
GMM's expected value (∑ w_i · μ_i over the components).  T@t is then the
fraction of |delta_pred − delta_gt| ≤ t/60 hours.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from tasks.poi_rec.metrics import compute_rec_scores, recall_at_k, ndcg_at_k, mrr  # noqa: F401


T_VALUES_DEFAULT_MIN: tuple[int, ...] = (5, 10, 30, 60, 120, 240)


def _norm_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight / weight.sum(dim=-1, keepdim=True)


def gmm_expected_value(weight: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """Expected value of the mixture: ``∑ w_i · μ_i``."""
    return (_norm_weight(weight) * loc).sum(dim=-1)


def gmm_mode_loc(weight: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """Pick the loc of the component with the largest mixture weight.

    A coarse but useful point estimator when the mixture is multimodal — the
    full expected value is pulled by long-tailed components (e.g. multi-day
    inter-visit gaps).  Returns a tensor of shape (N,).
    """
    w = _norm_weight(weight)
    idx = w.argmax(dim=-1, keepdim=True)
    return loc.gather(-1, idx).squeeze(-1)


def gmm_mass_within(weight: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor,
                    target: torch.Tensor, half_window: torch.Tensor,
                    target_space: str = "raw") -> torch.Tensor:
    """Probability mass of the predicted mixture within ``target ± half_window``.

    ``target_space='raw'``: GMM is over Δ directly; mass = F(target+h) - F(target-h).
    ``target_space='log'``: GMM is over Y = log(Δ+1); we transform the raw
    interval ``[target-h, target+h]`` (in hours) into ``[log(max(target-h,0)+1),
    log(target+h+1)]`` before reading the CDF.  This is exact (the log-CDF
    integral equals the raw-density integral over the same set thanks to the
    monotonic change of variable).
    """
    import torch.distributions as D
    w = _norm_weight(weight)
    mix = D.Categorical(probs=w)
    comp = D.Normal(loc, scale)
    gmm = D.MixtureSameFamily(mix, comp)
    if target_space == "log":
        upper = torch.log1p(target + half_window)
        lower = torch.log1p((target - half_window).clamp_min(0.0))
    elif target_space == "raw":
        upper = target + half_window
        lower = (target - half_window).clamp_min(0.0)
    else:
        raise ValueError(f"target_space must be 'raw' or 'log', got {target_space!r}")
    return gmm.cdf(upper) - gmm.cdf(lower)


def time_at_t(pred_delta_hours: np.ndarray, gt_delta_hours: np.ndarray,
              t_minutes: int) -> float:
    diff_min = np.abs(pred_delta_hours - gt_delta_hours) * 60.0
    return float((diff_min <= t_minutes).mean())


def compute_time_scores(
    pred_dt: dict[str, np.ndarray],
    gt_delta_hours: np.ndarray,
    t_values_min: Iterable[int] = T_VALUES_DEFAULT_MIN,
    mass_within: dict[int, np.ndarray] | None = None,
    print_result: bool = True,
) -> dict[str, float]:
    """Compute multi-variant T@t and MAE/MedAE for several point estimators.

    ``pred_dt`` is a dict ``estimator_name -> predictions (N,)`` so we can
    compare {mean, mode, ...} side by side.

    ``mass_within`` is an optional ``{t_minutes: mass (N,)}`` map for the
    soft, distribution-based T@t (mean of the predicted mass within the
    GT ± t-minute window).
    """
    out: dict[str, float] = {}
    for est_name, pred in pred_dt.items():
        for t in t_values_min:
            out[f"t@{t}min/{est_name}"] = time_at_t(pred, gt_delta_hours, t)
        out[f"mae_min/{est_name}"] = float(np.mean(np.abs(pred - gt_delta_hours)) * 60.0)
        out[f"medae_min/{est_name}"] = float(np.median(np.abs(pred - gt_delta_hours)) * 60.0)
    if mass_within is not None:
        for t, mass in mass_within.items():
            out[f"t@{t}min/mass"] = float(np.mean(mass))
    if print_result:
        for k, v in out.items():
            print(f"  {k}: {v:.4f}")
    return out

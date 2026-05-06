from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Point
from torch import tensor


def sample_from_aoi_box(aoi_box, size=1000):
    """
    Sample random points within the AOI box.
    """
    minx, miny, maxx, maxy = aoi_box.bounds
    xs = np.random.uniform(minx, maxx, size)
    ys = np.random.uniform(miny, maxy, size)
    return np.column_stack((xs, ys))


def sample_from_aoi(aoi):
    """Rejection Sampling"""
    lat, lon = float("inf"), float("inf")
    while not aoi.contains(gpd.points_from_xy([lon], [lat])[0]):
        lat = np.random.uniform(aoi.bounds[1], aoi.bounds[3])
        lon = np.random.uniform(aoi.bounds[0], aoi.bounds[2])
    return lon, lat  # (x, y)


def latlon_to_xy(lats, lons, utm_crs="EPSG:4326"):
    """Convert latitude and longitude to x, y coordinates."""

    # Create GeoDataFrame with lat/lon points
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    # Convert to UTM coordinates
    gdf_utm = gdf.to_crs(utm_crs)

    # Extract x, y coordinates
    x = gdf_utm.geometry.x.values
    y = gdf_utm.geometry.y.values

    return x, y


def make_instance_dict(stay_df, categories):
    stay_anomalous = stay_df["stay_anomalous"].infer_objects(copy=False).fillna(False).astype(bool)
    return {
        "location": tensor(
            stay_df[["latitude", "longitude"]].values, dtype=torch.float
        ),
        "start": tensor(stay_df["start"].values, dtype=torch.float),
        "stop": tensor(stay_df["stop"].values, dtype=torch.float),
        "category": tensor(stay_df[categories].values, dtype=torch.float),
        "agent": tensor(stay_df["agent"].values, dtype=torch.int64),
        "padding_mask": torch.zeros(len(stay_df), dtype=torch.bool),
        "stay_anomalous": tensor(stay_anomalous.values, dtype=torch.bool),
    }


def determine_batch_size(dataset, model, device, no_grad, min_batch=16, max_batch=4096):
    """Find a conservative batch size that fits in GPU memory.

    This probes randomized batches to account for variable sequence lengths and,
    for training (`no_grad=False`), includes a backward pass so activation
    gradients are part of the estimate.
    """
    from utils.dataset import collate_fn

    if len(dataset) == 0:
        return 1

    model_input_keys = {"location", "start", "stop", "category", "agent", "padding_mask", "neighbor_padding_mask"}
    min_batch_size = max(1, int(min_batch))
    max_batch_size = max(min_batch_size, int(max_batch))
    # Training keeps gradient activations, so reserve more headroom there.
    safety_factor = 0.75 if not no_grad else 0.85
    random_trials_per_size = 1 if no_grad else 2
    stress_trials_per_size = 1 if no_grad else 2

    def build_descending_length_indices():
        """Return sample indices sorted by descending sequence length when available."""
        if not hasattr(dataset, "grouped_stays") or not hasattr(dataset, "agent_list"):
            return np.arange(len(dataset), dtype=np.int64)

        grouped_stays = getattr(dataset, "grouped_stays")
        agent_list = getattr(dataset, "agent_list")
        try:
            group_sizes = grouped_stays.size()
            lengths = np.array([int(group_sizes.get(agent, 0)) for agent in agent_list])
            return np.argsort(lengths)[::-1]
        except Exception:
            # Fall back to stable ordering if sequence lengths cannot be read.
            return np.arange(len(dataset), dtype=np.int64)

    descending_indices = build_descending_length_indices()

    def is_oom_error(exc):
        return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def cleanup_cuda_cache():
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    def make_random_batch(batch_size):
        replace = batch_size > len(dataset)
        indices = np.random.choice(len(dataset), size=batch_size, replace=replace)
        samples = [dataset[int(i)] for i in indices]
        return collate_fn(samples)

    def make_stress_batch(batch_size):
        if len(descending_indices) == 0:
            return make_random_batch(batch_size)

        if batch_size <= len(descending_indices):
            indices = descending_indices[:batch_size]
        else:
            repeats = int(np.ceil(batch_size / len(descending_indices)))
            indices = np.tile(descending_indices, repeats)[:batch_size]

        samples = [dataset[int(i)] for i in indices]
        return collate_fn(samples)

    def probe_once(batch_size, stress):
        batch = None
        model_inputs = None
        logits = None
        loss = None
        try:
            batch = make_stress_batch(batch_size) if stress else make_random_batch(batch_size)
            batch = {k: v.to(device) for k, v in batch.items()}
            model_inputs = {k: v for k, v in batch.items() if k in model_input_keys}

            if no_grad:
                with torch.no_grad():
                    logits, _, _, _ = model(**model_inputs)
            else:
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits, _, _, _ = model(**model_inputs)
                    # Scalar probe loss to materialize backward activations.
                    loss = logits.float().mean()
                loss.backward()
                model.zero_grad(set_to_none=True)
        except RuntimeError as exc:
            if is_oom_error(exc):
                cleanup_cuda_cache()
                return False
            raise
        finally:
            del loss, logits, model_inputs, batch

        return True

    def can_fit(batch_size):
        for _ in range(stress_trials_per_size):
            if not probe_once(batch_size, stress=True):
                return False

        for _ in range(random_trials_per_size):
            if not probe_once(batch_size, stress=False):
                return False

        return True

    # Exponential search to bracket the largest feasible batch size.
    if not can_fit(min_batch_size):
        candidate = min_batch_size
        while candidate > 1:
            next_candidate = max(1, candidate // 2)
            if can_fit(next_candidate):
                min_batch_size = next_candidate
                break
            candidate = next_candidate
        else:
            return 1

    low = min_batch_size
    high = min_batch_size * 2
    while high <= max_batch_size and can_fit(high):
        low = high
        high *= 2

    right = min(max_batch_size, high - 1)
    while low < right:
        mid = (low + right + 1) // 2
        if can_fit(mid):
            low = mid
        else:
            right = mid - 1

    return max(1, int(low * safety_factor))


def split_train_to_train_val(train_df, val_ratio=0.2, random_seed=42, from_tail=False):
    """Split the training data into training and validation sets by start.

    By default, val = earliest ``val_ratio`` of stays. When ``from_tail=True``,
    val = latest ``val_ratio`` of stays (useful for next-step prediction tasks
    where val should be temporally adjacent to the held-out test window).
    """
    np.random.seed(random_seed)
    if from_tail:
        cutoff_time = train_df["start"].quantile(1.0 - val_ratio)
        train_stays = train_df[train_df["start"] < cutoff_time].reset_index(drop=True)
        val_stays = train_df[train_df["start"] >= cutoff_time].reset_index(drop=True)
    else:
        cutoff_time = train_df["start"].quantile(val_ratio)
        train_stays = train_df[train_df["start"] >= cutoff_time].reset_index(drop=True)
        val_stays = train_df[train_df["start"] < cutoff_time].reset_index(drop=True)
    return train_stays, val_stays
    
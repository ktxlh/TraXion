"""Dataset path registry for TraXion.

All locations are derived from two environment variables:

    TRAXION_DATA_DIR    root holding the prepared per-dataset directories
                        (default: ``./data`` relative to repo root)
    TRAXION_RUNS_DIR    where checkpoints and W&B local files are written
                        (default: ``./runs``)

Per-dataset paths follow the convention produced by ``preprocess/prepare_*.py``::

    <data_root>/<dataset>/<dataset>_train.parquet
    <data_root>/<dataset>/<dataset>_test.parquet
    <data_root>/<dataset>/<dataset>_poi.parquet
    <data_root>/<dataset>/<dataset>_aoi_box.geojson
    <data_root>/<dataset>/<dataset>_dataset_metadata.parquet

The eight dataset entries below correspond one-to-one with the eight cohorts
reported in the paper (Table~1 / Appendix~B).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


_DATA_ROOT = os.environ.get("TRAXION_DATA_DIR", "./data")
_RUNS_ROOT = os.environ.get("TRAXION_RUNS_DIR", "./runs")


def _layout(name: str, n_agents: int, *, data_subdir: str | None = None,
            file_stem: str | None = None) -> dict:
    """Standard per-dataset path layout, fully derived from TRAXION_DATA_DIR."""
    sub = data_subdir or name
    stem = file_stem or name
    base = os.path.join(_DATA_ROOT, sub)
    return {
        "raw_train_data_path":   os.path.join(base, f"{stem}_train.parquet"),
        "raw_test_data_path":    os.path.join(base, f"{stem}_test.parquet"),
        "poi_path":              os.path.join(base, f"{stem}_poi.parquet"),
        "aoi_path":              os.path.join(base, f"{stem}_aoi_box.geojson"),
        "dataset_metadata_path": os.path.join(base, f"{stem}_dataset_metadata.parquet"),
        "train_data_path":       os.path.join(base, f"{stem}_train.parquet"),
        "test_data_path":        os.path.join(base, f"{stem}_test.parquet"),
        "training_log_dir":      _RUNS_ROOT,
        "default_n_agents":      n_agents,
    }


DATASET_PATHS = {
    # Mobility — anomaly detection
    "numosim":         _layout("numosim",        n_agents=200_000),
    "urban-berlin":    _layout("urban-berlin",   n_agents=1_000),
    "urban-atlanta":   _layout("urban-atlanta",  n_agents=1_000),

    # Mobility — next-POI / next-visit / social link
    "foursquare-tokyo":     _layout("foursquare-tokyo",     n_agents=7_456),
    "gowalla-stockholm-v1": _layout("gowalla-stockholm-v1", n_agents=7_759),
    "gowalla-austin-v1":    _layout("gowalla-austin-v1",    n_agents=5_397),

    # Non-mobility MESES instances
    "lanl-core":  _layout("lanl-core",  n_agents=1_598),
    "eicu-demo":  _layout("eicu-demo",  n_agents=2_492),
}

SUPPORTED_DATASETS = tuple(sorted(DATASET_PATHS.keys()))


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_train_data_path: str
    raw_test_data_path: str
    poi_path: str
    aoi_path: str
    dataset_metadata_path: str
    train_data_path: str
    test_data_path: str
    training_log_dir: str
    default_n_agents: int
    default_local_tz: str = "UTC"
    default_timestamp_step_size: float = 1 / 3600
    default_utm_crs: str = "EPSG:4326"


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. "
            f"Expected one of: {', '.join(SUPPORTED_DATASETS)}"
        )
    selected = DATASET_PATHS[dataset_name]
    return DatasetConfig(name=dataset_name, **selected)

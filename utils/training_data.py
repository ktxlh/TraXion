from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_stays(stays: pd.DataFrame, min_time: pd.Timestamp | str) -> pd.DataFrame:
    """Normalize raw stays schema into model-ready columns.

    Ensures `start` and `stop` exist and are float hours from a global origin.
    """
    stays = stays.copy()

    start_col = next(
        (col for col in ["start", "start_datetime", "start_timestamp"] if col in stays),
        None,
    )
    stop_col = next(
        (
            col
            for col in ["stop", "end_datetime", "stop_datetime", "stop_timestamp"]
            if col in stays
        ),
        None,
    )

    if start_col is None:
        raise KeyError(
            "Expected start time column in stays data. "
            f"Found columns: {list(stays.columns)}"
        )
    has_stop = stop_col is not None and stop_col != start_col

    rename_map = {start_col: "start"}
    if has_stop:
        rename_map[stop_col] = "stop"
    stays = stays.rename(columns=rename_map)
    if not has_stop:
        # Dataset has only start times (point check-ins) — use start as stop.
        stays["stop"] = stays["start"]
    if "agent_id" in stays.columns and "agent" not in stays.columns:
        stays = stays.rename(columns={"agent_id": "agent"})

    stays["start"] = pd.to_datetime(stays["start"], errors="coerce")
    stays["stop"] = pd.to_datetime(stays["stop"], errors="coerce")

    parsed_min_time = pd.to_datetime(min_time, errors="coerce")
    if pd.isna(parsed_min_time):
        raise ValueError("Provided min_time is invalid")
    min_time = pd.Timestamp(parsed_min_time)

    stays["start"] = (stays["start"] - min_time).dt.total_seconds() / 3600.0
    stays["stop"] = (stays["stop"] - min_time).dt.total_seconds() / 3600.0

    required_cols = ["latitude", "longitude", "start", "stop", "agent"]
    before = len(stays)
    stays = stays.dropna(subset=required_cols).reset_index(drop=True)
    after = len(stays)
    if after < before:
        print(f"Dropped {before - after:,} invalid stays with missing required fields")

    return stays


def ensure_category_columns(stays: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Ensure stay dataframe contains one-hot POI category columns."""
    stays = stays.copy()

    if all(category in stays.columns for category in categories):
        return stays

    if "act_types" not in stays.columns:
        raise KeyError(
            "Stays dataframe must include `act_types` to derive category features "
            f"when missing columns: {categories}"
        )

    exploded = stays["act_types"].explode()
    exploded = exploded.dropna().astype(int).apply(lambda value: f"cat_{value}")

    if exploded.empty:
        for category in categories:
            stays[category] = 0.0
        return stays

    dummies = pd.get_dummies(exploded)
    category_counts = dummies.groupby(level=0).sum()
    category_counts = category_counts.reindex(stays.index, fill_value=0)

    for category in categories:
        stays[category] = category_counts.get(category, 0)

    return stays


def normalize_and_validate_agents(
    train_stays: pd.DataFrame,
    val_stays: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize agent ids to contiguous ints and validate embedding vocab size."""
    train_stays = train_stays.copy()
    val_stays = val_stays.copy()

    all_agents = pd.concat([train_stays["agent"], val_stays["agent"]], ignore_index=True)
    if all_agents.isna().any():
        raise ValueError("`agent` column contains missing values after preprocessing")

    unique_agents = pd.Index(all_agents.unique())
    agent_to_idx = pd.Series(np.arange(len(unique_agents), dtype=np.int64), index=unique_agents)
    train_stays["agent"] = train_stays["agent"].map(agent_to_idx).astype(np.int64)
    val_stays["agent"] = val_stays["agent"].map(agent_to_idx).astype(np.int64)

    return train_stays, val_stays

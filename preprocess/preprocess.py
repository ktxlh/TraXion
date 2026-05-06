import argparse
import os
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from config import (
    DATASET_PATHS,
    get_dataset_config,
)


def join_poi_coordinates(poi_df, df_with_poi_id):
    """Exact match join"""
    print(f"Joining POI coordinates for {len(df_with_poi_id):,} records.")
    df_with_coords = df_with_poi_id.merge(
        poi_df[["poi_id", "longitude", "latitude", "act_types"]],
        on="poi_id",
        how="left",
        suffixes=("", "_poi"),
    )
    return df_with_coords


def _ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_raw_stays(raw_path):
    if raw_path.lower().endswith(".parquet"):
        return pd.read_parquet(raw_path)
    if raw_path.lower().endswith(".tsv"):
        df = pd.read_csv(raw_path, sep="\t")
        # Some Urban files use .tsv extension but are comma-delimited.
        if len(df.columns) == 1 and "," in str(df.columns[0]):
            df = pd.read_csv(raw_path)
        return df
    return pd.read_csv(raw_path)


def _maybe_rename(df, candidates, target):
    if target in df.columns:
        return df

    for col in candidates:
        if col in df.columns:
            return df.rename(columns={col: target})
    return df


def _normalize_raw_stays_schema(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = _maybe_rename(
        df,
        ["AgentID", "agent_id", "agentid", "user_id", "person_id", "uid"],
        "agent",
    )
    df = _maybe_rename(df, ["Latitude", "lat", "y"], "latitude")
    df = _maybe_rename(df, ["Longitude", "lon", "lng", "x"], "longitude")
    df = _maybe_rename(
        df,
        ["VenueId", "venue_id", "place_id", "location_id", "poi"],
        "poi_id",
    )
    df = _maybe_rename(df, ["VenueType", "venue_type", "venueType"], "venue_type")
    df = _maybe_rename(df, ["Label", "label", "is_anomaly", "anomalous"], "anomaly")
    df = _maybe_rename(
        df,
        ["Time", "start_time", "start_ts", "started_at", "start_datetime", "start_timestamp"],
        "start",
    )
    df = _maybe_rename(
        df,
        [
            "end",
            "end_time",
            "end_datetime",
            "stop_time",
            "stop_datetime",
            "stop_timestamp",
        ],
        "stop",
    )
    if "anomaly" in df.columns:
        df["anomaly"] = _normalize_binary_labels(df["anomaly"])
    return df


def _normalize_binary_labels(values):
    labels = pd.Series(values)
    if labels.dtype == bool:
        return labels.astype(int).tolist()

    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all():
        return (numeric > 0).astype(int).tolist()

    lowered = labels.astype(str).str.strip().str.lower()
    positives = {"1", "true", "t", "yes", "y", "anomaly", "anomalous", "outlier"}
    return lowered.isin(positives).astype(int).tolist()


def _ensure_poi_id(train_df, test_df):
    def _fill(df):
        df = df.copy()
        if "poi_id" not in df.columns:
            df["poi_id"] = pd.Series([None] * len(df), index=df.index)

        has_coords = "latitude" in df.columns and "longitude" in df.columns
        if not has_coords:
            raise ValueError("Stay data must include latitude/longitude to build POIs")

        missing = df["poi_id"].isna() | (df["poi_id"].astype(str).str.len() == 0)
        if missing.any():
            lat_key = df.loc[missing, "latitude"].round(5)
            lon_key = df.loc[missing, "longitude"].round(5)
            df.loc[missing, "poi_id"] = (
                "gen_"
                + lat_key.astype(str)
                + "_"
                + lon_key.astype(str)
            )
        return df

    return _fill(train_df), _fill(test_df)


def _build_poi_table(train_df, test_df):
    combined = pd.concat([train_df, test_df], ignore_index=True)
    grouped = (
        combined.groupby("poi_id", dropna=False)
        .agg(longitude=("longitude", "median"), latitude=("latitude", "median"))
        .reset_index()
    )

    if grouped.empty:
        raise ValueError("Cannot infer POI table from empty stay data")

    venue_col = "venue_type" if "venue_type" in combined.columns else None
    if venue_col is None:
        logging.warning("No venue type column found; assigning all POIs to unknown category")
        poi_category = pd.Series(["unknown"] * len(grouped), index=grouped.index)
    else:
        typed = combined[["poi_id", venue_col]].copy()
        typed[venue_col] = typed[venue_col].astype(str).str.strip()
        typed = typed[typed[venue_col] != ""]

        if typed.empty:
            logging.warning("Venue type column exists but has no valid values; using unknown category")
            poi_category = pd.Series(["unknown"] * len(grouped), index=grouped.index)
        else:
            # Use the most frequent VenueType per POI for deterministic category assignment.
            poi_to_category = typed.groupby("poi_id")[venue_col].agg(
                lambda s: s.value_counts().index[0]
            )
            poi_category = grouped["poi_id"].map(poi_to_category).fillna("unknown")

    category_values = sorted(pd.Series(poi_category).dropna().unique().tolist())
    if "unknown" not in category_values:
        category_values.append("unknown")

    cat_to_id = {cat: idx for idx, cat in enumerate(category_values)}
    grouped["venue_type"] = poi_category
    grouped["act_types"] = grouped["venue_type"].map(
        lambda value: [int(cat_to_id.get(value, cat_to_id["unknown"]))]
    )
    return grouped


def _resolve_time_column(df):
    for col in ("start_datetime", "start", "start_timestamp"):
        if col in df.columns:
            return col
    raise ValueError("No time column found. Expected one of: start_datetime, start, start_timestamp")


def infer_local_tz(df):
    lat = df["latitude"].dropna()
    lon = df["longitude"].dropna()
    if lat.empty or lon.empty:
        return "UTC"

    center_lat = float(lat.median())
    center_lon = float(lon.median())

    if -126 <= center_lon <= -100:
        return "America/Los_Angeles"
    if -100 < center_lon <= -67:
        return "America/New_York"
    if 5 <= center_lon <= 20 and 45 <= center_lat <= 56:
        return "Europe/Berlin"
    if 110 <= center_lon <= 125 and 35 <= center_lat <= 45:
        return "Asia/Shanghai"
    return "UTC"


def infer_timestamp_step_size(df):
    time_col = _resolve_time_column(df)
    ts = pd.to_datetime(df[time_col], errors="coerce")
    ts = ts.dropna().drop_duplicates().sort_values()
    if len(ts) < 2:
        return 1 / 3600

    deltas = ts.diff().dropna().dt.total_seconds()
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.empty:
        return 1 / 3600

    min_step_seconds = float(positive_deltas.min())
    return min_step_seconds / 3600


def infer_utm_crs(df, sample_size=10000):
    coords = df[["longitude", "latitude"]].dropna()
    if coords.empty:
        logging.warning("No valid coordinates found to infer UTM CRS. Defaulting to EPSG:4326.")
        return "EPSG:4326"

    coords = coords.sample(min(sample_size, len(coords)), random_state=0)
    points = gpd.GeoSeries(
        gpd.points_from_xy(coords["longitude"], coords["latitude"]),
        crs="EPSG:4326",
    )
    utm = points.estimate_utm_crs()
    return utm.to_string() if utm is not None else "EPSG:4326"


def infer_n_agents(df):
    if "agent" in df.columns:
        return int(df["agent"].nunique(dropna=True))
    if "agent_id" in df.columns:
        return int(df["agent_id"].nunique(dropna=True))
    return 0


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Preprocess mobility dataset with POI join and metadata inference")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_PATHS.keys()),
        help="Dataset key to preprocess",
    )
    return parser.parse_args(args_list)


if __name__ == "__main__":
    args = parse_args()
    dataset_cfg = get_dataset_config(args.dataset)

    raw_train_data_path = dataset_cfg.raw_train_data_path
    raw_test_data_path = dataset_cfg.raw_test_data_path
    poi_path = dataset_cfg.poi_path
    aoi_path = dataset_cfg.aoi_path
    dataset_metadata_path = dataset_cfg.dataset_metadata_path
    train_data_path = dataset_cfg.train_data_path
    test_data_path = dataset_cfg.test_data_path

    print(f"Using dataset: {args.dataset}")

    # Load and normalize training/testing data
    train_df = _normalize_raw_stays_schema(_read_raw_stays(raw_train_data_path))
    test_df = _normalize_raw_stays_schema(_read_raw_stays(raw_test_data_path))

    required_stay_cols = ["longitude", "latitude"]
    missing_cols = [col for col in required_stay_cols if col not in train_df.columns or col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required stay columns after normalization: {missing_cols}")

    if os.path.isfile(poi_path):
        poi_df = pd.read_parquet(poi_path)
    else:
        print(f"POI file not found at {poi_path}; inferring POIs from stay points")
        train_df, test_df = _ensure_poi_id(train_df, test_df)
        poi_df = _build_poi_table(train_df, test_df)
        _ensure_parent_dir(poi_path)
        poi_df.to_parquet(poi_path, index=False)
        print(f"Inferred POI table saved to {poi_path}")

    train_df, test_df = _ensure_poi_id(train_df, test_df)

    # Use POI ID to get coordinates
    train_df = join_poi_coordinates(poi_df, train_df)
    test_df = join_poi_coordinates(poi_df, test_df)

    # Save the updated dataframes back to parquet at `processed/` directory
    _ensure_parent_dir(train_data_path)
    _ensure_parent_dir(test_data_path)
    train_df.to_parquet(train_data_path, index=False)
    test_df.to_parquet(test_data_path, index=False)
    print(f"Saved processed training data to {train_data_path}")
    print(f"Saved processed testing data to {test_data_path}")

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    if not os.path.isfile(aoi_path):
        # Compute the total bounds (minx, miny, maxx, maxy)
        minx, miny = combined_df[["longitude", "latitude"]].min().values
        maxx, maxy = combined_df[["longitude", "latitude"]].max().values
        aoi_box = box(minx, miny, maxx, maxy)
        print("AOI Bounding Box (UTM coordinates):")
        print(aoi_box)

        # Save AOI box to GeoJSON
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_box], crs="EPSG:4326")
        _ensure_parent_dir(aoi_path)
        aoi_gdf.to_file(aoi_path, driver="GeoJSON")
        print(f"AOI box saved to {aoi_path}")

    else:
        print(f"AOI box already exists at {aoi_path}, skipping computation.")

    if not os.path.isfile(dataset_metadata_path):
        time_col = _resolve_time_column(combined_df)
        min_time = pd.to_datetime(combined_df[time_col], errors="coerce").min()
        inferred_local_tz = infer_local_tz(combined_df)
        inferred_timestamp_step_size = infer_timestamp_step_size(combined_df)
        inferred_utm_crs = infer_utm_crs(combined_df)
        inferred_n_agents = infer_n_agents(combined_df)

        print(f"Minimum timestamp across datasets: {min_time}")
        print(f"Inferred local_tz: {inferred_local_tz}")
        print(f"Inferred timestamp_step_size: {inferred_timestamp_step_size}")
        print(f"Inferred utm_crs: {inferred_utm_crs}")
        print(f"Inferred n_agents: {inferred_n_agents}")

        metadata_df = pd.DataFrame(
            {
                "min_time": [min_time],
                "local_tz": [inferred_local_tz],
                "timestamp_step_size": [inferred_timestamp_step_size],
                "utm_crs": [inferred_utm_crs],
                "n_agents": [inferred_n_agents],
            }
        )
        _ensure_parent_dir(dataset_metadata_path)
        metadata_df.to_parquet(dataset_metadata_path, index=False)
        print(f"Dataset metadata saved to {dataset_metadata_path}")

    else:
        print(
            f"Dataset metadata file already exists at {dataset_metadata_path}, skipping computation."
        )

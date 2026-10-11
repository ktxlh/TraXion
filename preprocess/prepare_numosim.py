"""Prepare NUMOSIM-LA for TraXion.

Input: the NUMOSIM Los Angeles release (OSF, osf.io/sjyfr), read from --raw-dir:
  stay_points_train.parquet           one month of stays for all 200,000 agents
  stay_points_test_anomalous.parquet  the following month, with injected anomalies
                                      (`anomaly`, `anomaly_type` columns)
  poi.parquet                         poi_id, name, latitude, longitude, act_types

Mapping to TraXion's (agent, poi, time, category) schema:
  agent      = agent_id (simulated agent)
  poi_id     = poi_id
  longitude,
  latitude   = POI coordinates in WGS84 degrees (EPSG:4326), joined on poi_id
  start      = start_datetime
  stop       = end_datetime
  act_types  = the POI's list of activity-type codes
  anomaly    = the benchmark's stay-level label (test split only)

Cohort and split: all agents and the release's own train/test partition. No
filtering, resampling or coordinate scaling. Row order follows the raw files.

Output (to --out-dir):
  numosim_train.parquet, numosim_test.parquet  raw stays + longitude, latitude, act_types
  numosim_poi.parquet                          copy of the release's poi.parquet
  numosim_aoi_box.geojson                      bounding box of all stay coordinates
  numosim_dataset_metadata.parquet             min_time, n_agents

The metadata file holds only min_time and n_agents. The loader
(utils/file_io.load_dataset_metadata) fills local_tz, timestamp_step_size and
utm_crs from the DatasetConfig defaults ("UTC", 1/3600 h, "EPSG:4326"), so
coordinates stay in degrees.

Usage::

    NUMOSIM_RAW_DIR=/path/to/NUMOSIM/LA python -m preprocess.prepare_numosim
"""

import argparse
import os
import shutil

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

_DATA_ROOT = os.environ.get("TRAXION_DATA_DIR", "./data")
RAW_DIR = os.environ.get("NUMOSIM_RAW_DIR", os.path.join(_DATA_ROOT, "numosim_raw"))
OUT_DIR = os.path.join(_DATA_ROOT, "numosim")
TAG = "numosim"


def join_poi_coordinates(poi_df, df_with_poi_id):
    """Left join on poi_id; keeps every stay and the stay order."""
    print(f"Joining POI coordinates for {len(df_with_poi_id):,} records.")
    df_with_coords = df_with_poi_id.merge(
        poi_df[["poi_id", "longitude", "latitude", "act_types"]],
        on="poi_id",
        how="left",
        suffixes=("", "_poi"),
    )
    n_unmatched = int(df_with_coords["latitude"].isna().sum())
    if n_unmatched:
        print(f"  {n_unmatched:,} stays have no matching POI (NaN coordinates)")
    return df_with_coords


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", default=RAW_DIR,
                   help="directory holding the NUMOSIM-LA release parquet files")
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="directory for the prepared files")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    raw_train_path = os.path.join(args.raw_dir, "stay_points_train.parquet")
    raw_test_path = os.path.join(args.raw_dir, "stay_points_test_anomalous.parquet")
    raw_poi_path = os.path.join(args.raw_dir, "poi.parquet")

    out = lambda suffix: os.path.join(args.out_dir, f"{TAG}_{suffix}")
    train_path = out("train.parquet")
    test_path = out("test.parquet")
    poi_path = out("poi.parquet")
    aoi_path = out("aoi_box.geojson")
    metadata_path = out("dataset_metadata.parquet")
    os.makedirs(args.out_dir, exist_ok=True)

    poi_df = pd.read_parquet(raw_poi_path)
    train_df = pd.read_parquet(raw_train_path)
    test_df = pd.read_parquet(raw_test_path)

    train_df = join_poi_coordinates(poi_df, train_df)
    test_df = join_poi_coordinates(poi_df, test_df)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Saved {train_path} ({len(train_df):,} rows)")
    print(f"Saved {test_path} ({len(test_df):,} rows)")

    # The POI table is the release file, unchanged.
    shutil.copyfile(raw_poi_path, poi_path)
    print(f"Saved {poi_path} ({len(poi_df):,} POIs)")

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # AOI: axis-aligned bounding box of all stay coordinates (degrees).
    minx, miny = combined_df[["longitude", "latitude"]].min().values
    maxx, maxy = combined_df[["longitude", "latitude"]].max().values
    aoi_box = box(minx, miny, maxx, maxy)
    gpd.GeoDataFrame(geometry=[aoi_box], crs="EPSG:4326").to_file(aoi_path, driver="GeoJSON")
    print(f"Saved {aoi_path}: {aoi_box}")

    min_time = combined_df["start_datetime"].min()
    n_agents = int(combined_df["agent_id"].nunique())
    pd.DataFrame({"min_time": [min_time], "n_agents": [n_agents]}).to_parquet(
        metadata_path, index=False
    )
    print(f"Saved {metadata_path}: min_time={min_time}, n_agents={n_agents:,}")

    n_anom_agents = test_df.loc[test_df["anomaly"].astype(bool), "agent_id"].nunique()
    print(f"Events: {len(combined_df):,}  agents: {n_agents:,}  "
          f"POIs visited: {combined_df['poi_id'].nunique():,}  "
          f"anomalous test agents: {n_anom_agents:,}")


if __name__ == "__main__":
    main()

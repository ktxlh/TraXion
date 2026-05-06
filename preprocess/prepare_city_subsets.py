"""Prepare city-based subsets of Foursquare and Gowalla for Clique pre-training.

Instead of k-core social filtering, this script extracts subsets by metropolitan
area: venues are snapped to the nearest large city (pop >= 1M within 100km, or
pop >= 300k within 50km), then only checkins in the target metro are kept.

City subsets:
  Foursquare: Tokyo, Istanbul, Kuala Lumpur
  Gowalla:    Stockholm, Austin, SF Bay Area

Output structure (per subset):
  {out_dir}/{prefix}_train.parquet
  {out_dir}/{prefix}_test.parquet
  {out_dir}/{prefix}_poi.parquet
  {out_dir}/{prefix}_aoi_box.geojson
  {out_dir}/{prefix}_dataset_metadata.parquet
  {out_dir}/{prefix}_social_edges.csv
"""

import csv
import gzip
import json
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import reverse_geocode
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Metro-snapping helpers
# ---------------------------------------------------------------------------

def _haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def _load_city_db():
    """Load the reverse_geocode city database and build mega/large arrays."""
    gz_path = os.path.join(os.path.dirname(reverse_geocode.__file__), "geocode.gz")
    with gzip.open(gz_path, "rt") as f:
        all_cities = json.loads(f.read())
    countries_path = os.path.join(os.path.dirname(reverse_geocode.__file__), "countries.csv")
    cc_to_name = {}
    with open(countries_path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                cc_to_name[row[0]] = row[1]

    mega = [c for c in all_cities if c.get("population", 0) >= 1_000_000]
    large = [c for c in all_cities if 300_000 <= c.get("population", 0) < 1_000_000]

    def _arrays(cities):
        lats = np.array([c["latitude"] for c in cities])
        lons = np.array([c["longitude"] for c in cities])
        labels = [f"{c['city']}, {cc_to_name.get(c.get('country_code', ''), '?')}" for c in cities]
        return lats, lons, labels

    return _arrays(mega), _arrays(large)


# Manual merges for subdivisions that appear as separate mega-cities
METRO_MERGES = {
    "Manhattan, United States": "New York, United States",
    "Brooklyn, United States": "New York, United States",
    "Queens, United States": "New York, United States",
    "The Bronx, United States": "New York, United States",
    "New York City, United States": "New York, United States",
    "San Jose, United States": "SF Bay Area, United States",
    "San Francisco, United States": "SF Bay Area, United States",
    "Oakland, United States": "SF Bay Area, United States",
    "Yokohama, Japan": "Tokyo, Japan",
    "Kawasaki, Japan": "Tokyo, Japan",
    "Saitama, Japan": "Tokyo, Japan",
    "Chiba, Japan": "Tokyo, Japan",
}


def snap_venues_to_metro(lats, lons, mega, large):
    """Return a metro label for each (lat, lon) pair."""
    mega_lats, mega_lons, mega_labels = mega
    large_lats, large_lons, large_labels = large
    metros = []
    for i in range(len(lats)):
        dists = _haversine_vec(lats[i], lons[i], mega_lats, mega_lons)
        idx = np.argmin(dists)
        if dists[idx] < 100:
            label = mega_labels[idx]
            metros.append(METRO_MERGES.get(label, label))
            continue
        dists2 = _haversine_vec(lats[i], lons[i], large_lats, large_lons)
        idx2 = np.argmin(dists2)
        if dists2[idx2] < 50:
            label = large_labels[idx2]
            metros.append(METRO_MERGES.get(label, label))
        else:
            metros.append("_remote_")
    return metros


# ---------------------------------------------------------------------------
# Shared processing helpers
# ---------------------------------------------------------------------------

def build_metadata_and_aoi(train_df, test_df, out_dir, prefix):
    """Build and save dataset metadata + AOI bounding box."""
    combined = pd.concat([train_df, test_df], ignore_index=True)

    min_time = pd.to_datetime(train_df["start"]).min()
    ts_unique = pd.to_datetime(train_df["start"]).drop_duplicates().sort_values()
    deltas = ts_unique.diff().dropna().dt.total_seconds()
    pos = deltas[deltas > 0]
    min_step_s = float(pos.min()) if len(pos) > 0 else 1.0
    timestamp_step_size = min_step_s / 3600.0

    sample_n = min(10_000, len(combined))
    pts = gpd.GeoSeries(
        gpd.points_from_xy(
            combined["longitude"].sample(sample_n, random_state=0),
            combined["latitude"].sample(sample_n, random_state=0),
        ),
        crs="EPSG:4326",
    )
    utm = pts.estimate_utm_crs()
    utm_crs = utm.to_string() if utm is not None else "EPSG:4326"

    n_agents = int(train_df["agent"].nunique())

    meta = pd.DataFrame({
        "min_time": [min_time],
        "local_tz": ["UTC"],
        "timestamp_step_size": [timestamp_step_size],
        "utm_crs": [utm_crs],
        "n_agents": [n_agents],
    })
    meta_path = os.path.join(out_dir, f"{prefix}_dataset_metadata.parquet")
    meta.to_parquet(meta_path, index=False)
    print(f"  Saved metadata → {meta_path}  (n_agents={n_agents})")

    minx, maxx = combined["longitude"].min(), combined["longitude"].max()
    miny, maxy = combined["latitude"].min(), combined["latitude"].max()
    aoi = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    aoi_path = os.path.join(out_dir, f"{prefix}_aoi_box.geojson")
    aoi.to_file(aoi_path, driver="GeoJSON")
    print(f"  Saved AOI → {aoi_path}")

    return n_agents


def temporal_split_and_filter(checkins, base_cols, min_checkins=5):
    """90/10 temporal split, min-checkin filter, return (train, test)."""
    checkins = checkins.sort_values("start").reset_index(drop=True)
    split_idx = int(len(checkins) * 0.9)
    split_time = checkins.loc[split_idx, "start"]
    print(f"  Split time: {split_time}")

    train = checkins[checkins["start"] < split_time][base_cols].copy()
    test = checkins[checkins["start"] >= split_time][base_cols].copy()
    test = test[test["agent"].isin(set(train["agent"].unique()))].copy()

    agent_counts = train.groupby("agent").size()
    valid = set(agent_counts[agent_counts >= min_checkins].index)
    train = train[train["agent"].isin(valid)].copy()
    test = test[test["agent"].isin(valid)].copy()

    print(f"  Train: {len(train):,} checkins, {train['agent'].nunique():,} agents")
    print(f"  Test:  {len(test):,} checkins, {test['agent'].nunique():,} agents")
    return train, test


# ---------------------------------------------------------------------------
# Foursquare city-subset processing
# ---------------------------------------------------------------------------

def process_foursquare_city(metro_name, prefix, out_dir, checkins, poi_df, edges, mega, large):
    """Extract a Foursquare city subset."""
    print(f"\n{'#' * 70}")
    print(f"# Foursquare — {metro_name}")
    print(f"{'#' * 70}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Snap venues to metros ---
    venue_coords = checkins.groupby("venue_id").agg(
        lat=("latitude", "median"), lon=("longitude", "median")
    ).reset_index()
    print(f"  Snapping {len(venue_coords):,} venues to metros...")
    t0 = time.time()
    venue_coords["metro"] = snap_venues_to_metro(
        venue_coords["lat"].values, venue_coords["lon"].values, mega, large
    )
    print(f"  Done ({time.time() - t0:.1f}s)")

    metro_venues = set(venue_coords[venue_coords["metro"] == metro_name]["venue_id"])
    print(f"  Venues in {metro_name}: {len(metro_venues):,}")

    city = checkins[checkins["venue_id"].isin(metro_venues)].copy()
    print(f"  Checkins in city: {len(city):,}, users: {city['user_id'].nunique():,}")

    # --- POI join ---
    city = city.merge(
        poi_df[["venue_id", "latitude", "longitude", "category"]],
        on="venue_id", how="inner", suffixes=("_ck", ""),
    )
    # Prefer POI-level coords
    for col in ("latitude", "longitude"):
        if f"{col}_ck" in city.columns:
            city.drop(columns=[f"{col}_ck"], inplace=True)

    # --- Category encoding ---
    city["category"] = city["category"].fillna("unknown").str.strip()
    all_cats = sorted(city["category"].unique().tolist())
    if "unknown" not in all_cats:
        all_cats.append("unknown")
    cat_to_id = {c: i for i, c in enumerate(all_cats)}
    print(f"  Unique categories: {len(cat_to_id)}")

    # --- POI table ---
    poi_table = (
        city.groupby("venue_id")
        .agg(
            longitude=("longitude", "median"),
            latitude=("latitude", "median"),
            category=("category", lambda s: s.mode().iloc[0] if len(s) > 0 else "unknown"),
        )
        .reset_index()
        .rename(columns={"venue_id": "poi_id"})
    )
    poi_table["venue_type"] = poi_table["category"]
    poi_table["act_types"] = poi_table["category"].map(
        lambda c: [cat_to_id.get(c, cat_to_id["unknown"])]
    )
    poi_table = poi_table[["poi_id", "longitude", "latitude", "venue_type", "act_types"]].copy()
    poi_path = os.path.join(out_dir, f"{prefix}_poi.parquet")
    poi_table.to_parquet(poi_path, index=False)
    print(f"  Saved {len(poi_table):,} POIs → {poi_path}")

    # --- Prepare checkin columns ---
    venue_to_act = poi_table.set_index("poi_id")["act_types"]
    city["act_types"] = city["venue_id"].map(venue_to_act)
    city = city.rename(columns={"user_id": "agent", "venue_id": "poi_id", "local_dt": "start"})
    city["stop"] = city["start"]
    city["cat_0"] = city["act_types"].apply(
        lambda x: float(x[0]) if isinstance(x, list) and len(x) > 0 else 0.0
    )

    base_cols = ["agent", "longitude", "latitude", "poi_id", "act_types", "cat_0", "start", "stop"]
    train, test = temporal_split_and_filter(city, base_cols)

    train_path = os.path.join(out_dir, f"{prefix}_train.parquet")
    test_path = os.path.join(out_dir, f"{prefix}_test.parquet")
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"  Saved train → {train_path}")
    print(f"  Saved test  → {test_path}")

    # --- Social edges ---
    city_users = set(train["agent"].unique()) | set(test["agent"].unique())
    city_edges = edges[(edges["u"].isin(city_users)) & (edges["v"].isin(city_users))].copy()
    edges_path = os.path.join(out_dir, f"{prefix}_social_edges.csv")
    city_edges.to_csv(edges_path, index=False)
    print(f"  Social edges among city users: {len(city_edges):,} → {edges_path}")

    n_agents = build_metadata_and_aoi(train, test, out_dir, prefix)
    return n_agents


# ---------------------------------------------------------------------------
# Gowalla city-subset processing
# ---------------------------------------------------------------------------

def process_gowalla_city(metro_name, prefix, out_dir, checkins, edges, mega, large):
    """Extract a Gowalla city subset."""
    print(f"\n{'#' * 70}")
    print(f"# Gowalla — {metro_name}")
    print(f"{'#' * 70}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Snap locations to metros ---
    loc_coords = checkins.groupby("location_id").agg(
        lat=("latitude", "median"), lon=("longitude", "median")
    ).reset_index()
    print(f"  Snapping {len(loc_coords):,} locations to metros...")
    t0 = time.time()
    loc_coords["metro"] = snap_venues_to_metro(
        loc_coords["lat"].values, loc_coords["lon"].values, mega, large
    )
    print(f"  Done ({time.time() - t0:.1f}s)")

    metro_locs = set(loc_coords[loc_coords["metro"] == metro_name]["location_id"])
    print(f"  Locations in {metro_name}: {len(metro_locs):,}")

    city = checkins[checkins["location_id"].isin(metro_locs)].copy()
    print(f"  Checkins in city: {len(city):,}, users: {city['user_id'].nunique():,}")

    # --- POI table (no categories for Gowalla) ---
    poi_table = (
        city.groupby("location_id")
        .agg(longitude=("longitude", "median"), latitude=("latitude", "median"))
        .reset_index()
        .rename(columns={"location_id": "poi_id"})
    )
    poi_table["poi_id"] = poi_table["poi_id"].astype(str)
    poi_table["venue_type"] = "unknown"
    poi_table["act_types"] = [[0]] * len(poi_table)
    poi_path = os.path.join(out_dir, f"{prefix}_poi.parquet")
    poi_table.to_parquet(poi_path, index=False)
    print(f"  Saved {len(poi_table):,} POIs → {poi_path}")

    # --- Prepare checkin columns ---
    city["poi_id"] = city["location_id"].astype(str)
    city["stop"] = city["start"]
    city["act_types"] = [[0]] * len(city)
    city["cat_0"] = 0.0
    city = city.rename(columns={"user_id": "agent"})

    base_cols = ["agent", "longitude", "latitude", "poi_id", "act_types", "cat_0", "start", "stop"]
    train, test = temporal_split_and_filter(city, base_cols)

    train_path = os.path.join(out_dir, f"{prefix}_train.parquet")
    test_path = os.path.join(out_dir, f"{prefix}_test.parquet")
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"  Saved train → {train_path}")
    print(f"  Saved test  → {test_path}")

    # --- Social edges ---
    city_users = set(train["agent"].unique()) | set(test["agent"].unique())
    city_edges = edges[(edges["u"].isin(city_users)) & (edges["v"].isin(city_users))].copy()
    edges_path = os.path.join(out_dir, f"{prefix}_social_edges.csv")
    city_edges.to_csv(edges_path, index=False)
    print(f"  Social edges among city users: {len(city_edges):,} → {edges_path}")

    n_agents = build_metadata_and_aoi(train, test, out_dir, prefix)
    return n_agents


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("Loading city database for metro snapping...")
    mega, large = _load_city_db()
    print(f"  Mega cities (pop >= 1M): {len(mega[0])}")
    print(f"  Large cities (300k-1M): {len(large[0])}")

    # ---------------------------------------------------------------
    # Foursquare
    # ---------------------------------------------------------------
    _DATA = os.environ.get("TRAXION_DATA_DIR", "./data")
    FS_RAW = os.environ.get("FOURSQUARE_RAW_DIR", os.path.join(_DATA, "foursquare_raw"))
    FS_OUT_BASE = os.path.join(_DATA, "foursquare_processed")

    print("\n" + "=" * 70)
    print("Loading Foursquare raw data...")
    print("=" * 70)
    t0 = time.time()

    fs_checkins = pd.read_csv(
        os.path.join(FS_RAW, "dataset_WWW_Checkins_anonymized.txt"),
        sep="\t", header=None,
        names=["user_id", "venue_id", "utc_time", "tz_offset_min"],
        dtype={"user_id": np.int64, "venue_id": str, "tz_offset_min": np.int32},
    )
    print(f"  Checkins: {len(fs_checkins):,} ({time.time() - t0:.1f}s)")

    # Parse timestamps → local time (coerce bad rows)
    fs_checkins["utc_dt"] = pd.to_datetime(
        fs_checkins["utc_time"], format="%a %b %d %H:%M:%S +0000 %Y",
        utc=True, errors="coerce",
    )
    fs_checkins = fs_checkins.dropna(subset=["utc_dt"]).reset_index(drop=True)
    fs_checkins["local_dt"] = fs_checkins["utc_dt"] + pd.to_timedelta(
        fs_checkins["tz_offset_min"], unit="m"
    )
    fs_checkins["local_dt"] = fs_checkins["local_dt"].dt.tz_localize(None)

    # Load POI metadata
    print("  Loading POIs...")
    t0 = time.time()
    needed = set(fs_checkins["venue_id"].unique())
    poi_rows = []
    with open(os.path.join(FS_RAW, "raw_POIs.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5 and parts[0] in needed:
                poi_rows.append(parts[:5])
    fs_pois = pd.DataFrame(poi_rows, columns=["venue_id", "latitude", "longitude", "category", "country"])
    fs_pois["latitude"] = pd.to_numeric(fs_pois["latitude"], errors="coerce")
    fs_pois["longitude"] = pd.to_numeric(fs_pois["longitude"], errors="coerce")
    fs_pois = fs_pois.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset="venue_id")
    print(f"  POIs: {len(fs_pois):,} ({time.time() - t0:.1f}s)")

    # Add coords to checkins for metro snapping (keep venue_id for POI join later)
    fs_checkins = fs_checkins.merge(
        fs_pois[["venue_id", "latitude", "longitude"]], on="venue_id", how="inner"
    )
    fs_checkins = fs_checkins[
        fs_checkins["latitude"].between(-90, 90)
        & fs_checkins["longitude"].between(-180, 180)
        & (fs_checkins["latitude"].abs() > 0.1)
    ].copy()
    fs_checkins = fs_checkins.sort_values(["user_id", "local_dt"]).reset_index(drop=True)
    print(f"  Checkins after join+filter: {len(fs_checkins):,}")

    # Load social edges
    fs_edges = pd.read_csv(
        os.path.join(FS_RAW, "dataset_WWW_friendship_new.txt"),
        sep="\t", header=None, names=["u", "v"],
    )
    print(f"  Social edges: {len(fs_edges):,}")

    FS_CITIES = [
        ("Tokyo, Japan", "foursquare-tokyo", os.path.join(_DATA, "foursquare-tokyo")),
    ]

    fs_n_agents = {}
    for metro_name, prefix, out_dir in FS_CITIES:
        n = process_foursquare_city(
            metro_name, prefix, out_dir, fs_checkins, fs_pois, fs_edges, mega, large
        )
        fs_n_agents[prefix] = n

    # ---------------------------------------------------------------
    # Gowalla
    # ---------------------------------------------------------------
    GW_RAW = os.environ.get("GOWALLA_RAW_DIR", os.path.join(_DATA, "gowalla_raw"))
    GW_OUT_BASE = os.path.join(_DATA, "gowalla_processed")

    print("\n" + "=" * 70)
    print("Loading Gowalla raw data...")
    print("=" * 70)
    t0 = time.time()

    gw_checkins = pd.read_csv(
        os.path.join(GW_RAW, "loc-gowalla_totalCheckins.txt"),
        sep="\t", header=None,
        names=["user_id", "timestamp", "latitude", "longitude", "location_id"],
        dtype={"user_id": np.int64, "location_id": np.int64,
               "latitude": np.float64, "longitude": np.float64},
    )
    print(f"  Checkins: {len(gw_checkins):,} ({time.time() - t0:.1f}s)")

    gw_checkins["start"] = pd.to_datetime(
        gw_checkins["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", utc=True
    )
    gw_checkins["start"] = gw_checkins["start"].dt.tz_localize(None)
    gw_checkins = gw_checkins.dropna(subset=["start"]).reset_index(drop=True)
    gw_checkins = gw_checkins[
        gw_checkins["latitude"].between(-90, 90)
        & gw_checkins["longitude"].between(-180, 180)
        & (gw_checkins["latitude"].abs() > 0.1)
    ].copy()
    gw_checkins = gw_checkins.sort_values(["user_id", "start"]).reset_index(drop=True)
    print(f"  Checkins after filter: {len(gw_checkins):,}")

    gw_edges = pd.read_csv(
        os.path.join(GW_RAW, "loc-gowalla_edges.txt"),
        sep="\t", header=None, names=["u", "v"],
    )
    print(f"  Social edges: {len(gw_edges):,}")

    GW_CITIES = [
        ("Stockholm, Sweden",         "gowalla-stockholm-v1", os.path.join(_DATA, "gowalla-stockholm-v1")),
        ("Austin, United States",     "gowalla-austin-v1",    os.path.join(_DATA, "gowalla-austin-v1")),
    ]

    gw_n_agents = {}
    for metro_name, prefix, out_dir in GW_CITIES:
        n = process_gowalla_city(
            metro_name, prefix, out_dir, gw_checkins, gw_edges, mega, large
        )
        gw_n_agents[prefix] = n

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — default_n_agents for config.py")
    print("=" * 70)
    for name, n in {**fs_n_agents, **gw_n_agents}.items():
        print(f"  {name}: {n:,}")
    print("\nAll done!")

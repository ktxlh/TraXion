"""Prepare LANL Unified Host/Network dataset for Clique pre-training.

Cross-domain target: lateral-movement / insider-threat detection.

Mapping to Clique's (agent, poi, time, category) schema:
  agent      = src_user (U####@DOM#) performing the authentication
  poi_id     = dst_computer (C####) being accessed
  longitude,
  latitude   = 2D UMAP embedding of the user x host bipartite access matrix
  start      = synthetic datetime derived from auth time (seconds since day 0)
  stop       = start (point events, same convention as Foursquare/Gowalla)
  venue_type = host category string inferred from dominant auth_type at the host
  act_types  = list[int] encoding of the host category (one entry per POI)
  anomaly    = 1 iff the (time, user, src, dst) tuple appears in redteam.txt

Cohort (target ~3-5k agents, ~1-2M events, matches numosim-core scale):
  - All users that appear as user in redteam.txt (attack users, ~99)
  - Random sample of additional users until we hit the agent cap
  - Per-user cap: keep at most MAX_EVENTS_PER_AGENT most-recent events

Split:
  Temporal: first 70% of the cohort event stream = train, last 30% = test.
  Red-team events overwhelmingly land in days 2-27, so both splits carry
  anomalies; we report visit-level AP/Max F1 on test.

Output (to OUT_DIR):
  lanl_train.parquet, lanl_test.parquet, lanl_poi.parquet,
  lanl_aoi_box.geojson, lanl_dataset_metadata.parquet
"""

import argparse
import gzip
import os
import time as time_mod
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import umap
from scipy.sparse import csr_matrix
from shapely.geometry import box

_DATA = os.environ.get("TRAXION_DATA_DIR", "./data")
RAW_DIR = os.environ.get("LANL_RAW_DIR", os.path.join(_DATA, "lanl_raw"))
OUT_DIR = os.path.join(_DATA, "lanl-core")
AUTH_FILE = os.path.join(RAW_DIR, "auth.txt.gz")
REDTEAM_FILE = os.path.join(RAW_DIR, "redteam.txt.gz")

AUTH_COLS = [
    "time", "src_user", "dst_user",
    "src_computer", "dst_computer",
    "auth_type", "logon_type", "auth_orientation", "success",
]

# Synthetic base date so pandas Timestamp is well-formed. LANL times are
# seconds since an arbitrary start-of-dataset epoch, not a real calendar.
BASE_DATETIME = pd.Timestamp("2015-01-01 00:00:00")

# Cohort knobs
TARGET_N_AGENTS = 4000
MAX_EVENTS_PER_AGENT = 500
MIN_EVENTS_PER_AGENT = 20  # drop very inactive users
# Temporal split cutoff (seconds since the LANL epoch). Day 20 puts ~half of
# the red-team window (days 2-30) in train and the other half in test.
SPLIT_CUTOFF_SECONDS = 20 * 86400
RNG_SEED = 42


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--auth-file", default=AUTH_FILE,
                   help="path to auth.txt.gz")
    p.add_argument("--redteam-file", default=REDTEAM_FILE,
                   help="path to redteam.txt.gz")
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="directory for processed output")
    p.add_argument("--target-agents", type=int, default=TARGET_N_AGENTS,
                   help="target total number of agents (attack + benign)")
    p.add_argument("--max-events-per-agent", type=int, default=MAX_EVENTS_PER_AGENT)
    p.add_argument("--min-events-per-agent", type=int, default=MIN_EVENTS_PER_AGENT)
    p.add_argument("--chunksize", type=int, default=2_000_000,
                   help="pandas chunksize when streaming auth.txt.gz")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="cap on number of chunks to read (for fast dry-run)")
    p.add_argument("--dry-run", action="store_true",
                   help="limit to first 5 chunks for pipeline smoke test")
    p.add_argument("--tag", default="lanl-core",
                   help="filename stem for output parquet files (matches config.py)")
    p.add_argument("--cohort-cache",
                   default=os.path.join(OUT_DIR, "_cohort_events.parquet"),
                   help="cache raw cohort-filtered events here; skip the expensive "
                   "two-pass auth.txt.gz scan if it already exists.")
    p.add_argument("--split-cutoff-seconds", type=int, default=SPLIT_CUTOFF_SECONDS,
                   help="temporal train/test cutoff (seconds since LANL epoch). "
                   "Events with time<cutoff go to train, >=cutoff go to test.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Step 1: Parse redteam.txt.gz -> attack set + attack users
# ---------------------------------------------------------------------------
def load_redteam(redteam_file):
    print("=" * 60)
    print("Step 1: Loading red-team ground truth")
    print("=" * 60)
    t0 = time_mod.time()
    with gzip.open(redteam_file, "rt") as f:
        red = pd.read_csv(f, header=None,
                          names=["time", "src_user", "src_computer", "dst_computer"],
                          dtype={"time": np.int64, "src_user": str,
                                 "src_computer": str, "dst_computer": str})
    attack_users = set(red["src_user"].unique())
    attack_hosts = set(red["src_computer"].unique()) | set(red["dst_computer"].unique())
    attack_tuples = set(zip(red["time"].tolist(),
                            red["src_user"].tolist(),
                            red["src_computer"].tolist(),
                            red["dst_computer"].tolist()))
    print(f"  red-team events: {len(red):,}  ({time_mod.time()-t0:.1f}s)")
    print(f"  attack users: {len(attack_users)}  "
          f"attack hosts: {len(attack_hosts)}  "
          f"t-range: [{red['time'].min()}, {red['time'].max()}]s")
    return red, attack_users, attack_hosts, attack_tuples


# ---------------------------------------------------------------------------
# Step 2: Stream auth.txt.gz; first pass builds the user->event-count table
# so we can pick a cohort *before* a second pass collects full events.
# ---------------------------------------------------------------------------
def count_events_per_user(auth_file, chunksize, max_chunks, attack_users):
    """First pass: count events per src_user (cheap; tells us who to keep)."""
    print("\n" + "=" * 60)
    print("Step 2a: First pass - counting events per user")
    print("=" * 60)
    t0 = time_mod.time()
    counts = Counter()
    n_rows = 0
    n_chunks = 0
    reader = pd.read_csv(
        auth_file,
        compression="gzip",
        header=None,
        names=AUTH_COLS,
        dtype=str,
        na_values=["?"],  # '?' means unknown -> NaN
        keep_default_na=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["src_user", "dst_computer"])
        # Group and update counter
        c = chunk.groupby("src_user").size()
        counts.update(c.to_dict())
        n_rows += len(chunk)
        n_chunks += 1
        if n_chunks % 5 == 0:
            print(f"    chunk {n_chunks}: cum rows={n_rows:,}  "
                  f"unique users so far={len(counts):,}  "
                  f"({time_mod.time()-t0:.0f}s)")
        if max_chunks is not None and n_chunks >= max_chunks:
            print(f"  [early stop after {n_chunks} chunks]")
            break
    print(f"  scanned {n_rows:,} rows in {n_chunks} chunks  "
          f"({time_mod.time()-t0:.0f}s)")
    print(f"  unique src_users: {len(counts):,}")
    return counts, n_rows


def pick_cohort(event_counts, attack_users, target_n_agents,
                min_events, rng_seed):
    """Pick a cohort: all active attack users + random benign users."""
    qualified = {u: n for u, n in event_counts.items() if n >= min_events}
    print(f"  users with >={min_events} events: {len(qualified):,}")

    # Keep every attack user that meets the floor; log the drops.
    attack_qualified = {u for u in attack_users if u in qualified}
    attack_missing = attack_users - attack_qualified
    if attack_missing:
        print(f"  [warn] {len(attack_missing)} attack users below "
              f"min_events floor; including them anyway")
    cohort = set(attack_qualified) | attack_missing

    # Random benign users to fill out the cohort
    rng = np.random.default_rng(rng_seed)
    benign_pool = [u for u in qualified if u not in cohort]
    n_benign = max(0, target_n_agents - len(cohort))
    if n_benign >= len(benign_pool):
        cohort.update(benign_pool)
    else:
        idx = rng.choice(len(benign_pool), size=n_benign, replace=False)
        cohort.update(benign_pool[i] for i in idx)
    print(f"  cohort: {len(cohort):,} agents ({len(attack_users)} attack + "
          f"{len(cohort)-len(attack_users)} benign)")
    return cohort


# ---------------------------------------------------------------------------
# Step 3: Second pass - collect full events for the cohort only
# ---------------------------------------------------------------------------
def collect_cohort_events(auth_file, chunksize, max_chunks, cohort):
    print("\n" + "=" * 60)
    print("Step 2b: Second pass - collecting cohort events")
    print("=" * 60)
    t0 = time_mod.time()
    frames = []
    n_rows = 0
    n_kept = 0
    n_chunks = 0
    reader = pd.read_csv(
        auth_file,
        compression="gzip",
        header=None,
        names=AUTH_COLS,
        dtype=str,
        na_values=["?"],
        keep_default_na=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["src_user", "dst_computer"])
        chunk = chunk[chunk["src_user"].isin(cohort)].copy()
        if len(chunk) > 0:
            chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
            chunk = chunk.dropna(subset=["time"])
            chunk["time"] = chunk["time"].astype(np.int64)
            frames.append(chunk)
            n_kept += len(chunk)
        n_rows += len(chunk) if len(chunk) > 0 else 0  # post-filter
        n_chunks += 1
        if n_chunks % 5 == 0:
            print(f"    chunk {n_chunks}: cohort rows kept={n_kept:,}  "
                  f"({time_mod.time()-t0:.0f}s)")
        if max_chunks is not None and n_chunks >= max_chunks:
            print(f"  [early stop after {n_chunks} chunks]")
            break
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=AUTH_COLS)
    print(f"  collected {len(df):,} cohort events in {time_mod.time()-t0:.0f}s")
    return df


# ---------------------------------------------------------------------------
# Step 4: Per-user event cap, host table, category encoding, UMAP coords
# ---------------------------------------------------------------------------
def cap_per_agent(df, max_events, attack_tuples):
    """Cap non-anomalous events per user; keep ALL anomalous rows.

    Pre-flagging the anomaly column here (before capping) ensures every
    red-team event survives even for high-activity attackers.
    """
    before = len(df)
    # Pre-flag anomaly rows so capping can preserve them
    key = list(zip(df["time"].tolist(),
                   df["src_user"].tolist(),
                   df["src_computer"].tolist(),
                   df["dst_computer"].tolist()))
    df = df.copy()
    df["_is_anom"] = np.array([1 if k in attack_tuples else 0 for k in key],
                              dtype=np.int8)
    pre_anom = int(df["_is_anom"].sum())

    # Keep all anomalous rows, then cap benign rows per agent to max_events
    anom_rows = df[df["_is_anom"] == 1]
    benign_rows = (
        df[df["_is_anom"] == 0]
        .sort_values("time")
        .groupby("src_user", group_keys=False)
        .tail(max_events)
    )
    df = pd.concat([anom_rows, benign_rows], ignore_index=True).sort_values("time")
    df = df.reset_index(drop=True)
    post_anom = int(df["_is_anom"].sum())
    print(f"  per-agent cap {max_events} (keep all anom): "
          f"kept {len(df):,} / {before:,} events  "
          f"anom kept: {post_anom}/{pre_anom}")
    return df


def encode_categories(df):
    """Derive a per-host category from the host's dominant auth_type."""
    df = df.copy()
    df["auth_type"] = df["auth_type"].fillna("unknown").str.strip()
    df["logon_type"] = df["logon_type"].fillna("unknown").str.strip()
    df["auth_orientation"] = df["auth_orientation"].fillna("unknown").str.strip()
    df["success"] = df["success"].fillna("unknown").str.strip()

    # Host category = the (auth_type, logon_type) pair that occurs most often
    # on that host. Captures host role (DC/server/workstation) indirectly.
    host_cat_str = (
        df.assign(pair=df["auth_type"].astype(str) + "|" + df["logon_type"].astype(str))
          .groupby("dst_computer")["pair"]
          .agg(lambda s: s.value_counts().index[0])
    )
    unique_cats = sorted(host_cat_str.unique().tolist())
    if "unknown|unknown" not in unique_cats:
        unique_cats.append("unknown|unknown")
    cat_to_id = {c: i for i, c in enumerate(unique_cats)}
    print(f"  unique host categories: {len(cat_to_id)}  "
          f"(auth_type|logon_type pairs)")
    return host_cat_str, cat_to_id


def fit_umap_host_coords(df, rng_seed):
    """2D UMAP on the (user, host) access count matrix; rows are hosts."""
    print("\n" + "=" * 60)
    print("Step 3: Fitting UMAP host coordinates")
    print("=" * 60)
    t0 = time_mod.time()
    users = sorted(df["src_user"].unique())
    hosts = sorted(df["dst_computer"].unique())
    user_idx = {u: i for i, u in enumerate(users)}
    host_idx = {h: i for i, h in enumerate(hosts)}

    rows, cols = [], []
    for u, h in zip(df["src_user"].values, df["dst_computer"].values):
        rows.append(host_idx[h])
        cols.append(user_idx[u])
    data = np.ones(len(rows), dtype=np.int32)

    M = csr_matrix((data, (rows, cols)),
                   shape=(len(hosts), len(users)))
    # Aggregate duplicates
    M.sum_duplicates()
    print(f"  matrix: {M.shape[0]:,} hosts x {M.shape[1]:,} users  "
          f"nnz={M.nnz:,}")

    # UMAP with cosine distance on the access vectors
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(30, M.shape[0] - 1),
        min_dist=0.1,
        random_state=rng_seed,
        verbose=False,
    )
    coords = reducer.fit_transform(M)
    # Rescale to ~[0, 1] in each dim so Space2Vec's lambda range is reasonable
    mn, mx = coords.min(0), coords.max(0)
    span = np.where(mx - mn > 0, mx - mn, 1.0)
    coords = (coords - mn) / span
    print(f"  UMAP done in {time_mod.time()-t0:.0f}s; coord range "
          f"x=[{coords[:,0].min():.3f},{coords[:,0].max():.3f}]  "
          f"y=[{coords[:,1].min():.3f},{coords[:,1].max():.3f}]")
    host_coords = pd.DataFrame({
        "poi_id": hosts,
        "longitude": coords[:, 0].astype(np.float64),
        "latitude":  coords[:, 1].astype(np.float64),
    })
    return host_coords


def build_poi_table(host_coords, host_cat_str, cat_to_id):
    poi = host_coords.copy()
    poi["venue_type"] = poi["poi_id"].map(host_cat_str).fillna("unknown|unknown")
    poi["act_types"] = poi["venue_type"].map(
        lambda v: [int(cat_to_id.get(v, cat_to_id["unknown|unknown"]))]
    )
    return poi[["poi_id", "longitude", "latitude", "venue_type", "act_types"]]


# ---------------------------------------------------------------------------
# Step 5: Assemble final stay dataframe + anomaly labels + temporal split
# ---------------------------------------------------------------------------
def assemble_stays(df, poi, attack_tuples):
    print("\n" + "=" * 60)
    print("Step 4: Assembling stays + anomaly labels")
    print("=" * 60)
    # Map src_user -> stable integer agent ID, 0..N-1
    agents = sorted(df["src_user"].unique())
    agent_to_id = {u: i for i, u in enumerate(agents)}
    df = df.copy()
    df["agent"] = df["src_user"].map(agent_to_id).astype(np.int64)

    # Anomaly label: prefer the pre-flagged column from cap_per_agent when
    # available, otherwise recompute here.
    if "_is_anom" in df.columns:
        df["anomaly"] = df["_is_anom"].astype(np.int64)
        df = df.drop(columns=["_is_anom"])
    else:
        key = list(zip(df["time"].tolist(),
                       df["src_user"].tolist(),
                       df["src_computer"].tolist(),
                       df["dst_computer"].tolist()))
        df["anomaly"] = np.array([1 if k in attack_tuples else 0
                                  for k in key], dtype=np.int64)
    print(f"  anomalous auth events flagged: {df['anomaly'].sum():,} / {len(df):,}"
          f"  ({100*df['anomaly'].mean():.4f}%)")

    # Join POI coords/category
    df = df.merge(
        poi[["poi_id", "longitude", "latitude", "venue_type", "act_types"]],
        left_on="dst_computer", right_on="poi_id", how="left",
    )
    missing = df["longitude"].isna().sum()
    if missing > 0:
        print(f"  [warn] {missing} events with no UMAP coord (dropping)")
        df = df.dropna(subset=["longitude", "latitude"]).copy()

    # Synthetic datetime (seconds since BASE_DATETIME)
    df["start"] = BASE_DATETIME + pd.to_timedelta(df["time"], unit="s")
    df["stop"] = df["start"]

    # cat_0 one-hot index (legacy column kept for compatibility)
    df["cat_0"] = df["act_types"].apply(lambda xs: float(xs[0])
                                        if isinstance(xs, list) and xs else 0.0)

    df = df.sort_values(["agent", "start"]).reset_index(drop=True)
    print(f"  final stays: {len(df):,} agents={df['agent'].nunique():,} "
          f"POIs={df['poi_id'].nunique():,}")
    return df, agent_to_id


def temporal_split(stays, split_cutoff_seconds):
    print("\n" + "=" * 60)
    print(f"Step 5: Temporal split at t={split_cutoff_seconds:,}s "
          f"(~day {split_cutoff_seconds/86400:.1f})")
    print("=" * 60)
    # Compare on raw LANL seconds (column `time`) so the cutoff is robust to
    # BASE_DATETIME choice.
    train = stays[stays["time"] < split_cutoff_seconds].copy()
    test = stays[stays["time"] >= split_cutoff_seconds].copy()

    # Restrict test to agents present in train (otherwise user embeddings
    # are untrained for them).
    train_agents = set(train["agent"].unique())
    dropped = test[~test["agent"].isin(train_agents)]
    if len(dropped) > 0:
        print(f"  [warn] dropping {len(dropped):,} test events "
              f"from {dropped['agent'].nunique()} agents not seen in train")
    test = test[test["agent"].isin(train_agents)].copy()

    print(f"  train: {len(train):,} events  {train['agent'].nunique():,} agents  "
          f"{int(train['anomaly'].sum()):,} anomalies")
    print(f"  test:  {len(test):,} events  {test['agent'].nunique():,} agents  "
          f"{int(test['anomaly'].sum()):,} anomalies")
    return train, test


# ---------------------------------------------------------------------------
# Step 6: Save in Clique format
# ---------------------------------------------------------------------------
def save_outputs(out_dir, tag, train, test, poi):
    os.makedirs(out_dir, exist_ok=True)
    base_cols = ["agent", "longitude", "latitude",
                 "poi_id", "act_types", "cat_0",
                 "start", "stop", "venue_type"]
    train_cols = base_cols + ["anomaly"]  # include anomaly in train too (always 0s if filtered)
    test_cols = base_cols + ["anomaly"]

    train_path = os.path.join(out_dir, f"{tag}_train.parquet")
    test_path = os.path.join(out_dir, f"{tag}_test.parquet")
    poi_path = os.path.join(out_dir, f"{tag}_poi.parquet")
    meta_path = os.path.join(out_dir, f"{tag}_dataset_metadata.parquet")
    aoi_path = os.path.join(out_dir, f"{tag}_aoi_box.geojson")

    train[train_cols].to_parquet(train_path, index=False)
    test[test_cols].to_parquet(test_path, index=False)
    poi.to_parquet(poi_path, index=False)
    print(f"  saved train -> {train_path}")
    print(f"  saved test  -> {test_path}")
    print(f"  saved poi   -> {poi_path}")

    # Metadata
    all_comb = pd.concat([train, test], ignore_index=True)
    min_time = all_comb["start"].min()
    ts_unique = all_comb["start"].drop_duplicates().sort_values()
    diffs = ts_unique.diff().dropna().dt.total_seconds()
    positive = diffs[diffs > 0]
    step_s = float(positive.min()) if len(positive) > 0 else 1.0
    timestamp_step_size = step_s / 3600.0
    meta_df = pd.DataFrame({
        "min_time": [min_time],
        "local_tz": ["UTC"],
        "timestamp_step_size": [timestamp_step_size],
        "utm_crs": ["EPSG:4326"],  # synthetic coords; not true UTM
        "n_agents": [int(train["agent"].nunique())],
    })
    meta_df.to_parquet(meta_path, index=False)
    print(f"  saved meta  -> {meta_path}  (step={timestamp_step_size}h, "
          f"n_agents={int(train['agent'].nunique())})")

    # AOI = bounding box of UMAP coords
    minx, maxx = all_comb["longitude"].min(), all_comb["longitude"].max()
    miny, maxy = all_comb["latitude"].min(), all_comb["latitude"].max()
    aoi_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)],
                               crs="EPSG:4326")
    aoi_gdf.to_file(aoi_path, driver="GeoJSON")
    print(f"  saved aoi   -> {aoi_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    max_chunks = 5 if args.dry_run else args.max_chunks

    # Step 1
    red, attack_users, attack_hosts, attack_tuples = load_redteam(args.redteam_file)

    # Step 2: cohort pick + collect raw events
    # Use the cohort cache if present to skip the expensive two-pass scan.
    if args.cohort_cache and os.path.isfile(args.cohort_cache):
        print(f"\n[cache] loading cohort events from {args.cohort_cache}")
        df = pd.read_parquet(args.cohort_cache)
        print(f"[cache] loaded {len(df):,} rows  "
              f"{df['src_user'].nunique():,} unique users")
    else:
        # Step 2a: event count per user (full pass)
        counts, _ = count_events_per_user(
            args.auth_file, args.chunksize, max_chunks, attack_users)

        # Pick cohort
        cohort = pick_cohort(counts, attack_users, args.target_agents,
                             args.min_events_per_agent, RNG_SEED)

        # Step 2b: collect events for cohort only
        df = collect_cohort_events(args.auth_file, args.chunksize, max_chunks, cohort)
        if len(df) == 0:
            raise RuntimeError("No events collected; check cohort and auth file.")

        if args.cohort_cache:
            os.makedirs(os.path.dirname(args.cohort_cache), exist_ok=True)
            print(f"\n[cache] writing cohort events to {args.cohort_cache}")
            df.to_parquet(args.cohort_cache, index=False)

    # Step 3: per-agent cap (keeps all anomaly=1 rows)
    df = cap_per_agent(df, args.max_events_per_agent, attack_tuples)

    # Step 4: category encoding + UMAP coords
    host_cat_str, cat_to_id = encode_categories(df)
    host_coords = fit_umap_host_coords(df, RNG_SEED)
    poi = build_poi_table(host_coords, host_cat_str, cat_to_id)

    # Step 5: assemble stays and anomaly flags
    stays, agent_to_id = assemble_stays(df, poi, attack_tuples)
    train, test = temporal_split(stays, args.split_cutoff_seconds)

    # Step 6: save
    save_outputs(args.out_dir, args.tag, train, test, poi)

    # Dump cat_to_id for inspection
    pd.DataFrame([{"venue_type": k, "cat_id": v}
                  for k, v in sorted(cat_to_id.items(), key=lambda kv: kv[1])]
                 ).to_csv(os.path.join(args.out_dir,
                                       f"{args.tag}_categories.csv"), index=False)
    pd.DataFrame([{"src_user": u, "agent": i}
                  for u, i in agent_to_id.items()]
                 ).to_csv(os.path.join(args.out_dir,
                                       f"{args.tag}_agent_map.csv"), index=False)

    print("\n=== done ===")


if __name__ == "__main__":
    main()

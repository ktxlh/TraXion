"""Prepare eICU-CRD demo (PhysioNet, open-access) for Clique pre-training.

Cross-domain target: ICU mortality prediction from clinical event streams.

Mapping to Clique's (agent, poi, time, category) schema:
  agent      = patientunitstayid (one agent per ICU stay)
  poi_id     = "<table>::<item>" e.g. "lab::glucose", "med::ASPIRIN", ...
  longitude,
  latitude   = 2D UMAP embedding of the (POI x stay) co-occurrence matrix
  start      = synthetic datetime: BASE + offset minutes from unit-admit
  stop       = start (point events)
  venue_type = the table the event came from (lab / med / treatment / vitalA / infusion)
  act_types  = list[int] encoding of the event-type
  anomaly    = 0 in train (pretraining uses synthetic perturbations); the
               per-stay mortality label is stored in `mortality` and used by
               train_eicu.py / evaluate_eicu.py during fine-tuning.

Cohort: all 2,520 ICU stays; drop the 28 stays with NaN
hospitaldischargestatus.

Split: 70/15/15 by uniquepid (patient-level, no leakage), stratified on
hospital mortality.

Output (to OUT_DIR):
  eicu_train.parquet, eicu_test.parquet, eicu_poi.parquet,
  eicu_aoi_box.geojson, eicu_dataset_metadata.parquet,
  eicu_stay_labels.parquet  (extra: per-stay mortality labels for fine-tuning)
"""

import argparse
import os
import time as time_mod
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import umap
from scipy.sparse import csr_matrix
from shapely.geometry import box

_DATA_ROOT = os.environ.get("TRAXION_DATA_DIR", "./data")
EICU_DIR = os.environ.get(
    "EICU_RAW_DIR",
    os.path.join(_DATA_ROOT, "eicu-collaborative-research-database-demo-2.0.1"),
)
OUT_DIR = os.path.join(_DATA_ROOT, "eicu-demo")

# Synthetic base date so offset minutes -> datetime is well-formed. eICU records
# events as offsets (in minutes) from unit-admit; there are no real calendar
# dates in the demo, so any monotonic mapping works.
BASE_DATETIME = pd.Timestamp("2020-01-01 00:00:00")

RNG_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# POI vocabulary truncation: keep frequent items, lump the long tail under
# "<table>::OTHER". Aggressive truncation keeps the vocab tight (~500-600 POIs)
# which is well-conditioned for a tiny model.
TOP_K_PER_TABLE = {
    "lab": 150,        # ~all 147 lab names
    "med": 200,        # top-200 normalized drug names
    "treat": 100,      # ~81 level-2 treatment groups, keep all
    "infusion": 60,    # top-60 of 257
    "vitalA": 12,      # all distinct vital-aperiodic measurement columns
}


# ---------------------------------------------------------------------------
# Drug-name normalization helpers (medication.csv.gz)
# ---------------------------------------------------------------------------
def _normalize_drugname(name: str) -> str:
    """Collapse the many surface forms of the same drug.

    eICU drug names are deeply messy: "WARFARIN SODIUM 5 MG PO TABS" vs
    "warfarin sodium tab", etc. We lowercase, strip dosage/units, keep only
    the first 1-2 alpha tokens. Enough to merge the worst duplicates.
    """
    if not isinstance(name, str):
        return "unknown"
    n = name.upper().strip()
    # Drop dosage tokens like "0.9%", "5MG", "1000ML", "325 MG", route tags.
    import re
    n = re.sub(r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|UNITS?|IU|MEQ|CC|%)\b", " ", n)
    n = re.sub(r"\b(PO|IV|IM|SC|SQ|SUBQ|PR|TOPICAL|TAB|TABS|CAP|CAPS|SOLN|SOL|"
               r"INJ|INJECTION|ORAL|TABLET|CAPSULE|MBP|FLEX|CONT)\b", " ", n)
    n = re.sub(r"[^A-Z\s]", " ", n)
    tokens = [t for t in n.split() if len(t) > 1]
    if not tokens:
        return "unknown"
    # Keep first 2 alpha tokens — captures "SODIUM CHLORIDE", "POTASSIUM CHLORIDE",
    # while collapsing dosage-distinct duplicates.
    return " ".join(tokens[:2])


# ---------------------------------------------------------------------------
# Per-table event extractors. Each returns a DataFrame with cols:
#   patientunitstayid, offset_minutes, poi_id, table_name
# ---------------------------------------------------------------------------
def extract_lab_events(eicu_dir: str, top_k: int) -> pd.DataFrame:
    print("=" * 60)
    print("Step 1a: Extracting lab events")
    print("=" * 60)
    df = pd.read_csv(f"{eicu_dir}/lab.csv.gz", low_memory=False)
    df = df.dropna(subset=["patientunitstayid", "labresultoffset", "labname"])
    counts = df["labname"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["labname"].where(df["labname"].isin(keep), "OTHER")
    out = pd.DataFrame({
        "patientunitstayid": df["patientunitstayid"].astype(np.int64),
        "offset_minutes": df["labresultoffset"].astype(np.int64),
        "poi_id": "lab::" + df["item"].astype(str),
        "table_name": "lab",
    })
    print(f"  lab: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_med_events(eicu_dir: str, top_k: int) -> pd.DataFrame:
    print("Step 1b: Extracting medication events")
    df = pd.read_csv(f"{eicu_dir}/medication.csv.gz", low_memory=False)
    df = df.dropna(subset=["patientunitstayid", "drugstartoffset", "drugname"])
    df = df[df["drugordercancelled"] == "No"]
    df["item"] = df["drugname"].apply(_normalize_drugname)
    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")
    out = pd.DataFrame({
        "patientunitstayid": df["patientunitstayid"].astype(np.int64),
        "offset_minutes": df["drugstartoffset"].astype(np.int64),
        "poi_id": "med::" + df["item"].astype(str),
        "table_name": "med",
    })
    print(f"  med: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_treatment_events(eicu_dir: str, top_k: int) -> pd.DataFrame:
    print("Step 1c: Extracting treatment events")
    df = pd.read_csv(f"{eicu_dir}/treatment.csv.gz", low_memory=False)
    df = df.dropna(subset=["patientunitstayid", "treatmentoffset", "treatmentstring"])
    # treatmentstring is hierarchical "level1|level2|level3|...". Use first 2
    # levels (~80 unique) — enough granularity, stable vocab.
    df["item"] = (
        df["treatmentstring"].str.split("|").str[:2].str.join("|")
        .replace("", "unknown").fillna("unknown")
    )
    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")
    out = pd.DataFrame({
        "patientunitstayid": df["patientunitstayid"].astype(np.int64),
        "offset_minutes": df["treatmentoffset"].astype(np.int64),
        "poi_id": "treat::" + df["item"].astype(str),
        "table_name": "treat",
    })
    print(f"  treat: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_infusion_events(eicu_dir: str, top_k: int) -> pd.DataFrame:
    print("Step 1d: Extracting infusiondrug events")
    df = pd.read_csv(f"{eicu_dir}/infusiondrug.csv.gz", low_memory=False)
    df = df.dropna(subset=["patientunitstayid", "infusionoffset", "drugname"])
    # infusion drugnames are cleaner than medication's; keep as-is.
    df["item"] = df["drugname"].astype(str).str.strip()
    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")
    out = pd.DataFrame({
        "patientunitstayid": df["patientunitstayid"].astype(np.int64),
        "offset_minutes": df["infusionoffset"].astype(np.int64),
        "poi_id": "infusion::" + df["item"].astype(str),
        "table_name": "infusion",
    })
    print(f"  infusion: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_vitalA_events(eicu_dir: str, top_k: int) -> pd.DataFrame:
    """Treat each non-null measurement column as a separate event token.

    A single row of vitalAperiodic can carry up to 11 measurements (NIBP*3,
    paop, cardiacoutput, cardiacinput, svr, svri, pvr, pvri). We melt to long
    form: one event per (row, non-null column).
    """
    print("Step 1e: Extracting vitalAperiodic events (presence only)")
    df = pd.read_csv(f"{eicu_dir}/vitalAperiodic.csv.gz", low_memory=False)
    df = df.dropna(subset=["patientunitstayid", "observationoffset"])
    measurement_cols = [
        "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean",
        "paop", "cardiacoutput", "cardiacinput",
        "svr", "svri", "pvr", "pvri",
    ]
    long = df[["patientunitstayid", "observationoffset"] + measurement_cols].melt(
        id_vars=["patientunitstayid", "observationoffset"],
        value_vars=measurement_cols,
        var_name="item", value_name="val",
    )
    long = long.dropna(subset=["val"])
    counts = long["item"].value_counts()
    keep = set(counts.head(top_k).index)
    long["item"] = long["item"].where(long["item"].isin(keep), "OTHER")
    out = pd.DataFrame({
        "patientunitstayid": long["patientunitstayid"].astype(np.int64),
        "offset_minutes": long["observationoffset"].astype(np.int64),
        "poi_id": "vitalA::" + long["item"].astype(str),
        "table_name": "vitalA",
    })
    print(f"  vitalA: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


# ---------------------------------------------------------------------------
# Step 2: Load patient table; derive per-stay mortality labels & demographics
# ---------------------------------------------------------------------------
def load_patient_labels(eicu_dir: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 2: Loading patient table + mortality labels")
    print("=" * 60)
    p = pd.read_csv(f"{eicu_dir}/patient.csv.gz", low_memory=False)
    print(f"  raw stays: {len(p):,}")

    # Drop stays with unknown discharge status (28 rows in demo).
    before = len(p)
    p = p[p["hospitaldischargestatus"].isin(["Alive", "Expired"])].copy()
    print(f"  dropped {before - len(p)} stays with unknown discharge status")

    p["mortality"] = (p["hospitaldischargestatus"] == "Expired").astype(np.int64)
    print(f"  hospital mortality rate: {p['mortality'].mean():.4f}  "
          f"({p['mortality'].sum():,} / {len(p):,})")

    # Keep useful demographic / unit-stay metadata for the per-stay label table.
    keep_cols = [
        "patientunitstayid", "patienthealthsystemstayid", "uniquepid",
        "hospitalid", "wardid", "unittype", "unitvisitnumber",
        "gender", "age", "ethnicity", "apacheadmissiondx",
        "admissionheight", "admissionweight",
        "hospitaladmittime24", "unitadmittime24",
        "hospitaldischargestatus", "unitdischargestatus", "mortality",
    ]
    p = p[keep_cols].reset_index(drop=True)
    return p


# ---------------------------------------------------------------------------
# Step 3: Patient-level stratified split (70/15/15)
# ---------------------------------------------------------------------------
def patient_level_split(patient_df: pd.DataFrame, rng_seed: int):
    """Split by uniquepid so a patient's stays never appear in two splits.

    Stratified on mortality at the patient level (positive if any stay
    expired). For the 1,841 patients in the demo this gives ~155 mortal /
    1,686 non-mortal patients.
    """
    print("\n" + "=" * 60)
    print(f"Step 3: Patient-level split  ({TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%})")
    print("=" * 60)
    pat_label = (
        patient_df.groupby("uniquepid")["mortality"].max().rename("any_mortality")
    )
    pos_pids = pat_label[pat_label == 1].index.tolist()
    neg_pids = pat_label[pat_label == 0].index.tolist()

    rng = np.random.default_rng(rng_seed)
    rng.shuffle(pos_pids); rng.shuffle(neg_pids)

    def _split(pids):
        n = len(pids); n_tr = int(round(n * TRAIN_FRAC))
        n_va = int(round(n * VAL_FRAC)); n_te = n - n_tr - n_va
        return pids[:n_tr], pids[n_tr:n_tr+n_va], pids[n_tr+n_va:n_tr+n_va+n_te]

    tr_pos, va_pos, te_pos = _split(pos_pids)
    tr_neg, va_neg, te_neg = _split(neg_pids)

    splits = {
        "train": set(tr_pos) | set(tr_neg),
        "val":   set(va_pos) | set(va_neg),
        "test":  set(te_pos) | set(te_neg),
    }
    for name, pids in splits.items():
        sub = patient_df[patient_df["uniquepid"].isin(pids)]
        print(f"  {name}: {len(pids):,} patients, "
              f"{len(sub):,} stays, "
              f"mortality={sub['mortality'].mean():.4f}")
    return splits


# ---------------------------------------------------------------------------
# Step 4: UMAP coordinates over the (POI x stay) co-occurrence matrix
# ---------------------------------------------------------------------------
def fit_umap_poi_coords(events: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 4: UMAP POI coordinates")
    print("=" * 60)
    t0 = time_mod.time()

    pois = sorted(events["poi_id"].unique())
    stays = sorted(events["patientunitstayid"].unique())
    poi_idx = {p: i for i, p in enumerate(pois)}
    stay_idx = {s: i for i, s in enumerate(stays)}

    rows = events["poi_id"].map(poi_idx).to_numpy()
    cols = events["patientunitstayid"].map(stay_idx).to_numpy()
    data = np.ones(len(rows), dtype=np.int32)
    M = csr_matrix((data, (rows, cols)), shape=(len(pois), len(stays)))
    M.sum_duplicates()
    print(f"  matrix: {M.shape[0]:,} POIs × {M.shape[1]:,} stays  nnz={M.nnz:,}")

    n_neighbors = max(2, min(30, M.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2, metric="cosine",
        n_neighbors=n_neighbors, min_dist=0.1,
        random_state=rng_seed, verbose=False,
    )
    coords = reducer.fit_transform(M)
    mn, mx = coords.min(0), coords.max(0)
    span = np.where(mx - mn > 0, mx - mn, 1.0)
    coords = (coords - mn) / span  # rescale to [0, 1]^2
    print(f"  UMAP done in {time_mod.time()-t0:.0f}s; coord range "
          f"x=[{coords[:,0].min():.3f},{coords[:,0].max():.3f}]  "
          f"y=[{coords[:,1].min():.3f},{coords[:,1].max():.3f}]")

    poi_df = pd.DataFrame({
        "poi_id": pois,
        "longitude": coords[:, 0].astype(np.float64),
        "latitude":  coords[:, 1].astype(np.float64),
    })
    return poi_df


def build_poi_table(poi_coords: pd.DataFrame) -> pd.DataFrame:
    """Add venue_type (= source table) and act_types (cat_id list)."""
    poi = poi_coords.copy()
    poi["venue_type"] = poi["poi_id"].str.split("::").str[0]
    cats = sorted(poi["venue_type"].unique())
    cat_to_id = {c: i for i, c in enumerate(cats)}
    poi["act_types"] = poi["venue_type"].map(lambda c: [int(cat_to_id[c])])
    print(f"  venue types: {cats}  ({len(cats)} categories)")
    return poi[["poi_id", "longitude", "latitude", "venue_type", "act_types"]], cat_to_id


# ---------------------------------------------------------------------------
# Step 5: Assemble the per-event stay dataframe with timestamps + agent ids
# ---------------------------------------------------------------------------
def assemble_stays(
    events: pd.DataFrame,
    poi: pd.DataFrame,
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 5: Assembling stays")
    print("=" * 60)

    # Restrict events to stays that survived the patient-table filter.
    valid_stays = set(patient_df["patientunitstayid"].astype(np.int64).tolist())
    before = len(events)
    events = events[events["patientunitstayid"].isin(valid_stays)].copy()
    print(f"  events kept (valid stays): {len(events):,} / {before:,}")

    # Stable contiguous agent_id 0..N-1 keyed off patientunitstayid.
    stay_ids = sorted(valid_stays)
    stay_to_agent = {s: i for i, s in enumerate(stay_ids)}
    events["agent"] = events["patientunitstayid"].map(stay_to_agent).astype(np.int64)

    # Synthetic absolute datetime: BASE + offset minutes.
    events["start"] = BASE_DATETIME + pd.to_timedelta(
        events["offset_minutes"], unit="m"
    )
    events["stop"] = events["start"]

    # Mortality label per event (replicated from per-stay table — used by the
    # per-stay BCE head, NOT visit-level anomaly).
    stay_to_mort = dict(zip(
        patient_df["patientunitstayid"].astype(np.int64),
        patient_df["mortality"].astype(np.int64),
    ))
    events["mortality"] = events["patientunitstayid"].map(stay_to_mort).astype(np.int64)

    # During pre-training we use synthetic perturbations (Noisifier), so the
    # `anomaly` column is uniformly zero at this stage. The BCE-on-mortality
    # head used by train_eicu.py reads `mortality` directly.
    events["anomaly"] = 0

    # Join POI coords + categories.
    events = events.merge(
        poi[["poi_id", "longitude", "latitude", "venue_type", "act_types"]],
        on="poi_id", how="left",
    )
    missing = events["longitude"].isna().sum()
    if missing > 0:
        print(f"  [warn] {missing:,} events with no UMAP coord (dropping)")
        events = events.dropna(subset=["longitude", "latitude"]).copy()

    # cat_0 (legacy column kept for compatibility with downstream loaders).
    events["cat_0"] = events["act_types"].apply(
        lambda xs: float(xs[0]) if isinstance(xs, list) and xs else 0.0
    )

    events = events.sort_values(["agent", "start"]).reset_index(drop=True)
    print(f"  final events: {len(events):,}  agents={events['agent'].nunique():,}  "
          f"POIs={events['poi_id'].nunique():,}")
    return events, stay_to_agent


# ---------------------------------------------------------------------------
# Step 6: Save outputs
# ---------------------------------------------------------------------------
def save_outputs(
    out_dir: str, tag: str,
    train_events: pd.DataFrame, val_events: pd.DataFrame, test_events: pd.DataFrame,
    poi: pd.DataFrame, patient_df: pd.DataFrame, stay_to_agent: dict,
):
    os.makedirs(out_dir, exist_ok=True)
    base_cols = [
        "agent", "longitude", "latitude",
        "poi_id", "act_types", "cat_0",
        "start", "stop", "venue_type",
        "anomaly", "mortality",
        "patientunitstayid",
    ]

    # Pre-training and finetune both read train.parquet via load_stays("train", cfg).
    # The fine-tune script will read mortality + agent ids and build per-stay heads
    # over the union of train + val + test events (split by agent).
    #
    # Convention used here:
    #   - train.parquet  := events from agents in the train+val pid splits (used
    #                       for pretraining + fine-tune training).
    #   - test.parquet   := events from agents in the test pid split.
    # The fine-tune trainer further splits train.parquet's stays into a held-out
    # validation slice (by stay id) inside train_eicu.py.
    train_path = os.path.join(out_dir, f"{tag}_train.parquet")
    test_path = os.path.join(out_dir, f"{tag}_test.parquet")
    poi_path = os.path.join(out_dir, f"{tag}_poi.parquet")
    meta_path = os.path.join(out_dir, f"{tag}_dataset_metadata.parquet")
    aoi_path = os.path.join(out_dir, f"{tag}_aoi_box.geojson")
    labels_path = os.path.join(out_dir, f"{tag}_stay_labels.parquet")

    pretrain_events = pd.concat([train_events, val_events], ignore_index=True)
    pretrain_events.sort_values(["agent", "start"], inplace=True)
    pretrain_events = pretrain_events.reset_index(drop=True)

    pretrain_events[base_cols].to_parquet(train_path, index=False)
    test_events[base_cols].to_parquet(test_path, index=False)
    poi.to_parquet(poi_path, index=False)
    print(f"  saved pretrain (train+val) -> {train_path}  rows={len(pretrain_events):,}")
    print(f"  saved test                  -> {test_path}  rows={len(test_events):,}")
    print(f"  saved poi                   -> {poi_path}  pois={len(poi):,}")

    # Per-stay label table (used by train_eicu.py / evaluate_eicu.py).
    pl = patient_df.copy()
    pl["agent"] = pl["patientunitstayid"].map(stay_to_agent).astype("Int64")
    pl = pl.dropna(subset=["agent"]).copy()
    pl["agent"] = pl["agent"].astype(np.int64)
    # Tag each stay with its split.
    train_stays = set(train_events["patientunitstayid"].astype(np.int64).unique())
    val_stays = set(val_events["patientunitstayid"].astype(np.int64).unique())
    test_stays = set(test_events["patientunitstayid"].astype(np.int64).unique())

    def _split_of(s):
        if s in train_stays: return "train"
        if s in val_stays:   return "val"
        if s in test_stays:  return "test"
        return "drop"
    pl["split"] = pl["patientunitstayid"].map(_split_of)
    pl = pl[pl["split"] != "drop"].reset_index(drop=True)
    pl.to_parquet(labels_path, index=False)
    print(f"  saved stay labels -> {labels_path}  rows={len(pl):,}  "
          f"splits={pl['split'].value_counts().to_dict()}")

    # Metadata.
    all_events = pd.concat([train_events, val_events, test_events], ignore_index=True)
    min_time = all_events["start"].min()
    ts_unique = all_events["start"].drop_duplicates().sort_values()
    diffs = ts_unique.diff().dropna().dt.total_seconds()
    positive = diffs[diffs > 0]
    step_s = float(positive.min()) if len(positive) > 0 else 60.0
    timestamp_step_size = step_s / 3600.0
    n_total_agents = len(stay_to_agent)
    meta_df = pd.DataFrame({
        "min_time": [min_time],
        "local_tz": ["UTC"],
        "timestamp_step_size": [timestamp_step_size],
        "utm_crs": ["EPSG:4326"],
        "n_agents": [int(n_total_agents)],
    })
    meta_df.to_parquet(meta_path, index=False)
    print(f"  saved meta -> {meta_path}  step={timestamp_step_size}h  "
          f"n_agents={n_total_agents}")

    # AOI = bounding box of UMAP coords.
    minx, maxx = all_events["longitude"].min(), all_events["longitude"].max()
    miny, maxy = all_events["latitude"].min(), all_events["latitude"].max()
    aoi_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)],
                               crs="EPSG:4326")
    aoi_gdf.to_file(aoi_path, driver="GeoJSON")
    print(f"  saved aoi -> {aoi_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eicu-dir", default=EICU_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--tag", default="eicu-demo",
                   help="filename stem for output parquet files (matches config.py)")
    p.add_argument("--seed", type=int, default=RNG_SEED)
    return p.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: extract event streams from all five tables.
    events_lab = extract_lab_events(args.eicu_dir, TOP_K_PER_TABLE["lab"])
    events_med = extract_med_events(args.eicu_dir, TOP_K_PER_TABLE["med"])
    events_treat = extract_treatment_events(args.eicu_dir, TOP_K_PER_TABLE["treat"])
    events_inf = extract_infusion_events(args.eicu_dir, TOP_K_PER_TABLE["infusion"])
    events_va = extract_vitalA_events(args.eicu_dir, TOP_K_PER_TABLE["vitalA"])
    events = pd.concat(
        [events_lab, events_med, events_treat, events_inf, events_va],
        ignore_index=True,
    )
    print(f"\nTotal raw events: {len(events):,}  "
          f"unique POIs: {events['poi_id'].nunique():,}")

    # Step 2: load patient table + mortality labels.
    patient_df = load_patient_labels(args.eicu_dir)

    # Step 3: patient-level stratified split.
    splits = patient_level_split(patient_df, args.seed)

    # Step 4: UMAP POI coords (computed on all events so train/test share embedding).
    poi_coords = fit_umap_poi_coords(events, args.seed)
    poi, cat_to_id = build_poi_table(poi_coords)

    # Step 5: assemble stays (events + coords + labels).
    events, stay_to_agent = assemble_stays(events, poi, patient_df)

    # Split events into train/val/test by uniquepid.
    pid_of_stay = dict(zip(
        patient_df["patientunitstayid"].astype(np.int64),
        patient_df["uniquepid"],
    ))
    events["uniquepid"] = events["patientunitstayid"].map(pid_of_stay)

    train_pids = splits["train"]; val_pids = splits["val"]; test_pids = splits["test"]
    train_events = events[events["uniquepid"].isin(train_pids)].drop(columns=["uniquepid"]).copy()
    val_events = events[events["uniquepid"].isin(val_pids)].drop(columns=["uniquepid"]).copy()
    test_events = events[events["uniquepid"].isin(test_pids)].drop(columns=["uniquepid"]).copy()
    print(f"  events  train={len(train_events):,}  val={len(val_events):,}  "
          f"test={len(test_events):,}")

    # Step 6: save.
    save_outputs(
        args.out_dir, args.tag,
        train_events, val_events, test_events,
        poi, patient_df, stay_to_agent,
    )

    # Sidecar: cat-id mapping (for inspection/debugging).
    pd.DataFrame([{"venue_type": k, "cat_id": v}
                  for k, v in sorted(cat_to_id.items(), key=lambda kv: kv[1])]
                 ).to_csv(os.path.join(args.out_dir, f"{args.tag}_categories.csv"), index=False)

    print("\n=== done ===")


if __name__ == "__main__":
    main()

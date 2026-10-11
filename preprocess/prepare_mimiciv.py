"""Prepare PhysioNet MIMIC-IV 3.1 for TraXion pre-training and mortality fine-tuning.

The output uses the same file layout and column names as prepare_eicu.py, so
cli/train_eicu.py and cli/evaluate_eicu.py read it unchanged.

Mapping to TraXion's (agent, poi, time, category) schema:
  agent      = icu/icustays.stay_id (one agent per ICU stay; re-key to the patient
               with preprocess/rekey_patient.py for pre-training)
  poi_id     = "<table>::<item>" e.g. "lab::Glucose", "med::INSULIN", ...
  longitude,
  latitude   = 2D UMAP embedding of the (POI x stay) co-occurrence matrix
  start      = synthetic datetime: BASE + offset minutes from the stay's
               icustays.intime. MIMIC-IV shifts each patient's dates by an
               independent random offset, so times are made relative to each stay.
  stop       = start (point events)
  venue_type = the table the event came from (lab / med / treat / infusion / vital)
  act_types  = list[int] encoding of the event-type
  anomaly    = 0 (pre-training uses synthetic perturbations); the
               per-stay mortality label (hosp/admissions.hospital_expire_flag) is
               stored in `<tag>_stay_labels.parquet` and used by train_eicu.py /
               evaluate_eicu.py during fine-tuning.

Event sources:
  lab      = hosp/labevents (item names from hosp/d_labitems)
  med      = hosp/emar, administrations only (event_txt == "Administered")
  treat    = icu/procedureevents (item names from icu/d_items)
  infusion = icu/inputevents (item names from icu/d_items)
  vital    = icu/chartevents, restricted to a fixed list of itemids (non-invasive
             blood pressure, cardiac output, cardiac index, SVRI)
hosp/ tables are keyed by admission (hadm_id). Each hosp/ event is assigned to the
ICU stay of its admission whose [intime, outtime] contains it; events outside every
ICU stay are dropped. icu/ tables carry stay_id directly.

Stay labels: one row per ICU stay. Columns are renamed to the eICU names read by
tasks/eicu_mortality/tabular_features.py (stay_id -> patientunitstayid,
subject_id -> uniquepid, anchor_age -> age, race -> ethnicity,
first_careunit -> unittype).

Cohort: every ICU stay in icu/icustays with a hospital_expire_flag.

Split: 70/15/15 by subject_id (patient-level), stratified on hospital mortality.

Output (to --out-dir):
  <tag>_train.parquet, <tag>_test.parquet, <tag>_poi.parquet,
  <tag>_aoi_box.geojson, <tag>_dataset_metadata.parquet,
  <tag>_stay_labels.parquet (per-stay mortality labels for fine-tuning),
  <tag>_categories.csv
"""

from __future__ import annotations

import argparse
import os
import re
import time as time_mod

import geopandas as gpd
import numpy as np
import pandas as pd
import umap
from scipy.sparse import csr_matrix
from shapely.geometry import box

_DATA_ROOT = os.environ.get("TRAXION_DATA_DIR", "./data")
MIMIC_DIR = os.environ.get(
    "MIMICIV_RAW_DIR",
    os.path.join(_DATA_ROOT, "physionet.org", "files", "mimiciv", "3.1"),
)
OUT_DIR = os.path.join(_DATA_ROOT, "mimiciv-full")

# Synthetic base date (same as prepare_eicu.py); event times are stored as
# BASE_DATETIME + minutes since the stay's ICU admission.
BASE_DATETIME = pd.Timestamp("2020-01-01 00:00:00")

RNG_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# POI vocabulary truncation: keep the top-K most frequent items per table.
TOP_K_PER_TABLE = {
    "lab": 150,        # hosp/labevents
    "med": 200,        # hosp/emar, normalized medication names
    "treat": 100,      # icu/procedureevents
    "infusion": 60,    # icu/inputevents
    "vital": 12,       # icu/chartevents, fixed itemid list
}

# Medication source: "emar" (administrations, hosp/emar.csv.gz) or
# "prescriptions" (orders, hosp/prescriptions.csv.gz).
MED_SOURCE = "emar"


def _tbl(mimic_dir: str, module: str, name: str) -> str:
    """Resolve a MIMIC-IV table path under `module` ("hosp" or "icu").

    An exact filename match is used if present; otherwise a case-insensitive match.
    """
    base = os.path.join(mimic_dir, module)
    exact = os.path.join(base, name)
    if os.path.isfile(exact):
        return exact
    try:
        entries = os.listdir(base)
    except OSError as e:
        raise FileNotFoundError(f"MIMIC-IV {module}/ dir not readable: {base}") from e
    lowered = name.lower()
    for cand in entries:
        if cand.lower() == lowered:
            return os.path.join(base, cand)
    raise FileNotFoundError(
        f"{name} not found in {base} (also tried case-insensitive). "
        f"Present: {sorted(c for c in entries if c.endswith('.csv.gz'))}"
    )


# ---------------------------------------------------------------------------
# Step 0: ICU stay index
# ---------------------------------------------------------------------------
def load_icustays(mimic_dir: str) -> pd.DataFrame:
    """Load icu/icustays.csv.gz: subject_id, hadm_id, stay_id, intime, outtime, first_careunit."""
    path = _tbl(mimic_dir, "icu", "icustays.csv.gz")
    usecols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"]
    df = pd.read_csv(path, usecols=usecols, compression="gzip",
                      parse_dates=["intime", "outtime"])
    return df


# ---------------------------------------------------------------------------
# Step 1: per-stay mortality label + demographics
# ---------------------------------------------------------------------------
def load_stay_labels(mimic_dir: str, icustays: pd.DataFrame) -> pd.DataFrame:
    """Build one row per ICU stay with the hospital-mortality label and demographics.

    The label is hosp/admissions.hospital_expire_flag of the stay's admission.
    Columns use the eICU names read by tasks/eicu_mortality/tabular_features.py.
    apacheadmissiondx, admissionheight and admissionweight have no counterpart in
    the tables read here and are set to NaN.
    """
    adm_path = _tbl(mimic_dir, "hosp", "admissions.csv.gz")
    adm_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                "admission_type", "admission_location", "race", "hospital_expire_flag"]
    adm = pd.read_csv(adm_path, usecols=adm_cols, compression="gzip",
                       parse_dates=["admittime", "dischtime", "deathtime"])

    pat_path = _tbl(mimic_dir, "hosp", "patients.csv.gz")
    pat_cols = ["subject_id", "gender", "anchor_age"]
    pat = pd.read_csv(pat_path, usecols=pat_cols, compression="gzip")

    sl = icustays.merge(adm, on=["subject_id", "hadm_id"], how="left")
    sl = sl.merge(pat, on="subject_id", how="left")

    before = len(sl)
    sl = sl.dropna(subset=["hospital_expire_flag"]).copy()
    print(f"  dropped {before - len(sl)} stays with no matching admission / expire flag")

    sl["mortality"] = sl["hospital_expire_flag"].astype(np.int64)
    print(f"  hospital mortality rate: {sl['mortality'].mean():.4f}  "
          f"({sl['mortality'].sum():,} / {len(sl):,})")

    # Rename to the eICU column names read by tasks/eicu_mortality/tabular_features.py.
    sl = sl.rename(columns={
        "stay_id": "patientunitstayid",
        "subject_id": "uniquepid",
        "anchor_age": "age",
        "race": "ethnicity",
        "first_careunit": "unittype",
    })
    sl["gender"] = sl["gender"].map({"M": "Male", "F": "Female"})  # eICU string values

    sl["apacheadmissiondx"] = np.nan
    sl["admissionheight"] = np.nan
    sl["admissionweight"] = np.nan

    # unitvisitnumber: ordinal of the ICU stay within the patient, ordered by intime.
    sl = sl.sort_values(["uniquepid", "intime"])
    sl["unitvisitnumber"] = sl.groupby("uniquepid").cumcount() + 1

    keep_cols = [
        "patientunitstayid", "uniquepid", "hadm_id", "intime", "outtime",
        "gender", "age", "ethnicity", "apacheadmissiondx",
        "admissionheight", "admissionweight", "unitvisitnumber", "unittype",
        "mortality",
    ]
    return sl[keep_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: per-table event extractors. Each returns a DataFrame with cols:
#   stay_id, offset_minutes, poi_id, table_name
# ---------------------------------------------------------------------------
def _restrict_to_stay_window(events: pd.DataFrame, icustays: pd.DataFrame,
                              time_col: str) -> pd.DataFrame:
    """Assign each hosp/ event (keyed by subject_id, hadm_id) to the ICU stay whose
    [intime, outtime] contains `time_col`; events outside every stay are dropped.
    An admission can contain more than one ICU stay.
    """
    merged = events.merge(
        icustays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
        on=["subject_id", "hadm_id"], how="inner",
    )
    mask = (merged[time_col] >= merged["intime"]) & (merged[time_col] <= merged["outtime"])
    return merged[mask].copy()


def _offset_minutes_col(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Minutes from the stay's intime to `time_col`, rounded to an integer."""
    return ((df[time_col] - df["intime"]).dt.total_seconds() / 60.0).round().astype(np.int64)


def _normalize_med_name(name: str) -> str:
    """Collapse surface forms of the same medication: uppercase, strip dosage
    tokens and non-letters, keep the first two tokens.
    """
    if not isinstance(name, str):
        return "unknown"
    n = name.upper().strip()
    n = re.sub(r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|UNITS?|IU|MEQ|CC|%)\b", " ", n)
    n = re.sub(r"[^A-Z\s]", " ", n)
    tokens = [t for t in n.split() if len(t) > 1]
    return " ".join(tokens[:2]) if tokens else "unknown"


def extract_lab_events(mimic_dir: str, icustays: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """hosp/labevents.csv.gz, item names from hosp/d_labitems.csv.gz.

    Read in chunks; rows without hadm_id are dropped.
    """
    print("Step 1a: Extracting lab events (hosp/labevents.csv.gz)")
    d_labitems = pd.read_csv(_tbl(mimic_dir, "hosp", "d_labitems.csv.gz"),
                              usecols=["itemid", "label"], compression="gzip")
    item_to_label = dict(zip(d_labitems["itemid"], d_labitems["label"]))

    path = _tbl(mimic_dir, "hosp", "labevents.csv.gz")
    keep_hadm = set(icustays["hadm_id"].dropna().astype(np.int64))
    usecols = ["subject_id", "hadm_id", "itemid", "charttime"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, compression="gzip", chunksize=2_000_000,
                              parse_dates=["charttime"]):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "itemid", "charttime"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(np.int64)
        chunk = chunk[chunk["hadm_id"].isin(keep_hadm)]
        if len(chunk):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    df = _restrict_to_stay_window(df, icustays, "charttime")

    df["item"] = df["itemid"].map(item_to_label).fillna("UNKNOWN_ITEM")
    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")

    out = pd.DataFrame({
        "stay_id": df["stay_id"].astype(np.int64),
        "offset_minutes": _offset_minutes_col(df, "charttime"),
        "poi_id": "lab::" + df["item"].astype(str),
        "table_name": "lab",
    })
    print(f"  lab: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_med_events(mimic_dir: str, icustays: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Medication events from hosp/emar.csv.gz (administrations, event_txt ==
    "Administered") or, with MED_SOURCE = "prescriptions", hosp/prescriptions.csv.gz.
    """
    print(f"Step 1b: Extracting medication events (hosp/{MED_SOURCE}.csv.gz)")
    if MED_SOURCE == "emar":
        path = _tbl(mimic_dir, "hosp", "emar.csv.gz")
        usecols = ["subject_id", "hadm_id", "charttime", "medication", "event_txt"]
        keep_hadm = set(icustays["hadm_id"].dropna().astype(np.int64))
        chunks = []
        for chunk in pd.read_csv(path, usecols=usecols, compression="gzip", chunksize=2_000_000,
                                  parse_dates=["charttime"]):
            chunk = chunk.dropna(subset=["subject_id", "hadm_id", "charttime", "medication"])
            chunk["hadm_id"] = chunk["hadm_id"].astype(np.int64)
            chunk = chunk[chunk["hadm_id"].isin(keep_hadm)]
            # Keep administered doses only.
            chunk = chunk[chunk["event_txt"] == "Administered"]
            if len(chunk):
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
        df = _restrict_to_stay_window(df, icustays, "charttime")
        df["item"] = df["medication"].apply(_normalize_med_name)
        time_col = "charttime"
    else:  # MED_SOURCE == "prescriptions"
        path = _tbl(mimic_dir, "hosp", "prescriptions.csv.gz")
        usecols = ["subject_id", "hadm_id", "starttime", "drug"]
        keep_hadm = set(icustays["hadm_id"].dropna().astype(np.int64))
        chunks = []
        for chunk in pd.read_csv(path, usecols=usecols, compression="gzip", chunksize=2_000_000,
                                  parse_dates=["starttime"]):
            chunk = chunk.dropna(subset=["subject_id", "hadm_id", "starttime", "drug"])
            chunk["hadm_id"] = chunk["hadm_id"].astype(np.int64)
            chunk = chunk[chunk["hadm_id"].isin(keep_hadm)]
            if len(chunk):
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
        df = _restrict_to_stay_window(df, icustays, "starttime")
        df["item"] = df["drug"].apply(_normalize_med_name)
        time_col = "starttime"

    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")
    out = pd.DataFrame({
        "stay_id": df["stay_id"].astype(np.int64),
        "offset_minutes": _offset_minutes_col(df, time_col),
        "poi_id": "med::" + df["item"].astype(str),
        "table_name": "med",
    })
    print(f"  med: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_treat_events(mimic_dir: str, icustays: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """icu/procedureevents.csv.gz, item names from icu/d_items.csv.gz; timed by starttime."""
    print("Step 1c: Extracting treatment events (icu/procedureevents.csv.gz)")
    d_items = pd.read_csv(_tbl(mimic_dir, "icu", "d_items.csv.gz"),
                           usecols=["itemid", "label"], compression="gzip")
    item_to_label = dict(zip(d_items["itemid"], d_items["label"]))

    path = _tbl(mimic_dir, "icu", "procedureevents.csv.gz")
    keep_stays = set(icustays["stay_id"].astype(np.int64))
    usecols = ["stay_id", "starttime", "itemid"]
    df = pd.read_csv(path, usecols=usecols, compression="gzip", parse_dates=["starttime"])
    df = df.dropna(subset=usecols)
    df["stay_id"] = df["stay_id"].astype(np.int64)
    df = df[df["stay_id"].isin(keep_stays)]

    df["item"] = df["itemid"].map(item_to_label).fillna("UNKNOWN_ITEM")
    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")

    icu_small = icustays[["stay_id", "intime"]]
    df = df.merge(icu_small, on="stay_id", how="left")
    out = pd.DataFrame({
        "stay_id": df["stay_id"],
        "offset_minutes": _offset_minutes_col(df, "starttime"),
        "poi_id": "treat::" + df["item"].astype(str),
        "table_name": "treat",
    })
    print(f"  treat: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_infusion_events(mimic_dir: str, icustays: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """icu/inputevents.csv.gz, item names from icu/d_items.csv.gz; timed by starttime."""
    print("Step 1d: Extracting infusion events (icu/inputevents.csv.gz)")
    d_items = pd.read_csv(_tbl(mimic_dir, "icu", "d_items.csv.gz"),
                           usecols=["itemid", "label"], compression="gzip")
    item_to_label = dict(zip(d_items["itemid"], d_items["label"]))

    path = _tbl(mimic_dir, "icu", "inputevents.csv.gz")
    keep_stays = set(icustays["stay_id"].astype(np.int64))
    usecols = ["stay_id", "starttime", "itemid"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, compression="gzip", chunksize=2_000_000,
                              parse_dates=["starttime"]):
        chunk = chunk.dropna(subset=usecols)
        chunk["stay_id"] = chunk["stay_id"].astype(np.int64)
        chunk = chunk[chunk["stay_id"].isin(keep_stays)]
        if len(chunk):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

    df["item"] = df["itemid"].map(item_to_label).fillna("UNKNOWN_ITEM")
    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")

    icu_small = icustays[["stay_id", "intime"]]
    df = df.merge(icu_small, on="stay_id", how="left")
    out = pd.DataFrame({
        "stay_id": df["stay_id"],
        "offset_minutes": _offset_minutes_col(df, "starttime"),
        "poi_id": "infusion::" + df["item"].astype(str),
        "table_name": "infusion",
    })
    print(f"  infusion: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


def extract_vital_events(mimic_dir: str, icustays: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """icu/chartevents.csv.gz restricted to a fixed itemid list (VITAL_ITEMID_SLOTS).

    One event per row with a non-null valuenum. Read in chunks, filtered by itemid
    and stay_id.
    """
    print("Step 1e: Extracting vital events (icu/chartevents.csv.gz, curated subset)")
    # MIMIC-IV 3.1 icu/d_items itemid -> vital token. Token names follow eICU's
    # vitalAperiodic column names where one exists; cardiac-output itemids from
    # different devices share one token. 226865 / 226863 ("SVR/PVR %O2 Saturation
    # (PA Line)") are oxygen saturations and are not used.
    VITAL_ITEMID_SLOTS = {
        220179: "noninvasivesystolic",
        220180: "noninvasivediastolic",
        220181: "noninvasivemean",
        220088: "cardiacoutput",        # CO (thermodilution)
        224842: "cardiacoutput",        # CO (CCO)
        228369: "cardiacoutput",        # CO (NICOM)
        229897: "cardiacoutput",        # CO (Impella)
        228368: "cardiacindex",         # CI (NICOM)
        228185: "svri",                 # SVRI (PiCCO)
    }
    d_items = pd.read_csv(_tbl(mimic_dir, "icu", "d_items.csv.gz"),
                           usecols=["itemid", "label"], compression="gzip")
    present = set(d_items["itemid"].astype(int))
    missing = sorted(set(VITAL_ITEMID_SLOTS) - present)
    if missing:
        raise SystemExit(
            f"{len(missing)} expected vital itemid(s) absent from "
            f"{_tbl(mimic_dir, 'icu', 'd_items.csv.gz')}: {missing}. "
            "VITAL_ITEMID_SLOTS is defined for the MIMIC-IV 3.1 item dictionary."
        )
    item_to_label = dict(VITAL_ITEMID_SLOTS)
    keep_itemids = set(item_to_label)

    path = _tbl(mimic_dir, "icu", "chartevents.csv.gz")
    keep_stays = set(icustays["stay_id"].astype(np.int64))
    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, compression="gzip", chunksize=5_000_000,
                              parse_dates=["charttime"]):
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        if not len(chunk):
            continue
        chunk = chunk.dropna(subset=["stay_id", "charttime", "valuenum"])
        chunk["stay_id"] = chunk["stay_id"].astype(np.int64)
        chunk = chunk[chunk["stay_id"].isin(keep_stays)]
        if len(chunk):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

    df["item"] = df["itemid"].map(item_to_label).fillna("UNKNOWN_ITEM")
    counts = df["item"].value_counts()
    keep = set(counts.head(top_k).index)
    df["item"] = df["item"].where(df["item"].isin(keep), "OTHER")

    icu_small = icustays[["stay_id", "intime"]]
    df = df.merge(icu_small, on="stay_id", how="left")
    out = pd.DataFrame({
        "stay_id": df["stay_id"],
        "offset_minutes": _offset_minutes_col(df, "charttime"),
        "poi_id": "vital::" + df["item"].astype(str),
        "table_name": "vital",
    })
    print(f"  vital: {len(out):,} events  vocab={out['poi_id'].nunique()}")
    return out


# ---------------------------------------------------------------------------
# Step 3: UMAP coordinates over the (POI x stay) co-occurrence matrix
# ---------------------------------------------------------------------------
def fit_umap_poi_coords(events: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    """2D UMAP (cosine) of the POI x stay count matrix, rescaled to [0, 1]^2."""
    print("\nStep 3: UMAP POI coordinates")
    t0 = time_mod.time()
    pois = sorted(events["poi_id"].unique())
    stays = sorted(events["stay_id"].unique())
    poi_idx = {p: i for i, p in enumerate(pois)}
    stay_idx = {s: i for i, s in enumerate(stays)}

    rows = events["poi_id"].map(poi_idx).to_numpy()
    cols = events["stay_id"].map(stay_idx).to_numpy()
    data = np.ones(len(rows), dtype=np.int32)
    M = csr_matrix((data, (rows, cols)), shape=(len(pois), len(stays)))
    M.sum_duplicates()
    print(f"  matrix: {M.shape[0]:,} POIs x {M.shape[1]:,} stays  nnz={M.nnz:,}")

    n_neighbors = max(2, min(30, M.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=n_neighbors,
                         min_dist=0.1, random_state=rng_seed, verbose=False)
    coords = reducer.fit_transform(M)
    mn, mx = coords.min(0), coords.max(0)
    span = np.where(mx - mn > 0, mx - mn, 1.0)
    coords = (coords - mn) / span
    print(f"  UMAP done in {time_mod.time()-t0:.0f}s")

    return pd.DataFrame({
        "poi_id": pois,
        "longitude": coords[:, 0].astype(np.float64),
        "latitude": coords[:, 1].astype(np.float64),
    })


def build_poi_table(poi_coords: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add venue_type (= source table) and act_types (cat_id list)."""
    poi = poi_coords.copy()
    poi["venue_type"] = poi["poi_id"].str.split("::").str[0]
    cats = sorted(poi["venue_type"].unique())
    cat_to_id = {c: i for i, c in enumerate(cats)}
    poi["act_types"] = poi["venue_type"].map(lambda c: [int(cat_to_id[c])])
    print(f"  venue types: {cats}  ({len(cats)} categories)")
    return poi[["poi_id", "longitude", "latitude", "venue_type", "act_types"]], cat_to_id


# ---------------------------------------------------------------------------
# Step 4: Patient-level stratified split (70/15/15)
# ---------------------------------------------------------------------------
def patient_level_split(stay_labels: pd.DataFrame, rng_seed: int) -> dict[str, set]:
    """Split by uniquepid (MIMIC-IV subject_id) so a patient's stays never appear in
    two splits. Stratified on mortality at the patient level (positive if any stay
    expired).
    """
    print(f"\nStep 4: Patient-level split ({TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%})")
    pat_label = stay_labels.groupby("uniquepid")["mortality"].max().rename("any_mortality")
    pos_pids = pat_label[pat_label == 1].index.tolist()
    neg_pids = pat_label[pat_label == 0].index.tolist()

    rng = np.random.default_rng(rng_seed)
    rng.shuffle(pos_pids); rng.shuffle(neg_pids)

    def _split(pids):
        n = len(pids); n_tr = int(round(n * TRAIN_FRAC)); n_va = int(round(n * VAL_FRAC))
        return pids[:n_tr], pids[n_tr:n_tr+n_va], pids[n_tr+n_va:]

    tr_pos, va_pos, te_pos = _split(pos_pids)
    tr_neg, va_neg, te_neg = _split(neg_pids)
    splits = {
        "train": set(tr_pos) | set(tr_neg),
        "val": set(va_pos) | set(va_neg),
        "test": set(te_pos) | set(te_neg),
    }
    for name, pids in splits.items():
        sub = stay_labels[stay_labels["uniquepid"].isin(pids)]
        print(f"  {name}: {len(pids):,} patients, {len(sub):,} stays, "
              f"mortality={sub['mortality'].mean():.4f}")
    return splits


# ---------------------------------------------------------------------------
# Step 5: Assemble the per-event stay dataframe with timestamps + agent ids
# ---------------------------------------------------------------------------
def assemble_stays(events: pd.DataFrame, poi: pd.DataFrame,
                    stay_labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach agent ids (0..N-1 over stays sorted by stay_id), synthetic timestamps,
    mortality and POI coordinates to each event.
    """
    print("\nStep 5: Assembling stays")
    valid_stays = set(stay_labels["patientunitstayid"].astype(np.int64))
    before = len(events)
    events = events[events["stay_id"].isin(valid_stays)].copy()
    print(f"  events kept (valid stays): {len(events):,} / {before:,}")

    stay_ids = sorted(valid_stays)
    stay_to_agent = {s: i for i, s in enumerate(stay_ids)}
    events["agent"] = events["stay_id"].map(stay_to_agent).astype(np.int64)

    events["start"] = BASE_DATETIME + pd.to_timedelta(events["offset_minutes"], unit="m")
    events["stop"] = events["start"]

    stay_to_mort = dict(zip(stay_labels["patientunitstayid"].astype(np.int64),
                             stay_labels["mortality"].astype(np.int64)))
    events["mortality"] = events["stay_id"].map(stay_to_mort).astype(np.int64)
    events["anomaly"] = 0

    events = events.merge(
        poi[["poi_id", "longitude", "latitude", "venue_type", "act_types"]],
        on="poi_id", how="left",
    )
    missing = events["longitude"].isna().sum()
    if missing > 0:
        print(f"  [warn] {missing:,} events with no UMAP coord (dropping)")
        events = events.dropna(subset=["longitude", "latitude"]).copy()

    events["cat_0"] = events["act_types"].apply(
        lambda xs: float(xs[0]) if isinstance(xs, list) and xs else 0.0
    )
    # eICU column name; holds the MIMIC-IV stay_id.
    events["patientunitstayid"] = events["stay_id"]

    events = events.sort_values(["agent", "start"]).reset_index(drop=True)
    print(f"  final events: {len(events):,}  agents={events['agent'].nunique():,}  "
          f"POIs={events['poi_id'].nunique():,}")
    return events, stay_to_agent


# ---------------------------------------------------------------------------
# Step 6: Save outputs
# ---------------------------------------------------------------------------
def save_outputs(out_dir: str, tag: str, train_events: pd.DataFrame, val_events: pd.DataFrame,
                  test_events: pd.DataFrame, poi: pd.DataFrame, stay_labels: pd.DataFrame,
                  stay_to_agent: dict) -> None:
    """Write the train (train + val patients), test, POI, stay-label, metadata and
    AOI files.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_cols = [
        "agent", "longitude", "latitude", "poi_id", "act_types", "cat_0",
        "start", "stop", "venue_type", "anomaly", "mortality", "patientunitstayid",
    ]
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

    # Per-stay label table, tagged with each stay's split.
    pl = stay_labels.copy()
    train_stays = set(train_events["patientunitstayid"].astype(np.int64).unique())
    val_stays = set(val_events["patientunitstayid"].astype(np.int64).unique())
    test_stays = set(test_events["patientunitstayid"].astype(np.int64).unique())

    def _split_of(s):
        if s in train_stays: return "train"
        if s in val_stays: return "val"
        if s in test_stays: return "test"
        return "drop"
    pl["split"] = pl["patientunitstayid"].astype(np.int64).map(_split_of)
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
    meta_df = pd.DataFrame({
        "min_time": [min_time], "local_tz": ["UTC"],
        "timestamp_step_size": [step_s / 3600.0], "utm_crs": ["EPSG:4326"],
        "n_agents": [int(len(stay_to_agent))],
    })
    meta_df.to_parquet(meta_path, index=False)
    print(f"  saved meta -> {meta_path}  n_agents={len(stay_to_agent)}")

    # AOI = bounding box of UMAP coords.
    minx, maxx = all_events["longitude"].min(), all_events["longitude"].max()
    miny, maxy = all_events["latitude"].min(), all_events["latitude"].max()
    aoi_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    aoi_gdf.to_file(aoi_path, driver="GeoJSON")
    print(f"  saved aoi -> {aoi_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mimic-dir", default=MIMIC_DIR,
                    help="MIMIC-IV 3.1 root directory (contains hosp/ and icu/)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--tag", default="mimiciv-full",
                    help="filename stem for output parquet files (matches config.py)")
    p.add_argument("--seed", type=int, default=RNG_SEED)
    return p.parse_args(argv)


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Step 0: Loading ICU stay index")
    icustays = load_icustays(args.mimic_dir)
    print(f"  raw ICU stays: {len(icustays):,}")

    stay_labels = load_stay_labels(args.mimic_dir, icustays)

    print("\nStep 1: Extracting event streams from five MIMIC-IV tables")
    events = pd.concat([
        extract_lab_events(args.mimic_dir, icustays, TOP_K_PER_TABLE["lab"]),
        extract_med_events(args.mimic_dir, icustays, TOP_K_PER_TABLE["med"]),
        extract_treat_events(args.mimic_dir, icustays, TOP_K_PER_TABLE["treat"]),
        extract_infusion_events(args.mimic_dir, icustays, TOP_K_PER_TABLE["infusion"]),
        extract_vital_events(args.mimic_dir, icustays, TOP_K_PER_TABLE["vital"]),
    ], ignore_index=True)
    print(f"\nTotal raw events: {len(events):,}  unique POIs: {events['poi_id'].nunique():,}")

    splits = patient_level_split(stay_labels, args.seed)

    # UMAP is fit on all events so train/test share one embedding.
    poi_coords = fit_umap_poi_coords(events, args.seed)
    poi, cat_to_id = build_poi_table(poi_coords)

    events, stay_to_agent = assemble_stays(events, poi, stay_labels)

    # Split events into train/val/test by uniquepid.
    pid_of_stay = dict(zip(stay_labels["patientunitstayid"].astype(np.int64),
                            stay_labels["uniquepid"]))
    events["uniquepid"] = events["patientunitstayid"].map(pid_of_stay)
    train_events = events[events["uniquepid"].isin(splits["train"])].drop(columns=["uniquepid"]).copy()
    val_events = events[events["uniquepid"].isin(splits["val"])].drop(columns=["uniquepid"]).copy()
    test_events = events[events["uniquepid"].isin(splits["test"])].drop(columns=["uniquepid"]).copy()
    print(f"  events  train={len(train_events):,}  val={len(val_events):,}  "
          f"test={len(test_events):,}")

    save_outputs(args.out_dir, args.tag, train_events, val_events, test_events,
                 poi, stay_labels, stay_to_agent)

    # Sidecar: cat-id mapping.
    pd.DataFrame([{"venue_type": k, "cat_id": v}
                  for k, v in sorted(cat_to_id.items(), key=lambda kv: kv[1])]
                 ).to_csv(os.path.join(args.out_dir, f"{args.tag}_categories.csv"), index=False)

    print("\n=== done ===")


if __name__ == "__main__":
    main()

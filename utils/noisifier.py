"""Data augmentation module for generating anomalous mobility patterns.

This module provides the Noisifier class and related functions for creating
synthetic anomalies in mobility data following the NUMOSIM taxonomy:

- **Non-recurring anomalies** (anomaly_type="non_recurring"): singular, isolated
  deviations — a random new stay, or a one-time location/time perturbation.
- **Recurring anomalies** (anomaly_type="recurring"): pattern-level deviations
  that repeat at a regular cadence — e.g. visiting the same new location every
  weekend or every other day.

Semantic anomaly types (each with yellow/orange/red intensity):
- **Hunger anomalies** (anomaly_type="hunger"): insert extra food POI visits to
  simulate an increased food need driving agents to eat more frequently.
- **Interest anomalies** (anomaly_type="interest"): replace some recreational
  visits with POIs of a different recreational act_type, simulating a change in
  the agent's preferred leisure activity.
- **Social anomalies** (anomaly_type="social"): replace recreational visits with
  randomly chosen recreational POIs, disrupting the agent's social network by
  sending them to unfamiliar sites instead of habitual meeting places.
- **Work anomalies** (anomaly_type="work"): replace work POI visits with
  non-work activities, simulating an agent who stops going to work.
- **Replace-agent anomalies** (anomaly_type="replace_agent"): substitute the
  entire stay sequence with that of a randomly chosen different agent.  Every
  stay in the resulting sequence is anomalous because the observed mobility
  pattern belongs to a different individual.

Each anomalous stay carries:
  - stay_anomalous  : True whenever the stay is anomalous for any reason
  - insert_anomalous: True when the stay was synthetically inserted
  - loc_anomalous   : True when the location was perturbed
  - time_anomalous  : True when the timestamps were perturbed
  - anomaly_type    : "non_recurring" | "recurring" | "hunger" | "interest" |
                      "social" | "work" | None (normal stays)

The Noisifier is used during training to create a balanced dataset of normal
and anomalous mobility patterns for supervised learning.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from config import DatasetConfig
from utils.file_io import load_aoi, load_aoi_box_padded
from utils.misc import sample_from_aoi, sample_from_aoi_box

pd.set_option("future.no_silent_downcasting", True)

# Fraction of stays to affect at each intensity level (interest/social/work anomalies).
# Derived from outlierDegree values in Urban Anomalies paper (arxiv 2410.01844):
#   yellow = 0.20, orange = 0.50, red = 1.0.
_INTENSITY_FRACTION = {"yellow": 0.20, "orange": 0.50, "red": 1.0}

# Number of extra food visits per intensity level (hunger anomaly).
# Approximated from the paper's food-need multipliers:
#   yellow: 1.5× rate → +1–2 visits, orange: 2× rate → +3–4, red: perpetual → +5–7.
_HUNGER_N_EXTRA = {"yellow": (1, 2), "orange": (3, 4), "red": (5, 7)}

# Intensity levels and anomaly type pool used for uniform sampling (per paper).
_INTENSITY_LEVELS = ("yellow", "orange", "red")
_BEHAVIORAL_TYPES = ("hunger", "interest", "social", "work")

# Per-dataset act_type code mappings for behavioral anomalies.
# Based on NUMOSIM-LA activity type documentation (arxiv 2409.03024):
#   act_type 2  → Work          (409,920 POIs)
#   act_type 7  → EatOut / food (165,442 POIs)
#   act_type 9  → Recreation    (17,685 POIs)
#   act_type 10 → Exercise      (30,520 POIs)
#
# Urban Anomalies Berlin/Atlanta POI venue_type → act_type mapping:
#   act_type 0 → Apartment      (residential; not a behavioral anomaly target)
#   act_type 1 → Pub            (social/recreational)
#   act_type 2 → Restaurant     (food/dining)
#   act_type 3 → Workplace      (work)
_BEHAVIORAL_ACT_TYPES: dict[str, dict[str, list[int]]] = {
    "numosim": {
        "food": [7],
        "work": [2],
        "recreational": [9, 10],
    },
    "urban-berlin": {
        "food": [2],           # Restaurants
        "work": [3],           # Workplaces
        "recreational": [1],   # Pubs
    },
    "urban-atlanta": {
        "food": [2],           # Restaurants
        "work": [3],           # Workplaces
        "recreational": [1],   # Pubs
    },
}


def _filter_pois_by_act_types(pois: pd.DataFrame, act_types: list) -> pd.DataFrame:
    """Return the subset of POIs that have any of the given act_type codes."""
    col_names = [f"cat_{t}" for t in act_types]
    present = [c for c in col_names if c in pois.columns]
    if not present:
        return pois.iloc[0:0]
    return pois[(pois[present] > 0).any(axis=1)]


def _get_stay_act_type_mask(stays: pd.DataFrame, act_types: list) -> pd.Series:
    """Return a boolean mask for stays whose POI has any of the given act_type codes."""
    col_names = [f"cat_{t}" for t in act_types]
    present = [c for c in col_names if c in stays.columns]
    if not present:
        return pd.Series(False, index=stays.index)
    return (stays[present] > 0).any(axis=1)


def _insert_single_new_stay(agent_stays, new_stay_row):
    """
    Inserts a single new stay into a sorted list of agent_stays, handling overlaps.
    Assumes new_stay_row is a Series representing a single stay.
    """
    new_start = new_stay_row["start"]
    new_stop = new_stay_row["stop"]

    # 1. Identify rows that have ANY overlap with the new stay
    overlap_mask = (agent_stays["start"] < new_stop) & (agent_stays["stop"] > new_start)

    # 2. Filter out stays that are completely contained by the new stay
    # These are stays where (new_start <= start) AND (stop <= new_stop)
    contained_mask = (agent_stays["start"] >= new_start) & (
        agent_stays["stop"] <= new_stop
    )

    # Stays to KEEP (no overlap or only partial overlap)
    stays_to_process = agent_stays[overlap_mask & ~contained_mask].copy()

    # Stays to REMAIN completely untouched
    untouched_stays = agent_stays[~overlap_mask].copy()

    # 3. Handle Partial Overlap (Trimming)
    trimmed_stays = []
    # Trimmed stays inherit the anomaly_type of the stay that displaced them
    new_anomaly_type = new_stay_row.get("anomaly_type", None)

    for _, row in stays_to_process.iterrows():
        original_start = row["start"]
        original_stop = row["stop"]

        # Case A: Left part of the existing stay remains (row["start"] < new_start)
        if original_start < new_start:
            left_split = row.copy()
            left_split["stop"] = min(
                original_stop, new_start
            )  # Trims the end to new_start
            left_split["stay_anomalous"] = True
            left_split["insert_anomalous"] = True
            left_split["anomaly_type"] = new_anomaly_type
            trimmed_stays.append(left_split)

        # Case B: Right part of the existing stay remains (row["stop"] > new_stop)
        if original_stop > new_stop:
            right_split = row.copy()
            right_split["start"] = max(
                original_start, new_stop
            )  # Trims the start to new_stop
            right_split["stay_anomalous"] = True
            right_split["insert_anomalous"] = True
            right_split["anomaly_type"] = new_anomaly_type
            trimmed_stays.append(right_split)

    # 4. Combine all resulting stays: untouched, trimmed, and the new stay

    # Convert the new stay row (Series) back to a DataFrame for concatenation
    new_stay_df = pd.DataFrame([new_stay_row])

    all_stays = pd.concat(
        [untouched_stays, new_stay_df, pd.DataFrame(trimmed_stays)], ignore_index=True
    )

    # 5. Re-sort the final list by start time
    all_stays = all_stays.sort_values(by="start").reset_index(drop=True)

    all_stays = all_stays.drop_duplicates(subset=["start", "stop", "agent"])
    all_stays["stay_anomalous"] = all_stays["stay_anomalous"].fillna(False).astype(bool)
    all_stays["insert_anomalous"] = all_stays["insert_anomalous"].fillna(False).astype(bool)
    all_stays["loc_anomalous"] = all_stays["loc_anomalous"].fillna(False).astype(bool)
    all_stays["time_anomalous"] = all_stays["time_anomalous"].fillna(False).astype(bool)
    # anomaly_type stays as None/NaN for normal stays — do not fill with False
    if "anomaly_type" not in all_stays.columns:
        all_stays["anomaly_type"] = None
    return all_stays


def add_random_stays(agent_stays, pois, categories, nearest_poi_fn, aoi_box=None, aoi=None):
    assert len(agent_stays) > 0, "agent_stays must not be empty"
    if "agent" not in agent_stays.columns:
        raise KeyError("Expected `agent` column in agent_stays")

    # 1. Determine the number of new stays
    num_new_stays = np.random.binomial(6, 0.25)

    if num_new_stays == 0:
        agent_stays["stay_anomalous"] = agent_stays["stay_anomalous"].fillna(False).astype(bool)
        agent_stays["insert_anomalous"] = agent_stays["insert_anomalous"].fillna(False).astype(bool)
        return agent_stays

    # 2. Generate multiple new stays
    min_date = agent_stays["start"].min() // 24
    max_date = agent_stays["stop"].max() // 24

    new_stays_list = []
    for _ in range(num_new_stays):
        # Generate random time/duration for the new stay
        start_day = np.random.randint(min_date, max_date + 1)
        start_hod = np.random.rand() * 24
        duration_hours = np.random.rand() * 6 + 5 / 60  # scale to [5min, 6h]

        # Add a random travel buffer (10s to 30min) to the start/stop times
        start_travel = np.random.rand() * 25 / 60 + 10 / 3600
        end_travel = np.random.rand() * 25 / 60 + 10 / 3600

        new_start = start_day * 24 + start_hod - start_travel
        new_stop = start_day * 24 + start_hod + duration_hours + end_travel

        # Sample a random location, then attribute to the nearest POI.
        if aoi_box is not None:
            lon, lat = sample_from_aoi_box(aoi_box, 1).reshape(-1).tolist()
        elif aoi is not None:
            lon, lat = sample_from_aoi(aoi)
        else:
            lon, lat = agent_stays[["longitude", "latitude"]].sample(1).values[0]

        nearest = nearest_poi_fn(lat, lon)

        stay_data = {
            "agent": agent_stays["agent"].iloc[0],
            "start": new_start,
            "stop": new_stop,
            "latitude": lat,
            "longitude": lon,
            "poi_id": nearest["poi_id"],
            "stay_anomalous": True,
            "insert_anomalous": True,
            "anomaly_type": "non_recurring",
        }

        # Use nearest POI's categories
        for category in categories:
            stay_data[category] = nearest.get(category, 0)

        new_stays_list.append(stay_data)

    new_stays_df = pd.DataFrame(new_stays_list)

    # 3. Insert each new stay sequentially
    current_agent_stays = agent_stays

    # Sort new stays by start time to process them in chronological order
    new_stays_df = new_stays_df.sort_values(by="start")

    for _, new_stay_row in new_stays_df.iterrows():
        # The key is to repeatedly apply the single insertion logic
        current_agent_stays = _insert_single_new_stay(current_agent_stays, new_stay_row)

    return current_agent_stays


def add_recurring_stays(agent_stays, pois, categories, nearest_poi_fn, aoi_box=None, aoi=None):
    """Insert stays at a regular cadence to simulate recurring anomalous behavior.

    Picks a random period (1, 2, 3, or 7 days) and a fixed anomalous location,
    then inserts a visit to that location at each occurrence of the period within
    the agent's date range.  All inserted stays share the same location and a
    consistent time-of-day (with small jitter) so the pattern looks intentional.

    Args:
        agent_stays: DataFrame of stays for a single agent.
        pois: DataFrame of Points of Interest used for category sampling.
        categories: List of category column names.
        aoi_box: Optional bounding box for random location sampling.
        aoi: Optional AOI polygon for random location sampling.

    Returns:
        DataFrame with recurring anomalous stays inserted and labelled
        anomaly_type="recurring".
    """
    assert len(agent_stays) > 0, "agent_stays must not be empty"

    min_date = int(agent_stays["start"].min() // 24)
    max_date = int(agent_stays["stop"].max() // 24)
    date_range = max_date - min_date

    if date_range < 2:
        # Not enough span to establish a recognisable recurring pattern
        return agent_stays

    period_days = int(np.random.choice([1, 2, 3, 7]))
    first_day = min_date + np.random.randint(0, min(period_days, date_range))
    occurrence_days = list(range(int(first_day), max_date + 1, period_days))

    if len(occurrence_days) < 2:
        return agent_stays

    # Fixed anomalous location — same place every occurrence; attribute to nearest POI.
    if aoi_box is not None:
        lon, lat = sample_from_aoi_box(aoi_box, 1).reshape(-1).tolist()
    elif aoi is not None:
        lon, lat = sample_from_aoi(aoi)
    else:
        lon, lat = agent_stays[["longitude", "latitude"]].sample(1).values[0]

    nearest = nearest_poi_fn(lat, lon)

    # Consistent time-of-day with small per-occurrence jitter (±15 min)
    base_hod = np.random.rand() * 24
    base_duration = np.random.rand() * 3.0 + 0.5  # 30 min – 3.5 h

    seq_start = agent_stays["start"].min()
    seq_stop = agent_stays["stop"].max()

    new_stays_list = []
    for day in occurrence_days:
        hod = base_hod + np.random.randn() * 0.25
        duration_hours = max(base_duration + np.random.randn() * 0.1, 5 / 60)

        start_travel = np.random.rand() * 25 / 60 + 10 / 3600
        end_travel = np.random.rand() * 25 / 60 + 10 / 3600

        new_start = day * 24 + hod - start_travel
        new_stop = day * 24 + hod + duration_hours + end_travel

        # Only insert visits that fall strictly within the sequence's time bounds.
        # Off-limit recurring visits exist conceptually but are not observable here.
        if new_start <= seq_start or new_stop >= seq_stop:
            continue

        stay_data = {
            "agent": agent_stays["agent"].iloc[0],
            "start": new_start,
            "stop": new_stop,
            "latitude": lat,
            "longitude": lon,
            "poi_id": nearest["poi_id"],
            "stay_anomalous": True,
            "insert_anomalous": True,
            "anomaly_type": "recurring",
        }
        for category in categories:
            stay_data[category] = nearest.get(category, 0)
        new_stays_list.append(stay_data)

    if not new_stays_list:
        return agent_stays
    new_stays_df = pd.DataFrame(new_stays_list).sort_values(by="start")

    current = agent_stays
    for _, row in new_stays_df.iterrows():
        current = _insert_single_new_stay(current, row)
    return current


def apply_hunger_anomaly(agent_stays, food_pois, categories, intensity="orange"):
    """Insert extra food POI visits to simulate an increased food need.

    Intensity controls the number of extra visits inserted:
        yellow: 1–2, orange: 3–4, red: 5–7.

    Args:
        agent_stays: DataFrame of stays for a single agent.
        food_pois: Pre-filtered DataFrame of food POIs.
        categories: List of category column names.
        intensity: "yellow" | "orange" | "red"

    Returns:
        DataFrame with extra food visits inserted and labelled anomaly_type="hunger".
    """
    if len(food_pois) == 0:
        return agent_stays

    lo, hi = _HUNGER_N_EXTRA[intensity]
    num_new = np.random.randint(lo, hi + 1)
    if num_new == 0:
        return agent_stays

    min_date = int(agent_stays["start"].min() // 24)
    max_date = int(agent_stays["stop"].max() // 24)
    ts_min = agent_stays["start"].min()
    ts_max = agent_stays["stop"].max()

    new_stays_list = []
    for _ in range(num_new):
        start_day = np.random.randint(min_date, max_date + 1)
        start_hod = np.random.rand() * 24
        duration_hours = np.random.rand() * 1.5 + 0.25  # 15 min – 1.75 h (short food visit)

        start_travel = np.random.rand() * 25 / 60 + 10 / 3600
        end_travel = np.random.rand() * 25 / 60 + 10 / 3600

        stay_start = start_day * 24 + start_hod - start_travel
        stay_stop = start_day * 24 + start_hod + duration_hours + end_travel
        if stay_start < ts_min or stay_stop > ts_max:
            continue  # outside sequence bounds; would be trivially detectable

        poi = food_pois.sample(1).iloc[0]
        stay_data = {
            "agent": agent_stays["agent"].iloc[0],
            "start": stay_start,
            "stop": stay_stop,
            "latitude": poi["latitude"],
            "longitude": poi["longitude"],
            "poi_id": poi["poi_id"],
            "stay_anomalous": True,
            "insert_anomalous": True,
            "anomaly_type": "hunger",
        }
        for cat in categories:
            stay_data[cat] = poi.get(cat, 0)
        new_stays_list.append(stay_data)

    if not new_stays_list:
        return agent_stays
    new_stays_df = pd.DataFrame(new_stays_list).sort_values(by="start")
    current = agent_stays
    for _, row in new_stays_df.iterrows():
        current = _insert_single_new_stay(current, row)
    return current


def apply_interest_anomaly(agent_stays, pois, categories, recreational_act_types, intensity="orange"):
    """Replace a fraction of recreational visits with POIs of a different act_type.

    Infers the agent's current "interest" as the most common recreational act_type
    in their history, then redirects a fraction of visits to POIs of a different
    recreational act_type.

    Intensity controls the fraction of recreational stays affected:
        yellow: 25%, orange: 50%, red: 80%.

    Args:
        agent_stays: DataFrame of stays for a single agent.
        pois: Full POI DataFrame (must include category columns).
        categories: List of category column names.
        recreational_act_types: List of act_type int codes considered recreational.
        intensity: "yellow" | "orange" | "red"

    Returns:
        DataFrame with some recreational visits' POIs replaced (loc_anomalous=True,
        anomaly_type="interest").
    """
    if len(recreational_act_types) < 2:
        return agent_stays

    fraction = _INTENSITY_FRACTION[intensity]
    rec_mask = _get_stay_act_type_mask(agent_stays, recreational_act_types)
    rec_idx = agent_stays.index[rec_mask].tolist()
    if not rec_idx:
        return agent_stays

    # Infer agent's current dominant recreational interest
    type_counts = {}
    for t in recreational_act_types:
        col = f"cat_{t}"
        if col in agent_stays.columns:
            type_counts[t] = agent_stays.loc[rec_mask, col].sum()
    dominant = max(type_counts, key=type_counts.get) if type_counts else None

    # Choose a different act_type as the new interest
    other_types = [t for t in recreational_act_types if t != dominant]
    if not other_types:
        return agent_stays
    new_interest = int(np.random.choice(other_types))
    new_interest_pois = _filter_pois_by_act_types(pois, [new_interest])
    if len(new_interest_pois) == 0:
        return agent_stays

    n_to_replace = max(1, int(len(rec_idx) * fraction))
    selected_idx = np.random.choice(rec_idx, size=min(n_to_replace, len(rec_idx)), replace=False)

    stays_copy = agent_stays.copy()
    for idx in selected_idx:
        new_poi = new_interest_pois.sample(1).iloc[0]
        stays_copy.at[idx, "latitude"] = new_poi["latitude"]
        stays_copy.at[idx, "longitude"] = new_poi["longitude"]
        stays_copy.at[idx, "poi_id"] = new_poi["poi_id"]
        for cat in categories:
            stays_copy.at[idx, cat] = new_poi.get(cat, 0)
        stays_copy.at[idx, "stay_anomalous"] = True
        stays_copy.at[idx, "loc_anomalous"] = True
        stays_copy.at[idx, "anomaly_type"] = "interest"
    return stays_copy


def apply_social_anomaly(agent_stays, recreational_pois, categories, recreational_act_types, intensity="orange"):
    """Replace recreational visits with randomly chosen recreational POIs.

    Agents normally visit specific recreational sites based on their interests and
    social connections. This anomaly replaces those specific choices with random
    recreational POIs, disrupting their social network.

    Intensity controls the fraction of recreational stays affected:
        yellow: 25%, orange: 50%, red: 80%.

    Args:
        agent_stays: DataFrame of stays for a single agent.
        recreational_pois: Pre-filtered DataFrame of recreational POIs.
        categories: List of category column names.
        recreational_act_types: List of act_type int codes considered recreational.
        intensity: "yellow" | "orange" | "red"

    Returns:
        DataFrame with some recreational visits' POIs randomised (loc_anomalous=True,
        anomaly_type="social").
    """
    if len(recreational_pois) == 0:
        return agent_stays

    fraction = _INTENSITY_FRACTION[intensity]
    rec_mask = _get_stay_act_type_mask(agent_stays, recreational_act_types)
    rec_idx = agent_stays.index[rec_mask].tolist()
    if not rec_idx:
        return agent_stays

    n_to_replace = max(1, int(len(rec_idx) * fraction))
    selected_idx = np.random.choice(rec_idx, size=min(n_to_replace, len(rec_idx)), replace=False)

    stays_copy = agent_stays.copy()
    for idx in selected_idx:
        new_poi = recreational_pois.sample(1).iloc[0]
        stays_copy.at[idx, "latitude"] = new_poi["latitude"]
        stays_copy.at[idx, "longitude"] = new_poi["longitude"]
        stays_copy.at[idx, "poi_id"] = new_poi["poi_id"]
        for cat in categories:
            stays_copy.at[idx, cat] = new_poi.get(cat, 0)
        stays_copy.at[idx, "stay_anomalous"] = True
        stays_copy.at[idx, "loc_anomalous"] = True
        stays_copy.at[idx, "anomaly_type"] = "social"
    return stays_copy


def apply_work_anomaly(agent_stays, non_work_pois, categories, work_act_types, intensity="orange"):
    """Replace work POI visits with non-work activities.

    Simulates an agent who stops going to work, redirecting their work visits
    to other (non-work) locations.

    Intensity controls the fraction of work stays affected:
        yellow: 25%, orange: 50%, red: 80%.

    Args:
        agent_stays: DataFrame of stays for a single agent.
        non_work_pois: Pre-filtered DataFrame of non-work POIs.
        categories: List of category column names.
        work_act_types: List of act_type int codes considered work.
        intensity: "yellow" | "orange" | "red"

    Returns:
        DataFrame with some work visits replaced by non-work POIs (loc_anomalous=True,
        anomaly_type="work").
    """
    if len(non_work_pois) == 0:
        return agent_stays

    fraction = _INTENSITY_FRACTION[intensity]
    work_mask = _get_stay_act_type_mask(agent_stays, work_act_types)
    work_idx = agent_stays.index[work_mask].tolist()
    if not work_idx:
        return agent_stays

    n_to_replace = max(1, int(len(work_idx) * fraction))
    selected_idx = np.random.choice(work_idx, size=min(n_to_replace, len(work_idx)), replace=False)

    stays_copy = agent_stays.copy()
    for idx in selected_idx:
        new_poi = non_work_pois.sample(1).iloc[0]
        stays_copy.at[idx, "latitude"] = new_poi["latitude"]
        stays_copy.at[idx, "longitude"] = new_poi["longitude"]
        stays_copy.at[idx, "poi_id"] = new_poi["poi_id"]
        for cat in categories:
            stays_copy.at[idx, cat] = new_poi.get(cat, 0)
        stays_copy.at[idx, "stay_anomalous"] = True
        stays_copy.at[idx, "loc_anomalous"] = True
        stays_copy.at[idx, "anomaly_type"] = "work"
    return stays_copy


def apply_replace_agent(agent_stays: pd.DataFrame, all_agent_stays: pd.DataFrame) -> pd.DataFrame:
    """Substitute an agent's entire stay sequence with that of a different agent.

    Every stay in the replacement sequence is labelled anomalous because the
    observed mobility pattern belongs to a different individual.

    Args:
        agent_stays: DataFrame of stays for the current agent.
        all_agent_stays: DataFrame of stays for all agents in the dataset.

    Returns:
        DataFrame containing the replacement agent's stays with all stays
        labelled stay_anomalous=True and anomaly_type="replace_agent".
        The "agent" column is overwritten with the original agent's ID.
        Falls back to marking the original stays anomalous if no other
        agent is available.
    """
    original_agent = agent_stays["agent"].iloc[0]
    other_agents = all_agent_stays.loc[
        all_agent_stays["agent"] != original_agent, "agent"
    ].unique()

    if len(other_agents) == 0:
        result = agent_stays.copy()
        result["stay_anomalous"] = True
        result["insert_anomalous"] = False
        result["loc_anomalous"] = False
        result["time_anomalous"] = False
        result["anomaly_type"] = "replace_agent"
        return result

    replacement_agent = np.random.choice(other_agents)
    replacement_stays = (
        all_agent_stays[all_agent_stays["agent"] == replacement_agent]
        .copy()
        .reset_index(drop=True)
    )
    replacement_stays["agent"] = original_agent
    replacement_stays["stay_anomalous"] = True
    replacement_stays["insert_anomalous"] = False
    replacement_stays["loc_anomalous"] = False
    replacement_stays["time_anomalous"] = False
    replacement_stays["anomaly_type"] = "replace_agent"
    return replacement_stays


def apply_user_host_swap(
    agent_stays: pd.DataFrame,
    all_agent_stays: pd.DataFrame,
    swap_frac: float = 0.3,
) -> pd.DataFrame:
    """Per-event swap: replace the host fields of selected events with another user's host.

    Designed for LANL-style enterprise auth logs, where lateral-movement anomalies
    are characterised by a (legitimate) src_user authenticating to a dst_computer
    that does not belong to that user's normal host set.  We synthesise this by:
    for each event in the agent's sequence, with probability `swap_frac`, replace
    the host fields (longitude, latitude, poi_id, act_types, venue_type, cat_*)
    with those from a randomly sampled event by a different user, while keeping
    the original src_user and timestamps.  Hosts the original user has touched
    are excluded from the candidate pool.

    Args:
        agent_stays: DataFrame of stays for the current agent (single user).
        all_agent_stays: DataFrame of stays for all users (used as swap source).
        swap_frac: Per-event probability of being swapped.

    Returns:
        DataFrame with swapped host fields and stay_anomalous=True / loc_anomalous=True
        on the swapped rows; anomaly_type="user_host_swap".  Falls back to the
        original copy if no other users are available or if no swappable events.
    """
    out = agent_stays.copy()
    if len(out) == 0:
        return out

    original_user = out["agent"].iloc[0]
    other_mask = all_agent_stays["agent"] != original_user
    if not other_mask.any():
        return out

    other_stays = all_agent_stays.loc[other_mask]
    own_hosts = set(out["poi_id"].dropna().unique().tolist())

    swap_idx = out.index[np.random.rand(len(out)) < swap_frac]
    if len(swap_idx) == 0:
        return out

    cat_cols = [c for c in out.columns if c.startswith("cat_")]
    swap_cols = ["longitude", "latitude", "poi_id", "venue_type"] + cat_cols
    swap_cols = [c for c in swap_cols if c in out.columns and c in other_stays.columns]

    # Pre-filter candidate pool to events whose poi_id is not in own_hosts.
    if own_hosts:
        candidates = other_stays[~other_stays["poi_id"].isin(own_hosts)]
    else:
        candidates = other_stays
    if len(candidates) == 0:
        return out

    sampled = candidates.sample(n=len(swap_idx), replace=True)
    sampled.index = swap_idx

    for col in swap_cols:
        out.loc[swap_idx, col] = sampled[col]

    # act_types is a list/array per row — pandas handles it fine via .loc with object dtype.
    if "act_types" in out.columns and "act_types" in other_stays.columns:
        out.loc[swap_idx, "act_types"] = sampled["act_types"]

    out.loc[swap_idx, "stay_anomalous"] = True
    out.loc[swap_idx, "loc_anomalous"] = True
    # Don't overwrite a prior anomaly_type if already set.
    untyped = swap_idx[out.loc[swap_idx, "anomaly_type"].isna()]
    out.loc[untyped, "anomaly_type"] = "user_host_swap"
    return out


def insert_recurring_visit_anomaly(agent_stays, pois, categories, aoi_box=None, aoi=None):
    """Insert 4–10 recurring visits to the SAME novel POI, spaced ≥12h apart.

    Modelled on HAYSTAC schema types 1b (recurring area visits), 1c (recurring
    fixed-POI visits), and 2c (recurring two-location route).  The distinctive
    signal is that a single unfamiliar location is visited multiple times at
    regular cadence, unlike the one-off insertions of add_random_stays.

    Args:
        agent_stays: DataFrame of stays for a single agent.
        pois: Full POI DataFrame.
        categories: List of category column names.
        aoi_box / aoi: Unused; kept for API consistency.

    Returns:
        DataFrame with recurring covert-visit stays inserted, labelled
        anomaly_type="recurring_visit".
    """
    assert len(agent_stays) > 0, "agent_stays must not be empty"

    n_visits = np.random.randint(4, 11)  # 4–10 occurrences

    # Select a novel POI (not in agent's top-10 habitual locations)
    if "poi_id" in agent_stays.columns:
        habitual_pois = set(agent_stays["poi_id"].value_counts().head(10).index.tolist())
        candidate_pois = pois[~pois["poi_id"].isin(habitual_pois)]
        if len(candidate_pois) == 0:
            candidate_pois = pois
    else:
        candidate_pois = pois

    target_poi = candidate_pois.sample(1).iloc[0]

    min_start = agent_stays["start"].min()
    max_stop  = agent_stays["stop"].max()
    span_h = max_stop - min_start
    if span_h < 24:
        return agent_stays  # not enough span for recurring pattern

    # Space n_visits uniformly across the window with ≥12h gaps
    # Add some jitter (±2h) so visits don't look exactly periodic
    min_gap_h = max(12.0, span_h / (n_visits + 1))
    spacing = span_h / (n_visits + 1)

    new_stays_list = []
    for k in range(n_visits):
        jitter = np.random.uniform(-2.0, 2.0)
        anchor = min_start + spacing * (k + 1) + jitter
        duration_h = np.random.uniform(0.25, 1.5)  # 15min–1.5h per visit
        travel_buf = np.random.uniform(10 / 3600, 15 / 60)

        new_start = anchor - travel_buf
        new_stop  = anchor + duration_h + travel_buf

        # Clamp to sequence window
        if new_start <= min_start or new_stop >= max_stop:
            continue

        stay_data = {
            "agent":           agent_stays["agent"].iloc[0],
            "start":           new_start,
            "stop":            new_stop,
            "latitude":        target_poi["latitude"],
            "longitude":       target_poi["longitude"],
            "poi_id":          target_poi["poi_id"],
            "stay_anomalous":  True,
            "insert_anomalous": True,
            "anomaly_type":    "recurring_visit",
        }
        for cat in categories:
            stay_data[cat] = target_poi.get(cat, 0)
        new_stays_list.append(stay_data)

    if not new_stays_list:
        return agent_stays

    new_stays_df = pd.DataFrame(new_stays_list).sort_values("start")
    current = agent_stays
    for _, row in new_stays_df.iterrows():
        current = _insert_single_new_stay(current, row)
    return current


def insert_chained_visits_anomaly(agent_stays, pois, categories, aoi_box=None, aoi=None):
    """Insert two sequential visits to different novel POIs, the second following
    the first within 1–4h — simulating a two-stop clandestine trip pattern.

    Modelled on HAYSTAC schema types 2a (stay→trip→stay one-time) and 2b (same
    with specific travel modalities).  The distinctive signal is a pair of novel
    locations visited back-to-back in the same afternoon/evening, consistent with
    travelling to location A, meeting someone, then proceeding to location B.

    Args:
        agent_stays: DataFrame of stays for a single agent.
        pois: Full POI DataFrame.
        categories: List of category column names.
        aoi_box / aoi: Unused; kept for API consistency.

    Returns:
        DataFrame with two chained novel-visit stays inserted, labelled
        anomaly_type="chained_visit".
    """
    assert len(agent_stays) > 0, "agent_stays must not be empty"

    if "poi_id" in agent_stays.columns:
        habitual_pois = set(agent_stays["poi_id"].value_counts().head(10).index.tolist())
        candidate_pois = pois[~pois["poi_id"].isin(habitual_pois)]
        if len(candidate_pois) < 2:
            candidate_pois = pois
    else:
        candidate_pois = pois

    if len(candidate_pois) < 2:
        return agent_stays

    poi_a, poi_b = candidate_pois.sample(2).iloc[0], candidate_pois.sample(2).iloc[1]

    min_start = agent_stays["start"].min()
    max_stop  = agent_stays["stop"].max()
    min_day = int(min_start // 24)
    max_day = int(max_stop  // 24)

    new_stays_list = []
    for _ in range(np.random.randint(1, 3)):  # 1–2 chained-trip occurrences
        # First stop: duration 0.3–1.5h, between hours 10–18
        dur_a = np.random.uniform(0.3, 1.5)
        travel_buf = np.random.uniform(10 / 3600, 20 / 60)

        day = np.random.randint(min_day, max_day + 1)
        start_h_a = np.random.uniform(10, 18 - dur_a)
        new_start_a = day * 24 + start_h_a - travel_buf
        new_stop_a  = day * 24 + start_h_a + dur_a + travel_buf

        # Second stop: starts 1–4h after first stop ends
        gap_h = np.random.uniform(1.0, 4.0)
        dur_b = np.random.uniform(0.3, 1.5)
        new_start_b = new_stop_a + gap_h - travel_buf
        new_stop_b  = new_stop_a + gap_h + dur_b + travel_buf

        if (new_start_a <= min_start or new_stop_b >= max_stop):
            continue

        for poi, sa, so in [(poi_a, new_start_a, new_stop_a), (poi_b, new_start_b, new_stop_b)]:
            stay_data = {
                "agent":           agent_stays["agent"].iloc[0],
                "start":           sa,
                "stop":            so,
                "latitude":        poi["latitude"],
                "longitude":       poi["longitude"],
                "poi_id":          poi["poi_id"],
                "stay_anomalous":  True,
                "insert_anomalous": True,
                "anomaly_type":    "chained_visit",
            }
            for cat in categories:
                stay_data[cat] = poi.get(cat, 0)
            new_stays_list.append(stay_data)

    if not new_stays_list:
        return agent_stays

    new_stays_df = pd.DataFrame(new_stays_list).sort_values("start")
    current = agent_stays
    for _, row in new_stays_df.iterrows():
        current = _insert_single_new_stay(current, row)
    return current


def insert_covisit_anomaly(agent_stays, pois, categories, aoi_box=None, aoi=None):
    """Insert 1–3 short visits to unfamiliar POIs to simulate clandestine rendezvous.

    Modelled on the HAYSTAC anomaly pattern: small groups of agents meet at a
    shared location for roughly 0.2–2 hours during local business hours.  From a
    single-agent view the anomaly looks like an isolated visit to a POI the agent
    has never (or rarely) visited, at a time consistent with a private meeting.

    Differences from add_random_stays / add_recurring_stays:
      - Duration is short (0.2–2 h) to match observed rendezvous events (~1 h avg).
      - The POI is drawn from POIs NOT in the agent's top-10 most-visited locations,
        maximising the "unfamiliarity" signal.
      - Visits are constrained to plausible meeting times (local hours 8–22),
        assuming times in local hours (after time_modulo wrapping).
      - anomaly_type = "covisit" for a distinct training signal.

    Args:
        agent_stays: DataFrame of stays for a single agent (start/stop in hours).
        pois: Full POI DataFrame.
        categories: List of category column names.
        aoi_box: Optional bounding box (used as fallback POI source).
        aoi: Optional AOI polygon (used as fallback POI source).

    Returns:
        DataFrame with covisit stays inserted and labelled anomaly_type="covisit".
    """
    assert len(agent_stays) > 0, "agent_stays must not be empty"

    n_events = np.random.randint(1, 4)  # 1–3 events

    # Candidate POIs: exclude the agent's top-10 most-visited locations
    if "poi_id" in agent_stays.columns:
        habitual_pois = set(
            agent_stays["poi_id"].value_counts().head(10).index.tolist()
        )
        candidate_pois = pois[~pois["poi_id"].isin(habitual_pois)]
        if len(candidate_pois) == 0:
            candidate_pois = pois
    else:
        candidate_pois = pois

    min_start = agent_stays["start"].min()
    max_stop  = agent_stays["stop"].max()

    # Meeting time: constrained to hours 8–22 within any day in the sequence
    min_day = int(min_start // 24)
    max_day = int(max_stop  // 24)

    new_stays_list = []
    for _ in range(n_events):
        duration_h = np.random.uniform(0.2, 2.0)   # 12 min – 2 h
        travel_buf = np.random.uniform(10 / 3600, 20 / 60)  # 10 s – 20 min

        day = np.random.randint(min_day, max_day + 1)
        # Pick a meeting hour between 8 and 22 so the visit ends before midnight
        meeting_h = np.random.uniform(8, 22 - duration_h)
        new_start = day * 24 + meeting_h - travel_buf
        new_stop  = day * 24 + meeting_h + duration_h + travel_buf

        # Stay strictly within the agent's observed sequence window
        if new_start <= min_start or new_stop >= max_stop:
            continue

        poi = candidate_pois.sample(1).iloc[0]
        stay_data = {
            "agent":          agent_stays["agent"].iloc[0],
            "start":          new_start,
            "stop":           new_stop,
            "latitude":       poi["latitude"],
            "longitude":      poi["longitude"],
            "poi_id":         poi["poi_id"],
            "stay_anomalous": True,
            "insert_anomalous": True,
            "anomaly_type":   "covisit",
        }
        for cat in categories:
            stay_data[cat] = poi.get(cat, 0)
        new_stays_list.append(stay_data)

    if not new_stays_list:
        return agent_stays

    new_stays_df = pd.DataFrame(new_stays_list).sort_values("start")
    current = agent_stays
    for _, row in new_stays_df.iterrows():
        current = _insert_single_new_stay(current, row)
    return current


class Noisifier:
    """Generate synthetic anomalies in mobility data for training.

    The Noisifier applies various transformations to create anomalous patterns
    in agent stay sequences. With probability (1 - p_normal), it introduces
    anomalies by adding random stays, modifying times, or semantic behavioural
    changes (hunger / interest / social / work).

    Attributes:
        p_normal: Probability of returning unmodified (normal) data
        pois: DataFrame of Points of Interest
        categories: Category mapping for POIs
        aoi: Area of Interest polygon (optional)
        aoi_box: Bounding box of Area of Interest (optional)
        behavioral_anomaly: Whether behavioral anomaly regimes are enabled
    """

    def __init__(
        self,
        pois: pd.DataFrame,
        categories,
        p_normal: float,
        dataset_cfg: DatasetConfig,
        timestamp_step_size: float,
        sample_from: str = "aoi_box",
        insert_visits: bool = True,
        perturb_locations: bool = True,
        perturb_timestamps: bool = True,
        behavioral_anomaly: bool = False,
        legacy_perturb_timestamps: bool = False,
        replace_agent: bool = False,
        all_agent_stays: pd.DataFrame | None = None,
        insert_covisits: bool = False,
        insert_recurring_visits: bool = False,
        insert_chained_visits: bool = False,
        user_host_swap: bool = False,
        user_host_swap_frac: float = 0.3,
    ):
        """Initialize the Noisifier.

        Args:
            pois: DataFrame containing Points of Interest information
            categories: DataFrame mapping categories to POIs
            p_normal: Probability of returning unmodified data (0.0 to 1.0)
            sample_from: Source for random location sampling, either:
                - "aoi_box": Sample from bounding box of Area of Interest
                - "aoi": Sample from exact Area of Interest polygon
            insert_visits: Whether to insert synthetic anomalous visits via
                add_random_stays / add_recurring_stays.
            perturb_locations: Whether to apply location perturbations to
                flagged stays.
            perturb_timestamps: Whether to apply timestamp perturbations to
                flagged stays.
            behavioral_anomaly: Enable all four behavioral anomaly regimes
                (hunger, interest, social, work).  Anomaly type and intensity
                are sampled uniformly following the Urban Anomalies paper
                (arxiv 2410.01844): each of the 4 types × 3 intensity levels
                (yellow / orange / red) is equally likely.  Act-type code
                mappings are resolved automatically from the dataset name via
                _BEHAVIORAL_ACT_TYPES; if the dataset is not listed there the
                flag has no effect.
            replace_agent: Enable the replace-agent anomaly regime.  When
                selected, the agent's entire stay sequence is swapped with
                that of a randomly chosen different agent, and every stay is
                labelled anomalous.  Requires all_agent_stays to be provided;
                the flag has no effect otherwise.
            all_agent_stays: Full stays DataFrame for all agents in the
                dataset.  Required when replace_agent=True.
        """
        self.p_normal = p_normal
        self.insert_visits = insert_visits
        self.perturb_locations = perturb_locations
        self.perturb_timestamps = perturb_timestamps
        self.legacy_perturb_timestamps = legacy_perturb_timestamps
        self.pois = pois
        self.categories = categories
        self.timestamp_step_size = float(timestamp_step_size)

        self.aoi = self.aoi_box = None
        if sample_from == "aoi_box":
            self.aoi_box = load_aoi_box_padded(dataset_cfg)
        elif sample_from == "aoi":
            self.aoi = load_aoi(dataset_cfg)
        self._aoi = self.aoi_box if self.aoi_box is not None else self.aoi

        # Pre-compute spatial index over POIs for fast nearest-POI queries.
        print("Building POI spatial index for Noisifier...")
        poi_coords = pois[["latitude", "longitude"]].values
        self._poi_kdtree = KDTree(poi_coords, metric="euclidean")
        self._pois_reset = pois.reset_index(drop=True)
        print(f"POI spatial index built ({len(pois):,} POIs).")

        # Resolve behavioral anomaly act-type mappings from dataset name.
        act_type_map = _BEHAVIORAL_ACT_TYPES.get(dataset_cfg.name, {})
        self.behavioral_anomaly = behavioral_anomaly and bool(act_type_map)
        if behavioral_anomaly and not act_type_map:
            print(
                f"Warning: behavioral_anomaly=True but dataset '{dataset_cfg.name}' has no "
                "act-type mapping in _BEHAVIORAL_ACT_TYPES; behavioral anomaly disabled."
            )

        self._food_act_types: list[int] = act_type_map.get("food", [])
        self._work_act_types: list[int] = act_type_map.get("work", [])
        self._recreational_act_types: list[int] = act_type_map.get("recreational", [])

        self._food_pois = _filter_pois_by_act_types(pois, self._food_act_types)
        self._recreational_pois = _filter_pois_by_act_types(pois, self._recreational_act_types)
        if self._work_act_types:
            work_cols = [f"cat_{t}" for t in self._work_act_types]
            present = [c for c in work_cols if c in pois.columns]
            self._non_work_pois = pois[~(pois[present] > 0).any(axis=1)] if present else pois
        else:
            self._non_work_pois = pois.iloc[0:0]

        self.replace_agent = replace_agent
        self._all_agent_stays = all_agent_stays
        self.insert_covisits = insert_covisits
        self.insert_recurring_visits = insert_recurring_visits
        self.insert_chained_visits = insert_chained_visits
        self.user_host_swap = user_host_swap
        self.user_host_swap_frac = float(user_host_swap_frac)

    def nearest_poi(self, lat: float, lon: float) -> pd.Series:
        """Return the nearest POI row for a given (lat, lon) coordinate."""
        _, idx = self._poi_kdtree.query([[lat, lon]], k=1)
        return self._pois_reset.iloc[idx[0, 0]]

    def __call__(self, agent_stays: pd.DataFrame) -> pd.DataFrame:
        """Apply noise/anomalies to agent stay data.

        With probability p_normal, returns the original data unchanged.
        Otherwise, randomly selects an anomaly regime and applies it.

        Structural regimes (enabled when insert_visits / perturb_* flags are set):
        - "non_recurring" (40 %): isolated random insertions and/or per-stay
          location/time perturbations; anomaly_type="non_recurring".
        - "recurring"     (40 %): a regularly-spaced series of visits to the
          same new location; anomaly_type="recurring".
        - "mixed"         (20 %): both structural regimes applied together.

        Agent-replacement regime (enabled via replace_agent=True +
        all_agent_stays; ~10–12.5 % when combined with other active regimes):
        - "replace_agent": all stays swapped with a different agent's sequence;
          anomaly_type="replace_agent" on every stay.  Returns immediately
          (no further perturbation applied).

        Behavioral regimes (enabled via behavioral_anomaly=True; split 50/50
        with the structural pool):
        - "hunger"   : extra food POI visits; intensity sampled uniformly.
        - "interest" : recreational visits redirected to a different act_type.
        - "social"   : recreational visits redirected to random POIs.
        - "work"     : work visits replaced by non-work activities.

        Type and intensity are sampled uniformly from all 12 combinations
        (4 types × 3 intensities) following the Urban Anomalies paper
        (arxiv 2410.01844) combined-dataset distribution.

        Args:
            agent_stays: DataFrame containing stay records for a single agent

        Returns:
            DataFrame: Modified stay data with stay_anomalous and anomaly_type
            columns populated.
        """
        stays_copy = self.__class__.copy(agent_stays)

        if np.random.rand() < self.p_normal:
            return stays_copy

        structural_active = self.insert_visits or self.perturb_locations or self.perturb_timestamps
        replace_agent_active = self.replace_agent and self._all_agent_stays is not None
        covisit_active = self.insert_covisits
        recurring_visit_active = self.insert_recurring_visits
        chained_visit_active = self.insert_chained_visits
        user_host_swap_active = self.user_host_swap and self._all_agent_stays is not None

        # Build regime pool and weights dynamically so each active group gets
        # equal share: structural ~40 %, covisit ~10 %, behavioral ~40 %,
        # replace_agent ~10 % (when all are on).
        structural_regimes = ["non_recurring", "recurring", "mixed"]
        structural_weights = [0.4, 0.4, 0.2]
        behavioral_regimes = list(_BEHAVIORAL_TYPES)

        pool: list[str] = []
        weights: list[float] = []

        if structural_active:
            pool.extend(structural_regimes)
            weights.extend(structural_weights)
        if covisit_active:
            pool.append("covisit")
            weights.append(0.25)  # relative weight; will be normalised below
        if recurring_visit_active:
            pool.append("recurring_visit")
            weights.append(0.25)
        if chained_visit_active:
            pool.append("chained_visit")
            weights.append(0.25)
        if self.behavioral_anomaly:
            pool.extend(behavioral_regimes)
            weights.extend([1.0 / len(behavioral_regimes)] * len(behavioral_regimes))
        if replace_agent_active:
            pool.append("replace_agent")
            weights.append(0.125)
        if user_host_swap_active:
            pool.append("user_host_swap")
            weights.append(1.0)  # if enabled, it dominates the pool (LANL fine-tune use case)

        if not pool:
            return stays_copy

        total = sum(weights)
        probs = [w / total for w in weights]
        regime = np.random.choice(pool, p=probs)

        # Sample behavioral intensity uniformly (used only for behavioral regimes)
        intensity = np.random.choice(_INTENSITY_LEVELS)

        # --- Apply chosen regime ---
        if regime == "replace_agent":
            return apply_replace_agent(stays_copy, self._all_agent_stays)

        if regime == "user_host_swap":
            return apply_user_host_swap(
                stays_copy, self._all_agent_stays, swap_frac=self.user_host_swap_frac
            )

        is_semantic = regime in ("hunger", "interest", "social", "work", "covisit",
                                  "recurring_visit", "chained_visit")

        if regime in ("non_recurring", "mixed"):
            if self.insert_visits:
                stays_copy = add_random_stays(
                    stays_copy, self.pois, self.categories, self.nearest_poi, self.aoi_box, self.aoi
                )
            if self.perturb_locations or self.perturb_timestamps:
                all_idx = stays_copy.index
                selected_idx = all_idx[np.random.rand(len(all_idx)) < 0.3]
                if len(selected_idx) > 0:
                    stays_copy.loc[selected_idx, "stay_anomalous"] = True
                    untyped = selected_idx[
                        stays_copy.loc[selected_idx, "anomaly_type"].isna()
                    ]
                    stays_copy.loc[untyped, "anomaly_type"] = "non_recurring"
                    if self.perturb_locations and self.perturb_timestamps:
                        choices = np.random.randint(0, 3, len(selected_idx))
                        stays_copy.loc[selected_idx[choices != 1], "time_anomalous"] = True
                        stays_copy.loc[selected_idx[choices != 0], "loc_anomalous"] = True
                    elif self.perturb_locations:
                        stays_copy.loc[selected_idx, "loc_anomalous"] = True
                    else:
                        stays_copy.loc[selected_idx, "time_anomalous"] = True

        if regime in ("recurring", "mixed"):
            if self.insert_visits:
                stays_copy = add_recurring_stays(
                    stays_copy, self.pois, self.categories, self.nearest_poi, self.aoi_box, self.aoi
                )

        if regime == "hunger":
            stays_copy = apply_hunger_anomaly(
                stays_copy, self._food_pois, self.categories, intensity
            )

        if regime == "interest":
            stays_copy = apply_interest_anomaly(
                stays_copy, self.pois, self.categories,
                self._recreational_act_types, intensity
            )

        if regime == "social":
            stays_copy = apply_social_anomaly(
                stays_copy, self._recreational_pois, self.categories,
                self._recreational_act_types, intensity
            )

        if regime == "work":
            stays_copy = apply_work_anomaly(
                stays_copy, self._non_work_pois, self.categories,
                self._work_act_types, intensity
            )

        if regime == "covisit":
            stays_copy = insert_covisit_anomaly(
                stays_copy, self.pois, self.categories, self.aoi_box, self.aoi
            )

        if regime == "recurring_visit":
            stays_copy = insert_recurring_visit_anomaly(
                stays_copy, self.pois, self.categories, self.aoi_box, self.aoi
            )

        if regime == "chained_visit":
            stays_copy = insert_chained_visits_anomaly(
                stays_copy, self.pois, self.categories, self.aoi_box, self.aoi
            )

        # Guarantee at least one stay_anomalous=True for structural regimes when
        # no insertion or perturbation has fired yet.
        if not is_semantic and not stays_copy["stay_anomalous"].any() and (
            self.perturb_locations or self.perturb_timestamps
        ):
            force_idx = np.random.choice(stays_copy.index)
            stays_copy.loc[force_idx, "stay_anomalous"] = True
            stays_copy.loc[force_idx, "anomaly_type"] = "non_recurring"
            if self.perturb_locations and self.perturb_timestamps:
                choice = np.random.randint(0, 3)
                if choice != 1:
                    stays_copy.loc[force_idx, "time_anomalous"] = True
                if choice != 0:
                    stays_copy.loc[force_idx, "loc_anomalous"] = True
            elif self.perturb_locations:
                stays_copy.loc[force_idx, "loc_anomalous"] = True
            else:
                stays_copy.loc[force_idx, "time_anomalous"] = True

        # Apply generic loc/time perturbations only for structural regimes.
        # Semantic anomalies set loc_anomalous themselves with specific POI choices
        # that must not be overwritten by the random perturbation step.
        if not is_semantic:
            loc_mask = stays_copy["loc_anomalous"].fillna(False).astype(bool)
            time_mask = stays_copy["time_anomalous"].fillna(False).astype(bool)

            if self.perturb_locations and loc_mask.any():
                stays_copy = perturb_locations(stays_copy, loc_mask, self.categories, self._aoi, self.nearest_poi)
            if self.perturb_timestamps and time_mask.any():
                _pt_fn = perturb_timestamps_legacy if self.legacy_perturb_timestamps else perturb_timestamps
                stays_copy = _pt_fn(
                    stays_copy,
                    time_mask,
                    timestamp_step_size=self.timestamp_step_size,
                )

        return stays_copy

    @staticmethod
    def copy(agent_stays):
        copy = agent_stays.copy()
        copy["stay_anomalous"] = False
        copy["insert_anomalous"] = False
        copy["loc_anomalous"] = False
        copy["time_anomalous"] = False
        copy["anomaly_type"] = None
        return copy


def perturb_locations(stay_df, anomalous_mask, categories, aoi, nearest_poi_fn):
    n_anomalous = anomalous_mask.sum()
    new_coords = sample_from_aoi_box(aoi, size=n_anomalous)  # (N, 2): lon, lat
    stay_df.loc[anomalous_mask, ["longitude", "latitude"]] = new_coords

    # Attribute each perturbed stay to its nearest POI and use that POI's categories.
    anom_idx = stay_df.index[anomalous_mask]
    for i, idx in enumerate(anom_idx):
        lon, lat = new_coords[i]
        nearest = nearest_poi_fn(lat, lon)
        stay_df.at[idx, "poi_id"] = nearest["poi_id"]
        for cat in categories:
            stay_df.at[idx, cat] = nearest.get(cat, 0)

    return stay_df


def perturb_timestamps_legacy(stay_df, anomalous_mask, timestamp_step_size):
    """Legacy (pre-5549b53) timestamp perturbation.

    Randomly resamples start and stop for each anomalous stay independently,
    without preserving travel-gap continuity and without adjusting or flagging
    adjacent stays.  Kept for ablation comparison against the current
    travel-gap-preserving version.
    """
    n_anomalous = anomalous_mask.sum()
    step_size = float(timestamp_step_size)

    min_start_steps = int(stay_df["start"].min() / step_size)
    max_start_steps = int(stay_df["start"].max() / step_size)
    starts_steps = np.random.randint(min_start_steps, max_start_steps + 1, n_anomalous)
    starts = starts_steps * step_size
    stay_df.loc[anomalous_mask, "start"] = starts

    min_duration_hours = 5 / 60
    max_duration_hours = 6
    min_duration_steps = int(min_duration_hours / step_size)
    max_duration_steps = int(max_duration_hours / step_size)
    durations_steps = np.random.randint(min_duration_steps, max_duration_steps + 1, n_anomalous)
    durations = durations_steps * step_size
    stay_df.loc[anomalous_mask, "stop"] = starts + durations

    return stay_df


def perturb_timestamps(stay_df, anomalous_mask, timestamp_step_size):
    """Perturb timestamps of anomalous stays while preserving travel-gap continuity.

    For each perturbed stay B, the immediate predecessor A and successor C are
    adjusted so that the A→B and B→C travel times are unchanged:
        new_A.stop  = new_B.start - gap_before
        new_C.start = new_B.stop  + gap_after

    A and C are marked stay_anomalous / time_anomalous because their durations
    change. New start and duration for B are sampled uniformly (in step_size
    units) within the feasible range that leaves at least 5 minutes of duration
    for both A and C. Stays with no feasible perturbation are left unchanged.

    Stays are processed in sequence order so that cascading adjustments (when
    two anomalous stays are adjacent) accumulate naturally.
    """
    step_size = float(timestamp_step_size)
    min_dur = 5 / 60   # minimum stay duration in hours
    max_dur = 6.0      # maximum stay duration in hours
    min_dur_steps = max(1, int(round(min_dur / step_size)))
    max_dur_steps = int(round(max_dur / step_size))

    all_idx = stay_df.index.tolist()
    idx_to_pos = {idx: p for p, idx in enumerate(all_idx)}

    for idx in stay_df.index[anomalous_mask]:
        pos = idx_to_pos[idx]

        orig_start = float(stay_df.at[idx, "start"])
        orig_stop  = float(stay_df.at[idx, "stop"])

        prev_idx = all_idx[pos - 1] if pos > 0 else None
        next_idx = all_idx[pos + 1] if pos < len(all_idx) - 1 else None

        # Travel gaps to immediate neighbors (preserved after perturbation)
        gap_before = (orig_start - float(stay_df.at[prev_idx, "stop"])) if prev_idx is not None else None
        gap_after  = (float(stay_df.at[next_idx, "start"]) - orig_stop)  if next_idx is not None else None

        # Lower bound on new_B.start: leave at least min_dur for prev stay A
        if prev_idx is not None:
            lo_start = float(stay_df.at[prev_idx, "start"]) + min_dur + gap_before
        else:
            lo_start = float(stay_df["start"].min())

        # Upper bound on new_B.stop: leave at least min_dur for next stay C
        if next_idx is not None:
            hi_stop = float(stay_df.at[next_idx, "stop"]) - min_dur - gap_after
        else:
            hi_stop = float(stay_df["stop"].max())

        lo_start_steps = int(np.ceil(lo_start / step_size))
        hi_stop_steps  = int(np.floor(hi_stop  / step_size))
        hi_start_steps = hi_stop_steps - min_dur_steps  # B itself needs at least min_dur

        if lo_start_steps > hi_start_steps:
            continue  # no feasible perturbation; skip this stay

        new_start_steps = np.random.randint(lo_start_steps, hi_start_steps + 1)

        max_this_dur_steps = min(max_dur_steps, hi_stop_steps - new_start_steps)
        if min_dur_steps > max_this_dur_steps:
            continue

        new_dur_steps = np.random.randint(min_dur_steps, max_this_dur_steps + 1)
        new_start = new_start_steps * step_size
        new_stop  = (new_start_steps + new_dur_steps) * step_size

        stay_df.at[idx, "start"] = new_start
        stay_df.at[idx, "stop"]  = new_stop

        anomaly_type = stay_df.at[idx, "anomaly_type"]

        if prev_idx is not None:
            stay_df.at[prev_idx, "stop"] = new_start - gap_before
            stay_df.at[prev_idx, "stay_anomalous"] = True
            stay_df.at[prev_idx, "time_anomalous"] = True
            if pd.isna(stay_df.at[prev_idx, "anomaly_type"]):
                stay_df.at[prev_idx, "anomaly_type"] = anomaly_type

        if next_idx is not None:
            stay_df.at[next_idx, "start"] = new_stop + gap_after
            stay_df.at[next_idx, "stay_anomalous"] = True
            stay_df.at[next_idx, "time_anomalous"] = True
            if pd.isna(stay_df.at[next_idx, "anomaly_type"]):
                stay_df.at[next_idx, "anomaly_type"] = anomaly_type

    return stay_df


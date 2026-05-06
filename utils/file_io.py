import geopandas as gpd
import pandas as pd

from config import DatasetConfig


def load_stays(split, dataset_cfg: DatasetConfig):
    if split == "train":
        return pd.read_parquet(dataset_cfg.train_data_path)
    elif split == "test":
        return pd.read_parquet(dataset_cfg.test_data_path)
    else:
        raise ValueError(f"Unknown split: {split}")


def load_pois(dataset_cfg: DatasetConfig, category_col="act_types"):
    # Read the data
    pois = pd.read_parquet(dataset_cfg.poi_path)

    # Explode the category lists into separate rows
    exploded = pois.explode(category_col)
    exploded[category_col] = exploded[category_col].apply(lambda x: f"cat_{int(x)}")

    # Create dummy variables for the categories
    dummies = pd.get_dummies(exploded[category_col])

    # Group back by the original index and sum up the dummies
    category_counts = dummies.groupby(exploded.index).sum()

    # Merge the category counts back into the original dataframe
    pois = pois.join(category_counts)

    categories = exploded[category_col].unique().tolist()

    print(f"Loaded {len(pois):,} POIs with categories: {categories}")

    return pois, categories

def load_aoi(dataset_cfg: DatasetConfig):
    aoi = gpd.read_file(dataset_cfg.aoi_path).geometry.values[0]
    return aoi


def load_aoi_box_padded(dataset_cfg: DatasetConfig, buffer=0.01):
    aoi = load_aoi(dataset_cfg)
    return aoi.buffer(buffer).envelope


def load_min_time(dataset_cfg: DatasetConfig):
    min_time = pd.read_parquet(dataset_cfg.dataset_metadata_path)["min_time"][0]
    return min_time


def load_dataset_metadata(dataset_cfg: DatasetConfig):
    metadata_df = pd.read_parquet(dataset_cfg.dataset_metadata_path)
    if metadata_df.empty:
        raise ValueError(f"Dataset metadata file is empty: {dataset_cfg.dataset_metadata_path}")

    row = metadata_df.iloc[0]
    return {
        "min_time": row["min_time"],
        "local_tz": row.get("local_tz", dataset_cfg.default_local_tz),
        "timestamp_step_size": float(
            row.get("timestamp_step_size", dataset_cfg.default_timestamp_step_size)
        ),
        "utm_crs": row.get("utm_crs", dataset_cfg.default_utm_crs),
        "n_agents": int(row.get("n_agents", dataset_cfg.default_n_agents)),
    }

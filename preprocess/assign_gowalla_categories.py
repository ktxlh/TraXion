"""Assign OpenStreetMap categories to the POIs of a prepared Gowalla city subset.

Gowalla check-ins come without venue categories. Each POI gets the tag of the
nearest OpenStreetMap feature within 50 m, among features carrying one of the
keys amenity, shop, leisure, tourism, office (checked in that order), as
"<key>:<value>" (e.g. "amenity:restaurant"). POIs with no such feature within
50 m are labeled "unknown". Feature locations are the node itself, or a
representative point inside the way / relation geometry; distances are
Euclidean in the local UTM projection.

OSM features are downloaded from the Overpass API as of --osm-date (the snapshot
used for the paper), in 0.1-degree tiles cached under <dataset dir>/osm_cache/.

Category ids are assigned by POI count (most frequent first) with "unknown"
last. The script rewrites, in place, <dataset>_poi.parquet (venue_type,
act_types) and <dataset>_{train,test}.parquet (act_types, cat_0).

Run after prepare_city_subsets.py::

    python preprocess/assign_gowalla_categories.py --dataset gowalla-stockholm-v1
    python preprocess/assign_gowalla_categories.py --dataset gowalla-austin-v1

Map data (c) OpenStreetMap contributors, available under the Open Database License.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

DATA_ROOT = os.environ.get("TRAXION_DATA_DIR", "./data")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_KEYS = ["amenity", "shop", "leisure", "tourism", "office"]
RADIUS_M = 50.0
TILE_DEG = 0.1


def _query(bbox: str, osm_date: str) -> bytes:
    query = (f'[out:json][timeout:180][date:"{osm_date}"];('
             + "".join(f'nwr["{key}"]({bbox});' for key in OSM_KEYS)
             + ");out geom tags;")
    req = urllib.request.Request(OVERPASS_URL, data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": "traxion-preprocess/1.0"})
    raw = urllib.request.urlopen(req, timeout=300).read()
    remark = json.loads(raw).get("remark", "")
    if "runtime error" in remark:
        raise RuntimeError(remark)
    return raw


def _fetch_tile(la: float, lo: float, size: float, path: str, osm_date: str, depth: int = 0) -> list[str]:
    """Fetch one tile into `path`; on repeated timeouts split it into four quarter tiles."""
    if os.path.exists(path):
        return [path]
    split_marker = path + ".split"
    if os.path.exists(split_marker):
        with open(split_marker) as f:
            return [q for line in f for q in [line.strip()] if q]
    for attempt in range(2 if depth < 3 else 8):
        try:
            raw = _query(f"{la},{lo},{la + size},{lo + size}", osm_date)
            with open(path, "wb") as f:
                f.write(raw)
            return [path]
        except Exception as ex:  # rate limits / timeouts: back off and retry
            print(f"  {os.path.basename(path)}: retry {attempt} ({str(ex)[:80]})")
            time.sleep(15 * (attempt + 1))
    if depth >= 3:
        raise RuntimeError(f"failed to download {path}")
    half = size / 2
    parts = []
    for i in (0, 1):
        for j in (0, 1):
            parts += _fetch_tile(la + i * half, lo + j * half, half,
                                 path.replace(".json", f"_{i}{j}.json"), osm_date, depth + 1)
    with open(split_marker, "w") as f:
        f.write("\n".join(parts))
    return parts


def fetch_tiles(pois: pd.DataFrame, cache_dir: str, osm_date: str) -> list[dict]:
    """Download (or load from cache) every OSM element near the POIs."""
    os.makedirs(cache_dir, exist_ok=True)
    s, w = pois["latitude"].min(), pois["longitude"].min()
    # Every tile that intersects some POI's search radius (0.001 deg >= 50 m in both axes
    # at the latitudes used here).
    tiles = set()
    for dlat in (-0.001, 0.0, 0.001):
        for dlon in (-0.001, 0.0, 0.001):
            ai = np.floor((pois["latitude"] + dlat - s) / TILE_DEG).astype(int)
            bi = np.floor((pois["longitude"] + dlon - w) / TILE_DEG).astype(int)
            tiles |= set(zip(ai, bi))
    elements = {}
    for k, (a, b) in enumerate(sorted(tiles)):
        paths = _fetch_tile(s + a * TILE_DEG, w + b * TILE_DEG, TILE_DEG,
                            os.path.join(cache_dir, f"{a:03d}_{b:03d}.json"), osm_date)
        for path in paths:
            with open(path) as f:
                for el in json.load(f)["elements"]:
                    elements[(el["type"], el["id"])] = el
        print(f"  tiles {k + 1}/{len(tiles)}  elements={len(elements):,}", end="\r")
    print()
    return list(elements.values())


def _geometry(el: dict):
    if el["type"] == "node":
        return Point(el["lon"], el["lat"])
    if el["type"] == "way":
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) < 2:
            return None
        return Polygon(coords) if len(coords) >= 4 and coords[0] == coords[-1] else LineString(coords)
    lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
             for m in el.get("members", []) if len(m.get("geometry", []) or []) >= 2]
    polys = list(polygonize(lines))
    if polys:
        return unary_union(polys)
    return unary_union(lines) if lines else None


def _label(tags: dict) -> str | None:
    for key in OSM_KEYS:
        if key in tags:
            return f"{key}:{tags[key]}"
    return None


def assign(pois: pd.DataFrame, elements: list[dict]) -> np.ndarray:
    rows = [(_label(el.get("tags", {})), _geometry(el)) for el in elements]
    rows = [(lab, g) for lab, g in rows if lab is not None and g is not None and not g.is_empty]
    feats = gpd.GeoDataFrame({"label": [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=4326)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pois["longitude"], pois["latitude"]), crs=4326)
    utm = pts.estimate_utm_crs()
    feat_pts = feats.to_crs(utm).geometry.representative_point()
    poi_pts = pts.to_crs(utm).geometry
    tree = cKDTree(np.c_[feat_pts.x, feat_pts.y])
    dist, idx = tree.query(np.c_[poi_pts.x, poi_pts.y], k=1, distance_upper_bound=RADIUS_M)
    labels = feats["label"].to_numpy()
    return np.array([labels[j] if np.isfinite(d) else "unknown" for d, j in zip(dist, idx)], dtype=object)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="e.g. gowalla-austin-v1")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--osm-date", default="2026-04-21T21:00:00Z",
                    help="OSM snapshot date passed to Overpass")
    args = ap.parse_args()

    d = os.path.join(args.data_root, args.dataset)
    poi_path = os.path.join(d, f"{args.dataset}_poi.parquet")
    pois = pd.read_parquet(poi_path)
    print(f"{args.dataset}: {len(pois):,} POIs")

    elements = fetch_tiles(pois, os.path.join(d, "osm_cache"), args.osm_date)
    venue_type = assign(pois, elements)

    counts = pd.Series(venue_type[venue_type != "unknown"]).value_counts()
    order = sorted(counts.index, key=lambda c: (-counts[c], c)) + ["unknown"]
    cat_id = {c: i for i, c in enumerate(order)}
    pois["venue_type"] = venue_type
    pois["act_types"] = [[cat_id[v]] for v in venue_type]
    pois.to_parquet(poi_path, index=False)
    print(f"  {len(order)} categories, unknown = {np.mean(venue_type == 'unknown'):.1%} of POIs")

    poi_to_id = dict(zip(pois["poi_id"], (cat_id[v] for v in venue_type)))
    for split in ("train", "test"):
        path = os.path.join(d, f"{args.dataset}_{split}.parquet")
        df = pd.read_parquet(path)
        ids = df["poi_id"].map(poi_to_id).astype(np.int64)
        df["act_types"] = [[i] for i in ids]
        df["cat_0"] = ids.astype(np.float64)
        df.to_parquet(path, index=False)
        print(f"  rewrote {path}")


if __name__ == "__main__":
    main()

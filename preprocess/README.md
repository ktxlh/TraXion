# Data preparation scripts

Each script reads a public-dataset raw release and writes parquet files in the
canonical layout that `config.py` expects:

```
$TRAXION_DATA_DIR/<dataset>/<dataset>_train.parquet
$TRAXION_DATA_DIR/<dataset>/<dataset>_test.parquet
$TRAXION_DATA_DIR/<dataset>/<dataset>_poi.parquet
$TRAXION_DATA_DIR/<dataset>/<dataset>_aoi_box.geojson
$TRAXION_DATA_DIR/<dataset>/<dataset>_dataset_metadata.parquet
```

## NUMOSIM-LA / Urban Anomalies

Place the OSF-released parquet files for each dataset at the canonical layout
above. The OSF release already ships in this schema; no preparation script is
required.

## Foursquare-Tokyo, Gowalla-Stockholm-v1, Gowalla-Austin-v1

```bash
FOURSQUARE_RAW_DIR=/path/to/dataset_WWW2019 \
GOWALLA_RAW_DIR=/path/to/loc-gowalla \
  python -m preprocess.prepare_city_subsets
```

## LANL Auth. Log

```bash
LANL_RAW_DIR=/path/to/lanl_cyber1/raw \
  python -m preprocess.prepare_lanl
```

## eICU-CRD demo

```bash
EICU_RAW_DIR=/path/to/eicu-collaborative-research-database-demo-2.0.1 \
  python -m preprocess.prepare_eicu
```

## Social-link splits (one per LBSN city)

```bash
python -m preprocess.prepare_social_splits --dataset foursquare-tokyo
python -m preprocess.prepare_social_splits --dataset gowalla-stockholm-v1
python -m preprocess.prepare_social_splits --dataset gowalla-austin-v1
```

Splits land at `$TRAXION_DATA_DIR/social_splits/<dataset>/`.

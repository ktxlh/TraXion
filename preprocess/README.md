# Data preparation scripts

Each script reads a dataset's original release and writes parquet files in the
layout that `config.py` expects:

```
$TRAXION_DATA_DIR/<dataset>/<dataset>_train.parquet
$TRAXION_DATA_DIR/<dataset>/<dataset>_test.parquet
$TRAXION_DATA_DIR/<dataset>/<dataset>_poi.parquet
$TRAXION_DATA_DIR/<dataset>/<dataset>_aoi_box.geojson
$TRAXION_DATA_DIR/<dataset>/<dataset>_dataset_metadata.parquet
```

| Script | Datasets |
|---|---|
| `prepare_numosim.py` | NUMOSIM-LA (`numosim`) |
| `preprocess.py` | Urban Anomalies-Berlin / -Atlanta (`urban-berlin`, `urban-atlanta`) |
| `prepare_city_subsets.py` | Foursquare-Tokyo, Gowalla-Stockholm, Gowalla-Austin |
| `assign_gowalla_categories.py` | OpenStreetMap POI categories for the two Gowalla subsets |
| `prepare_social_splits.py` | social-link edge splits for the three city subsets |
| `prepare_lanl.py` | LANL authentication log (`lanl-core`); host coordinates in `assets/` |
| `prepare_eicu.py` | eICU-CRD 2.0 (`eicu-full`) |
| `prepare_mimiciv.py` | MIMIC-IV 3.1 (`mimiciv-full`) |
| `rekey_patient.py` | patient-keyed pre-training copies (`eicu-full-pid`, `mimiciv-full-pid`) |

The commands, in order, are in §2 of the top-level [README](../README.md).

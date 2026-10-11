"""Re-key a prepared EHR corpus so that `agent` is the patient instead of the ICU stay.

TraXion pre-trains the EHR corpora with the patient as the entity (`uniquepid`,
which holds eICU-CRD's uniquepid and MIMIC-IV's subject_id). Mortality fine-tuning
and evaluation keep using the stay-keyed corpus written by prepare_eicu.py /
prepare_mimiciv.py, because the label is defined per stay.

This script copies <src>_{train,test}.parquet to <dst>_{train,test}.parquet with
only the `agent` column replaced (row order and all other columns unchanged),
copies the POI, category and AOI files, and rewrites n_agents in the metadata.
No stay-label file is written for the patient-keyed corpus.

Usage::

    python preprocess/rekey_patient.py --src eicu-full --dst eicu-full-pid
    python preprocess/rekey_patient.py --src mimiciv-full --dst mimiciv-full-pid
"""
import argparse
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = os.environ.get("TRAXION_DATA_DIR", "./data")


def rekey(src_name: str, dst_name: str, id_col: str = "uniquepid", data_root: str = DATA_ROOT):
    src_dir = os.path.join(data_root, src_name)
    dst_dir = os.path.join(data_root, dst_name)
    os.makedirs(dst_dir, exist_ok=True)

    labels = pd.read_parquet(os.path.join(src_dir, f"{src_name}_stay_labels.parquet"),
                             columns=["patientunitstayid", id_col, "split"])

    # The split is patient-level, so no patient may appear in more than one split.
    leak = (labels.groupby(id_col)["split"].nunique() > 1).sum()
    if leak:
        raise RuntimeError(f"{src_name}: {leak} patients span more than one split")

    # Dense agent id: sorted unique patient id -> 0..N-1.
    uniq_patients = pd.Index(sorted(labels[id_col].unique()))
    patient_to_new_agent = pd.Series(range(len(uniq_patients)), index=uniq_patients, dtype="int64")
    n_new_agents = len(uniq_patients)
    stay_to_patient = dict(zip(labels["patientunitstayid"], labels[id_col]))

    seen = set()
    for split_file in ("train", "test"):
        src_path = os.path.join(src_dir, f"{src_name}_{split_file}.parquet")
        dst_path = os.path.join(dst_dir, f"{dst_name}_{split_file}.parquet")
        df = pq.read_table(src_path).to_pandas()
        patient = df["patientunitstayid"].map(stay_to_patient)
        if patient.isna().any():
            raise RuntimeError(f"{src_name}/{split_file}: stays missing from the stay-label table")
        new_agent = patient.map(patient_to_new_agent).astype("int64")
        df["agent"] = new_agent.values
        agents = set(new_agent.unique().tolist())
        if agents & seen:
            raise RuntimeError(f"{src_name}/{split_file}: patients shared across train and test")
        seen |= agents
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), dst_path)
        print(f"  wrote {dst_path}  rows={len(df):,}  agents={len(agents):,}")

    for suffix in ("poi.parquet", "categories.csv", "aoi_box.geojson"):
        src_f = os.path.join(src_dir, f"{src_name}_{suffix}")
        if not os.path.exists(src_f):
            raise RuntimeError(f"missing {src_f}")
        shutil.copyfile(src_f, os.path.join(dst_dir, f"{dst_name}_{suffix}"))

    meta = pd.read_parquet(os.path.join(src_dir, f"{src_name}_dataset_metadata.parquet")).copy()
    meta.loc[0, "n_agents"] = int(n_new_agents)
    meta.to_parquet(os.path.join(dst_dir, f"{dst_name}_dataset_metadata.parquet"), index=False)
    print(f"  {src_name} -> {dst_name}: n_agents = {n_new_agents:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="stay-keyed dataset name, e.g. eicu-full")
    ap.add_argument("--dst", required=True, help="output dataset name, e.g. eicu-full-pid")
    ap.add_argument("--data-root", default=DATA_ROOT)
    args = ap.parse_args()
    rekey(args.src, args.dst, data_root=args.data_root)

# TraXion: Co-occurrence-aware Pre-training for Multi-Entity Sparse Event Streams

This repository is the official implementation of TraXion. It is the NeurIPS
supplementary code asset associated with the submission and is licensed under
the MIT License (see [`LICENSE`](LICENSE)).

The codebase contains everything needed to reproduce every TraXion number reported
in the paper: pre-training, per-task fine-tuning, evaluation, and the ablation
variants. Baseline implementations live in their respective upstream repositories
and are documented in the paper's appendix; only TraXion numbers are produced
here.

```
.
├── config.py                # dataset path registry (env-var driven)
├── cli/                     # entry-point scripts (run as python -m cli.<name>)
│   ├── train.py             #   pre-training + anomaly fine-tune
│   ├── train_poi.py         #   next-POI fine-tune
│   ├── train_next_visit.py  #   next-visit (joint POI + time) fine-tune
│   ├── train_social.py      #   social-link fine-tune
│   ├── train_eicu.py        #   ICU-mortality fine-tune
│   ├── train_ablation_pretrain.py   # variants C4 / C5 pre-training
│   ├── evaluate*.py         #   per-task evaluators
│   └── score_lanl_unsup.py  #   LANL rank-fusion scorer
├── modules/                 # backbone (factorized attention, prototype, embeddings)
├── tasks/                   # per-task heads, datasets, criteria, metrics
├── utils/                   # argparse, dataset, noisifier, training runtime, ...
├── preprocess/              # one prepare_*.py per public dataset
└── scripts/                 # bash wrappers with paper-default hyperparameters
```

All `cli/*.py` scripts are entry points. Run them from the project root using
`python -m cli.<name>` (e.g. `python -m cli.train ...`); this puts the project
root on `sys.path` so that absolute imports such as `from config import ...`
resolve correctly.


## 1. Environment setup

```bash
conda env create -f environment.yml
conda activate traxion
```

A pip-only fallback (`pip install -r requirements.txt`) works against any
recent CUDA-12 PyTorch wheel.

GPU: every run reported in the paper completes on a single NVIDIA RTX 6000 Ada
(48 GB) within five days end-to-end (pre-train + per-task fine-tune).


## 2. Data preparation

TraXion uses eight public datasets. Download each from its official source under
its license, then run the matching preparation script. By default scripts read
`$TRAXION_DATA_DIR` (default `./data`) and write back to the same root in the
canonical layout `<TRAXION_DATA_DIR>/<dataset>/<dataset>_{train,test,poi,
dataset_metadata}.parquet`. Override with `export TRAXION_DATA_DIR=/path/to/data`.

| Dataset (paper)        | `--dataset` flag        | Source / license |
|------------------------|-------------------------|------------------|
| NUMOSIM-LA             | `numosim`               | OSF `osf.io/sjyfr` (public release; no formal license) |
| Urban Anomalies-Berlin | `urban-berlin`          | OSF `osf.io/dg6t3` (CC-BY 4.0) |
| Urban Anomalies-Atlanta| `urban-atlanta`         | OSF `osf.io/dg6t3` (CC-BY 4.0) |
| Foursquare-Tokyo       | `foursquare-tokyo`      | Yang's homepage / WWW2019 (citation required) |
| Gowalla-Stockholm      | `gowalla-stockholm-v1`  | SNAP / Stanford (citation required) |
| Gowalla-Austin         | `gowalla-austin-v1`     | SNAP / Stanford (citation required) |
| LANL Auth. Log         | `lanl-core`             | `csr.lanl.gov/data/cyber1` (CC0 / public domain) |
| eICU-CRD demo          | `eicu-demo`             | PhysioNet (Open Database License, ODbL) |

Preparation commands:

```bash
# NUMOSIM-LA: place the OSF-released parquet files at
#   $TRAXION_DATA_DIR/numosim/numosim_{train,test,poi,dataset_metadata}.parquet
#   plus numosim_aoi_box.geojson. No script is needed — the raw release is
#   already in the expected layout.

# Urban Anomalies (Berlin / Atlanta): place the released staypoints TSVs at
#   $TRAXION_DATA_DIR/urban-{berlin,atlanta}/staypoints_{train,test}.tsv
#   plus the city's POI parquet and dataset_metadata. The OSF release ships
#   each city already in the expected schema.

# Foursquare-Tokyo and Gowalla-{Stockholm,Austin}-v1
#   put the WWW2019 Foursquare release under $FOURSQUARE_RAW_DIR and the
#   Gowalla SNAP release under $GOWALLA_RAW_DIR, then run:
python -m preprocess.prepare_city_subsets

# LANL — point at the gzipped auth.txt.gz / redteam.txt.gz from cyber1:
LANL_RAW_DIR=/path/to/lanl/raw \
  python -m preprocess.prepare_lanl

# eICU-CRD demo — point at the downloaded CSV-bundle directory:
EICU_RAW_DIR=/path/to/eicu-collaborative-research-database-demo-2.0.1 \
  python -m preprocess.prepare_eicu

# Social-link splits (one per LBSN city)
python -m preprocess.prepare_social_splits --dataset foursquare-tokyo
python -m preprocess.prepare_social_splits --dataset gowalla-stockholm-v1
python -m preprocess.prepare_social_splits --dataset gowalla-austin-v1
```


## 3. Reproducing the paper

Each row in every results table comes from **(i) one pre-training run per
dataset, then (ii) one fine-tuning run per (task × model variant × seed)**.
Every command below runs end-to-end on a single GPU; pass `--seed 0`, `--seed
101`, `--seed 202` for the three-seed protocol (single seed for NUMOSIM-LA per
the paper).

`$WANDB_ENTITY` (optional) sets the W&B entity; logs go offline if W&B is
disabled with `--no_wandb`. All checkpoints land in `$TRAXION_RUNS_DIR`
(default `./runs`).

The ready-to-copy reproduction commands for every cell are in
[`scripts/`](scripts/). A summary of the calling convention follows; see the
paper's Appendix B for the full hyperparameter tables.

### 3.1 Pre-training (one per dataset)

```bash
bash scripts/pretrain.sh <dataset> [--seed 0] [--device cuda:0]
```

Equivalently — for example, NUMOSIM-LA:

```bash
python -m cli.train --dataset numosim --device cuda:0 \
    --time_modulo weekly --clique_size 8 \
    --num_epochs 200 --early_stopping_patience 20 \
    --perturb_locations --perturb_timestamps \
    --no-insert_visits --no-behavioral_anomaly \
    --early_stopping_criterion total_loss --ema_early_stopping \
    --lr 2e-4
```

The script `scripts/pretrain.sh` switches `--time_modulo`, `--clique_size`,
`--lambda_{min,max}`, `--num_epochs`, and `--early_stopping_patience` per
dataset to match Appendix Table 5 of the paper. eICU additionally drops the
agent embedding and uses a smaller backbone:

```bash
python -m cli.train --dataset eicu-demo --device cuda:0 \
    --time_modulo daily --clique_size 4 --no_agent_emb \
    --model_dim 128 --num_layers 2 --n_heads 4 --max_seq_len 32 \
    --num_epochs 200 --early_stopping_patience 40 \
    --early_stopping_criterion total_loss --ema_early_stopping --lr 2e-4
```

### 3.2 Fine-tuning per (task × variant)

TraXion and TraXion-NoColoc are fine-tuned **separately** from the same pre-trained
backbone. TraXion-NoColoc additionally passes `--no_neighbor_attn`.

```bash
# anomaly detection
bash scripts/finetune_anomaly.sh <dataset> <pretrain_ckpt> [variant] [--seed 0]

# next-POI recommendation
bash scripts/finetune_poi.sh <dataset> <pretrain_ckpt> [variant] [--seed 0]

# next-visit (joint POI + time)
bash scripts/finetune_next_visit.sh <dataset> <pretrain_ckpt> [variant] [--seed 0]

# social-link inference
bash scripts/finetune_social.sh <dataset> <pretrain_ckpt> [variant] [--seed 0]

# ICU mortality prediction
bash scripts/finetune_eicu.sh <pretrain_ckpt> [--seed 1]
```

`variant` is `active` for TraXion (default) or `bypass` for TraXion-NoColoc.

LANL fine-tuning uses a task-specific perturbation operator (`--user_host_swap`),
the noise-detection BCE head, and rank fusion of the two pre-training heads at
test time:

```bash
bash scripts/finetune_lanl.sh <pretrain_ckpt> [variant] [--seed 0]
# After fine-tuning, dump per-event scores and apply rank fusion:
python -m cli.evaluate_lanl_unsup --checkpoint runs/<finetune_id>.pt \
    --out_dir logs/lanl_unsup_dumps/<tag> --no_wandb --device cuda:0
python -m cli.score_lanl_unsup --dump_dir logs/lanl_unsup_dumps/<tag>
```

### 3.3 Ablation study (Section 4.6)

The ablation table reports seven variants C1–C7. **Each variant has its own
pre-training run**, and each task is then fine-tuned from that variant's own
backbone. Pre-training:

```bash
# Per-variant ablation pre-training on UA-Berlin / Gowalla-Austin-v1 / LANL.
bash scripts/ablation.sh pretrain <dataset> <variant>     # variant ∈ {C1..C7}
```

Fine-tuning a downstream task from a variant's backbone is identical to §3.2
but with `<pretrain_ckpt>` pointing at the variant's pre-training checkpoint
and (per the paper) reading the better of the active/bypass settings on
validation.

```bash
bash scripts/ablation.sh finetune <task> <dataset> <variant_pretrain_ckpt>
```

### 3.4 Evaluation (re-running test metrics on a saved checkpoint)

```bash
python -m cli.evaluate --checkpoint runs/<id>.pt --no_wandb --device cuda:0
python -m cli.evaluate_poi --checkpoint runs/poi_rec_<id>.pt --no_wandb --device cuda:0
python -m cli.evaluate_next_visit --checkpoint runs/next_visit_<id>.pt --no_wandb --device cuda:0
python -m cli.evaluate_social --checkpoint runs/social_<id>.pt --no_wandb --device cuda:0
python -m cli.evaluate_eicu --checkpoint runs/eicu_<id>.pt --no_wandb --device cuda:0
```

All evaluators print the metrics reported in the corresponding paper table
to four decimal places. Train scripts already run a final test-set evaluation
at the end of training; the standalone evaluators are useful for re-scoring
released checkpoints.


## 4. Mapping paper terminology to CLI flags

| Paper symbol / term                | CLI                                |
|-----------------------------------|------------------------------------|
| co-occurrence size $C$             | `--clique_size`                    |
| Space2Vec scale range $[\lambda_{\min},\lambda_{\max}]$ | `--lambda_min`, `--lambda_max` |
| Time2Vec wrap period $P$           | `--time_modulo {weekly,daily,none}`|
| $\mathcal{L}_{\text{noise}}$ off (C2 ablation) | `--no_noise_loss` (alias `--no_bce`) |
| prototype-as-input off (C3)        | `--no_agent_emb` + `--model_dim 1024` |
| flat transformer encoder (C7)      | `--use_transformer_encoder`        |
| TraXion-NoColoc (bypass at fine-tune)| `--no_neighbor_attn`               |
| LANL fine-tune perturbation $\eta_{\text{LANL}}$ | `--user_host_swap --user_host_swap_frac 0.3` |
| anchor-time in next-visit head     | `--use_anchor_time_in_head --anchor_time_emb_dim 64 --anchor_time_modulo daily` |


## 5. License

This codebase is released under the MIT License (`LICENSE`). Each public
dataset listed in §2 is consumed under its own published license (CC-BY 4.0,
ODbL, CC0, or research-citation terms); we do not redistribute any raw dataset.
Baseline implementations referenced in the paper retain the licenses of their
upstream repositories.

# TraXion: Rethinking Pre-training Frameworks for Mobility and Beyond

Official implementation of TraXion (NeurIPS 2026), released under the MIT License
(see [`LICENSE`](LICENSE)).

TraXion pre-trains one backbone per dataset on multi-entity spatiotemporal event
streams and fine-tunes it per downstream task. This repository contains the data
preparation, pre-training, fine-tuning, evaluation and ablation code behind every
TraXion number in the paper.

```
.
├── config.py                # dataset path registry (env-var driven)
├── HYPERPARAMETER_SEARCH.md # hyperparameter-selection protocol and the verbatim agent prompt
├── cli/                     # entry points (run as python -m cli.<name>)
│   ├── train.py             #   pre-training + anomaly fine-tuning
│   ├── train_poi.py         #   next-POI fine-tuning
│   ├── train_next_visit.py  #   next-visit (joint POI + time) fine-tuning
│   ├── train_social.py      #   social-link fine-tuning
│   ├── train_eicu.py        #   ICU-mortality fine-tuning (eICU-CRD, MIMIC-IV)
│   ├── train_ablation_pretrain.py   # autoregressive / masked-token pre-training (ablation C4 / C5)
│   ├── evaluate*.py         #   per-task evaluators
│   └── score_lanl_unsup.py  #   LANL scoring: rank fusion of the two pre-training heads
├── modules/                 # backbone (factorized attention, entity prototype, embeddings)
├── tasks/                   # per-task heads, datasets, criteria, metrics
├── utils/                   # argparse, dataset, perturbation operator, training runtime, ...
├── preprocess/              # data preparation, one script per source
└── scripts/                 # run scripts with the paper's hyperparameters
```

Run everything from the repository root (`python -m cli.<name> ...`).


## 1. Environment

```bash
conda env create -f environment.yml
conda activate traxion
```

or `pip install -r requirements.txt` in a Python 3.13 environment. The paper's
runs used Python 3.13 and PyTorch 2.6.0 (CUDA 12.4), each on one NVIDIA RTX 6000
Ada (48 GB). Fine-tuning on the full eICU-CRD corpus needs about 82 GiB of host
memory.

Paths are set with two environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `TRAXION_DATA_DIR` | `./data` | prepared datasets, `<dataset>/<dataset>_{train,test,poi,...}.parquet` |
| `TRAXION_RUNS_DIR` | `./runs` | checkpoints |

Training logs to Weights & Biases with `--use_wandb` (set `WANDB_ENTITY` and,
optionally, `--wandb_project`).


## 2. Data preparation

Download each dataset from its official source under its license; we do not
redistribute any raw dataset.

| Dataset | `--dataset` | Source / license |
|---|---|---|
| NUMOSIM-LA | `numosim` | OSF `osf.io/sjyfr` |
| Urban Anomalies-Berlin / -Atlanta | `urban-berlin`, `urban-atlanta` | OSF `osf.io/dg6t3` (CC-BY 4.0) |
| Foursquare-Tokyo | `foursquare-tokyo` | Yang et al., WWW 2019 (research use, citation) |
| Gowalla-Stockholm / -Austin | `gowalla-stockholm-v1`, `gowalla-austin-v1` | SNAP (research use, citation); POI categories from OpenStreetMap (ODbL) |
| LANL authentication log | `lanl-core` | `csr.lanl.gov/data/cyber1` (CC0) |
| eICU-CRD 2.0 | `eicu-full` (fine-tuning), `eicu-full-pid` (pre-training) | PhysioNet Credentialed Health Data License |
| MIMIC-IV 3.1 | `mimiciv-full` (fine-tuning), `mimiciv-full-pid` (pre-training) | PhysioNet Credentialed Health Data License |

```bash
# NUMOSIM-LA (stay_points_train.parquet, stay_points_test_anomalous.parquet, poi.parquet)
NUMOSIM_RAW_DIR=/path/to/NUMOSIM/LA python -m preprocess.prepare_numosim

# Urban Anomalies (staypoints_{train,test}.tsv of the *_combined_outliers release)
python -m preprocess.preprocess --dataset urban-berlin \
    --raw-train /path/to/Berlin/berlin_combined_outliers/staypoints_train.tsv \
    --raw-test  /path/to/Berlin/berlin_combined_outliers/staypoints_test.tsv
python -m preprocess.preprocess --dataset urban-atlanta \
    --raw-train /path/to/Atlanta/atlanta_combined_outliers/staypoints_train.tsv \
    --raw-test  /path/to/Atlanta/atlanta_combined_outliers/staypoints_test.tsv

# Foursquare-Tokyo and Gowalla-{Stockholm,Austin}-v1
FOURSQUARE_RAW_DIR=/path/to/dataset_WWW2019 GOWALLA_RAW_DIR=/path/to/loc-gowalla \
    python -m preprocess.prepare_city_subsets
# Gowalla POI categories: nearest OpenStreetMap feature within 50 m
# (downloads the 2026-04-21 OSM snapshot from the Overpass API)
python -m preprocess.assign_gowalla_categories --dataset gowalla-stockholm-v1
python -m preprocess.assign_gowalla_categories --dataset gowalla-austin-v1
# Social-link edge splits
for d in foursquare-tokyo gowalla-stockholm-v1 gowalla-austin-v1; do
    python -m preprocess.prepare_social_splits --dataset $d
done

# LANL (auth.txt.gz, redteam.txt.gz). Host coordinates are read from
# preprocess/assets/lanl-core_host_coords.parquet.
LANL_RAW_DIR=/path/to/lanl/cyber1 python -m preprocess.prepare_lanl

# eICU-CRD and MIMIC-IV: stay-keyed corpora, then patient-keyed copies for pre-training
EICU_RAW_DIR=/path/to/physionet.org/files/eicu-crd/2.0 python -m preprocess.prepare_eicu
MIMICIV_RAW_DIR=/path/to/physionet.org/files/mimiciv/3.1 python -m preprocess.prepare_mimiciv
python -m preprocess.rekey_patient --src eicu-full --dst eicu-full-pid
python -m preprocess.rekey_patient --src mimiciv-full --dst mimiciv-full-pid
```


## 3. Reproducing the paper

Each dataset has one pre-trained backbone (five on LANL, one per seed), shared by
all of its downstream tasks. Each task is then fine-tuned with seeds 1–5. For every
seed, both co-occurrence routes are fine-tuned — `active` (default) and `bypass`
(the pre-trained co-occurrence sub-layer is skipped, `--no_neighbor_attn`) — and
the route with the better validation criterion is kept:

```bash
python scripts/select_route.py --active runs/<active>.pt --bypass runs/<bypass>.pt
```

The selected checkpoint is then evaluated on the test split. Reported numbers are
mean ± standard deviation over the five seeds.

### 3.1 Pre-training

```bash
bash scripts/pretrain.sh <dataset> --device cuda:0 [--use_wandb]
```

`<dataset>` is one of `numosim`, `urban-atlanta`, `urban-berlin`,
`foursquare-tokyo`, `gowalla-stockholm-v1`, `gowalla-austin-v1`, `lanl-core`
(run with `--seed 0` … `--seed 4`), `eicu-full-pid`, `mimiciv-full-pid`. The
checkpoint is written to `$TRAXION_RUNS_DIR/<run id>.pt`.

### 3.2 Fine-tuning and evaluation

```bash
# Mobility anomaly detection (Table 1): numosim, urban-atlanta, urban-berlin
bash scripts/finetune_anomaly.sh <dataset> <pretrain_ckpt> <active|bypass> --seed <1-5>
python -m cli.evaluate --checkpoint runs/<id>.pt --no_wandb --device cuda:0

# Next-POI recommendation (Table 2): foursquare-tokyo, gowalla-stockholm-v1, gowalla-austin-v1
bash scripts/finetune_poi.sh <dataset> <pretrain_ckpt> <active|bypass> --seed <1-5>
python -m cli.evaluate_poi --checkpoint runs/poi_rec_<id>.pt --no_wandb --device cuda:0

# Next-visit prediction (Table 2); T±60 is the key time/t@60min/mass
bash scripts/finetune_next_visit.sh <dataset> <pretrain_ckpt> <active|bypass> --seed <1-5>
python -m cli.evaluate_next_visit --checkpoint runs/next_visit_<id>.pt --no_wandb --device cuda:0

# Social-link inference (Table 3); reports the "TEST:" line at the validation-selected alpha
bash scripts/finetune_social.sh <dataset> <pretrain_ckpt> <active|bypass> --seed <1-5>
python -m tasks.social.ensemble_eval --dataset <dataset> --social_checkpoint runs/social_<id>.pt --device cuda:0

# ICU mortality (Table 5): eicu-full, mimiciv-full, from the matching *-pid backbone.
# Test metrics are printed at the end of training; to recompute them:
bash scripts/finetune_eicu.sh <dataset> <pretrain_ckpt> <active|bypass> --seed <1-5>
python -m cli.evaluate_eicu --dataset <dataset> --resume_from_checkpoint runs/eicu_<id>.pt --device cuda:0
```

**Enterprise authentication logs (Table 4).** The pre-trained LANL backbones are
scored directly, without fine-tuning, by rank-fusing the noise-detection and
prototype-contrastive heads. For each of the five backbones:

```bash
python -m cli.evaluate_lanl_unsup --checkpoint runs/<pretrain_id>.pt \
    --out_dir logs/lanl/<pretrain_id> --skip_train --batch_size 16 --device cuda:0
python -m cli.score_lanl_unsup --dump_dir logs/lanl/<pretrain_id>
```

`score_lanl_unsup` prints event-level (`event/*`) and user-level (`user/*`) AP,
AUROC and Max F1.

### 3.3 Ablation study (Table 6)

`scripts/ablation.sh` runs every cell of Table 6 (rows Full and C1–C7)
with all hyperparameters passed explicitly.

```bash
bash scripts/ablation.sh pretrain   <arm> <task> [--seed N] [--device cuda:0]
bash scripts/ablation.sh finetune   <arm> <task> <pretrain_ckpt> [--device cuda:0]
bash scripts/ablation.sh finetune   C6    <task> [--seed N] [--device cuda:0]
bash scripts/ablation.sh evaluate   <task> <finetuned_ckpt> [--device cuda:0]
bash scripts/ablation.sh score-lanl <ckpt> <out_dir> [--device cuda:0]
```

| `task` | Dataset | Route | Reported metrics |
|---|---|---|---|
| `mobility_ad` | UA-Berlin | active | `combined/agent/AP`, `combined/agent/Max F1` |
| `next_visit` | Gowalla-Austin | bypass | `loc/recall@10`, `time/t@60min/mass` |
| `social` | Gowalla-Austin | active | AP, Hits@10 (model score) |
| `auth_log` | LANL | active | `user/AP`, `user/MaxF1` (`score-lanl`) |

`arm` is one of `Full`, `C1` (w/o L_prototype), `C2` (w/o L_noise),
`C3` (w/o prototype-as-input), `C4` (autoregressive next-token pre-training),
`C5` (masked-token pre-training), `C6` (no pre-training) and
`C7` (flat transformer encoder). Pre-training is per arm and task; C6 has
none and is trained from random weights by `finetune C6 <task>`.

Cells with ± average five runs:

- `mobility_ad`: Full and C5 over pre-training seeds 0, 101, 202, 303, 404.
- `next_visit`: Full over pre-training seeds 101, 202, 303, 404, 505; C4 and C5 over 0, 101, 202, 303, 404.
- `social`: every pre-trained arm over pre-training seeds 0, 101, 202, 303, 404; C6 over fine-tuning seeds 0, 101, 202, 303, 404.
- `auth_log`: Full over pre-training seeds 0–4; C6 over seeds 0, 101, 202, 303, 404.

The fine-tuning seed is fixed by the script. Full on `auth_log` is scored
without fine-tuning:

```bash
bash scripts/ablation.sh pretrain Full auth_log --seed 1 --device cuda:0
bash scripts/ablation.sh score-lanl runs/<id>.pt logs/ablation/full_s1 --device cuda:0
```

Example for one single-run cell (C3, next-visit):

```bash
bash scripts/ablation.sh pretrain C3 next_visit --device cuda:0
bash scripts/ablation.sh finetune C3 next_visit runs/<pretrain_id>.pt --device cuda:0
bash scripts/ablation.sh evaluate next_visit runs/next_visit_<id>.pt --device cuda:0
```

### 3.4 Hyperparameter selection

The values in `scripts/*.sh` are the outcome of a validation-guided iterative
search, applied identically to TraXion and to every retuned baseline. See
[`HYPERPARAMETER_SEARCH.md`](HYPERPARAMETER_SEARCH.md) for the protocol and the
verbatim prompt of the agent that ran it.


## 4. Paper notation and CLI flags

| Paper | CLI |
|---|---|
| co-occurrence window size $C$ | `--clique_size` |
| Space2Vec scale range $[\lambda_{\min},\lambda_{\max}]$ | `--lambda_min`, `--lambda_max` |
| Time2Vec wrap period | `--time_modulo {weekly,daily,none}` |
| co-occurrence route: bypass | `--no_neighbor_attn` |
| context / time perturbation of the operator $\eta$ | `--perturb_locations`, `--perturb_timestamps` |
| ablation C1 (w/o $\mathcal{L}_{\text{prototype}}$) | `--no_contrastive` |
| ablation C2 (w/o $\mathcal{L}_{\text{noise}}$) | `--no_bce` |
| ablation C3 (w/o prototype-as-input) | `--no_agent_emb --model_dim 1024` |
| ablation C4 / C5 (autoregressive / masked-token) | `cli.train_ablation_pretrain --pretrain_objective {causal,stm}` (+ `--keep_prototype` to add $\mathcal{L}_{\text{prototype}}$) |
| ablation C7 (flat transformer encoder) | `--use_transformer_encoder` |


## 5. Citation

```bibtex
@inproceedings{hsu2026traxion,
  title     = {TraXion: Rethinking Pre-training Frameworks for Mobility and Beyond},
  author    = {Hsu, Shang-Ling and Tenzer, Mark and Shahabi, Cyrus and Shafique, Khurram},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```


## 6. License

The code is released under the MIT License. Each dataset is used under its own
license (§2). Gowalla POI categories are derived from OpenStreetMap data,
© OpenStreetMap contributors, available under the Open Database License.

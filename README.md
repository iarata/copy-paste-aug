# copy-paste-aug

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Copy-Paste Augmentation for ADLCV 2026 Spring

## Install

```bash
make requirements
```

## Data (full COCO-2017)

```bash
make download-coco OUTPUT=data/raw/coco2017.zip
```
After downloading and **unzipping**, the COCO 2017 dataset, make sure to update the `configs/dataset/default.yaml` file with the correct path to the dataset.

## Train
You must make sure that your uv env is active before running train
```bash
make train
```

Useful overrides:

```bash
make train ARGS="dataset.batch_size=16"
make train ARGS="models.scale=s"
make train ARGS="dataset.augmentations.name=none"
make train ARGS="dataset.augmentations.name=cpa dataset.augmentations.prob=1.0"
make train ARGS="dataset.train_subset_percent=10 dataset.val_subset_percent=20 seed=123"
make train ARGS="models.name=configs/models/yolo26/yolo26-p6.yaml"
make train ARGS="models.name=yolo26-seg.yaml models.scale=m"
make train ARGS="training.limit_train_batches=10 training.limit_val_batches=10 evaluation.run_after_fit=false"

# if debug is True it will run on fast-dev mode which just runs 1 epoch
make train ARGS="debug=true"

# training DEITR transformer with/without copy-paste augmentation
make train ARGS="--config-name instance_transformer_no_aug debug=False dataset.batch_size=32 evaluation.run_epoch_metrics=false"
make train ARGS="--config-name instance_transformer_aug debug=False dataset.batch_size=32 evaluation.run_epoch_metrics=false"
```
## YOLO26 Pipeline

The default setup targets COCO 2017 instance segmentation trained from scratch:

- `models.name=configs/models/yolo26/yolo26-seg.yaml`
- `models.scale=n`
- `models.weights=null`
- `dataset.task=segment`

All upstream YOLO26 config families from `ultralytics/cfg/models/26` are mirrored under [`configs/models/yolo26`](/Users/arata/Desktop/Research/copy-paste-aug/configs/models/yolo26), so changing architectures is just a config override.
You can also point `models.name` at an Ultralytics built-in model name like `yolo26-seg.yaml`; `models.scale` will expand it to the right family member such as `yolo26s-seg.yaml`.

Copy-paste experiment options:

- `dataset.augmentations.name=none`
- `dataset.augmentations.name=ultralytics_flip`
- `dataset.augmentations.name=ultralytics_mixup`
- `dataset.augmentations.name=cpa`

## Premade COCO Copy-Paste Datasets

Offline copy-paste datasets are generated under a separate output root, so the
original COCO2017 folder is never modified.  The default selector builds a
category-balanced train and validation subset: each class keeps at least the
requested percentage of images containing that class.  If
`--val-subset-percent` is omitted, it defaults to `--train-subset-percent`.

```bash
make premade-coco ARGS="\
  --source-root data.nosync/raw/coco2017 \
  --output-root data.nosync/processed/coco2017_simple_cp_seed42_sub50 \
  --method simple \
  --seed 42 \
  --train-subset-percent 50 \
  --copy-paste-percent 100 \
  --workers 8 \
  --parallel-backend process"
```

The output contains COCO JSON files, a `coco_data.yaml`, a `manifest.json`, and
list files under `lists/` for original, augmented, and combined train images.
Use `--parallel-backend thread` for thread workers, `--parallel-backend process`
for multiprocessing, and `--no-progress` to disable tqdm progress bars.  Loguru
logging is enabled by the CLI and can be controlled with `--log-level`.  If a
long premade build is interrupted, rerun the same command with `--resume`; valid
completed augmented images are reused and only missing/corrupt tasks are
generated again.  The executor only keeps a bounded number of submitted tasks in
memory; override it with `--max-in-flight N` if needed.  Harmonized CUDA/MPS runs
default to `--max-in-flight 1` because the GPU model is serialized internally.
By default, source-image symlink aliases are removed from the output tree after a
successful build and original-image COCO records point back to the source COCO
files.  Pass `--no-cleanup-aliases` if you want to keep those symlinks in the
output folder.
To train on the premade dataset, point the dataset config at the generated root
and disable online augmentation:

```bash
make train ARGS="\
  dataset.root=data.nosync/processed/coco2017_simple_cp_seed42_sub50 \
  dataset.train_json=annotations/instances_train2017.json \
  dataset.val_json=annotations/instances_val2017.json \
  dataset.train_images=train2017 \
  dataset.val_images=val2017 \
  dataset.augmentations.name=none"
```

The generator is method-registry based (`--method simple|harmonized`), so additional
offline copy-paste methods can be added without changing the CLI contract.
`--method harmonized` uses the Libcom PCTNet/LBM model code directly after the
same copy-paste placement as `simple`, avoiding Libcom's CUDA-only public
`ImageHarmonizationModel` wrapper. It accepts `--harmonization-model-type
PCTNet`, `--harmonization-model-type PCNet` (alias for `PCTNet`), or
`--harmonization-model-type LBM`, plus `--harmonization-device auto|cpu|mps|cuda`.
LBM also uses `--harmonization-steps` and `--harmonization-resolution`; the
resolution must be divisible by 8.
Threaded CUDA/MPS harmonization uses one shared model and serializes model
inference for reproducibility, so `--max-in-flight > 1` does not run multiple
GPU forwards with `--parallel-backend thread`.  To use extra VRAM for LBM, run
separate process workers instead; each process loads its own model copy, so set
`--workers` to the same value as `--max-in-flight` and increase gradually while
watching VRAM.  Lowering `--harmonization-steps` gives near-linear speedups, and
lowering `--harmonization-resolution` reduces compute roughly with image area.

```bash
make premade-coco ARGS="\
  --source-root data.nosync/raw/coco2017 \
  --output-root data.nosync/processed/coco2017_harmonized_pct_seed42_sub50 \
  --method harmonized \
  --seed 42 \
  --train-subset-percent 50 \
  --copy-paste-percent 100 \
  --harmonization-model-type PCTNet \
  --harmonization-device auto \
  --workers 1"
```

```bash
python -m cpa.premade_datasets.coco2017 \
  --source-root data/raw/coco2017 \
  --output-root data/processed/coco2017_harmonized_lbm_seed42_sub50 \
  --method harmonized \
  --seed 42 \
  --train-subset-percent 50 \
  --copy-paste-percent 100 \
  --harmonization-model-type LBM \
  --harmonization-device cuda \
  --parallel-backend process \
  --workers 2 \
  --max-in-flight 2 \
  --resume
```

### Evaluate

```bash
make eval ARGS="evaluation.checkpoint_path=/abs/path/to/checkpoints/last.ckpt"
```

Evaluation writes metrics to `evaluation/metrics.json` inside the Hydra run directory and uses the Ultralytics validators for COCO-style metrics.
During training, epoch-end validator metrics can be toggled with `evaluation.run_epoch_metrics=true|false`. When enabled, they run at the end of every epoch, are written under `validation_benchmark/epoch_XXX/metrics.json`, and are logged to W&B with `val/*` and `benchmark/*` series such as `val/mAP50`, `val/mAP50-95`, `val/precision`, `val/recall`, `val/f1`, `benchmark/mAP50-95`, and `benchmark/inference_ms_per_image`.
Training and validation losses are logged epoch-wise so W&B charts line up with the validator metrics on the same epoch axis.

-------------

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── configs            <- Configuration files for trained models and experiments
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         cpa and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── cpa   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes cpa a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

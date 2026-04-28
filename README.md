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

```bash
make train
```

Useful overrides:

```bash
make train ARGS="dataset.batch_size=16"
make train ARGS="models.scale=s"
make train ARGS="dataset.augmentations.name=none"
make train ARGS="dataset.augmentations.name=cpa dataset.augmentations.prob=1.0"
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

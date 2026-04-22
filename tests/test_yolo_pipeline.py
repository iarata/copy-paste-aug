from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from cpa.training import build_trainer, log_wandb_run_config, update_wandb_summary
from cpa.utils.configs import (
    AugmentationsConfig,
    Config,
    DatasetConfig,
    EvaluationConfig,
    ModelsConfig,
    TrainingConfig,
    WandbConfig,
)
from cpa.yolo.data import COCOJsonDataModule
from cpa.yolo.lightning import YOLO26LightningModule, resolve_model_source, summarize_validator_metrics


def _categories() -> list[dict[str, object]]:
    return [{"id": index + 1, "name": f"class_{index}"} for index in range(80)]


def _image_record(image_id: int) -> dict[str, object]:
    return {"id": image_id, "file_name": f"{image_id:012d}.jpg", "height": 128, "width": 128}


def _annotation(ann_id: int, image_id: int) -> dict[str, object]:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": [[16.0, 16.0, 64.0, 16.0, 64.0, 64.0, 16.0, 64.0]],
        "bbox": [16.0, 16.0, 48.0, 48.0],
        "area": 48.0 * 48.0,
        "iscrowd": 0,
    }


@pytest.fixture
def coco_root(tmp_path: Path) -> Path:
    root = tmp_path / "coco2017"
    (root / "train2017").mkdir(parents=True)
    (root / "val2017").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    rng = np.random.default_rng(0)
    train_ids = [1, 2]
    val_ids = [3]

    for image_id in train_ids:
        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(
            root / "train2017" / f"{image_id:012d}.jpg"
        )
    for image_id in val_ids:
        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(
            root / "val2017" / f"{image_id:012d}.jpg"
        )

    train_json = {
        "images": [_image_record(image_id) for image_id in train_ids],
        "annotations": [_annotation(index + 1, image_id) for index, image_id in enumerate(train_ids)],
        "categories": _categories(),
    }
    val_json = {
        "images": [_image_record(image_id) for image_id in val_ids],
        "annotations": [_annotation(100 + index, image_id) for index, image_id in enumerate(val_ids)],
        "categories": _categories(),
    }
    (root / "annotations" / "instances_train2017.json").write_text(json.dumps(train_json), encoding="utf-8")
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps(val_json), encoding="utf-8")
    return root


def _config(
    coco_root: Path,
    *,
    aug_name: str,
    task: str = "segment",
    model_name: str = "configs/models/yolo26/yolo26-seg.yaml",
    model_scale: str = "n",
) -> Config:
    dataset_cfg = DatasetConfig(
        root=str(coco_root),
        train_images="train2017",
        val_images="val2017",
        train_json="annotations/instances_train2017.json",
        val_json="annotations/instances_val2017.json",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        cache=False,
        task=task,
        augmentations=AugmentationsConfig(
            name=aug_name,
            prob=1.0,
            mode="mixup",
            blend=False,
            sigma=1.0,
            pct_objects_paste=1.0,
            max_paste_objects=4,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
        ),
    )
    return Config(
        dataset=dataset_cfg,
        models=ModelsConfig(name=model_name, scale=model_scale, weights=None),
        training=TrainingConfig(epochs=1, optimizer="AdamW", precision="32-true"),
        evaluation=EvaluationConfig(run_after_fit=False),
        wandb=WandbConfig(project="tests", mode="disabled"),
    )


def test_datamodule_emits_ultralytics_batch(coco_root: Path):
    cfg = _config(coco_root, aug_name="none")
    datamodule = COCOJsonDataModule(cfg.dataset, project_root=Path.cwd())
    datamodule.setup("fit")

    batch = next(iter(datamodule.train_dataloader()))

    assert {"img", "cls", "bboxes", "batch_idx", "im_file", "ori_shape", "masks"}.issubset(batch.keys())
    assert batch["img"].shape[0] == 1
    assert batch["bboxes"].shape[1] == 4
    assert batch["cls"].shape[1] == 1
    assert batch["masks"].ndim == 3


def test_lightning_module_computes_yolo26_loss(coco_root: Path):
    cfg = _config(coco_root, aug_name="none")
    datamodule = COCOJsonDataModule(cfg.dataset, project_root=Path.cwd())
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    module = YOLO26LightningModule(cfg, datamodule.names, project_root=Path.cwd())
    batch = module._preprocess_batch(batch)
    loss, loss_items = module.model.loss(batch)

    assert torch.isfinite(loss).all()
    assert torch.isfinite(loss_items).all()
    assert loss_items.numel() == 5


def test_local_family_yaml_respects_models_scale(coco_root: Path):
    cfg = _config(coco_root, aug_name="none", model_scale="s")
    datamodule = COCOJsonDataModule(cfg.dataset, project_root=Path.cwd())
    module = YOLO26LightningModule(cfg, datamodule.names, project_root=Path.cwd())

    assert module.model.yaml.get("scale") == "s"


def test_builtin_family_yaml_resolves_scaled_ultralytics_name():
    resolved = resolve_model_source("yolo26-seg.yaml", "s", project_root=None)

    assert resolved == "yolo26s-seg.yaml"


def test_render_sample_panel_returns_side_by_side_visual(coco_root: Path):
    cfg = _config(coco_root, aug_name="none")
    datamodule = COCOJsonDataModule(cfg.dataset, project_root=Path.cwd())
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))

    module = YOLO26LightningModule(cfg, datamodule.names, project_root=Path.cwd())
    batch = module._preprocess_batch(batch)
    panel = module._render_sample_panel(batch, sample_idx=0)

    assert panel.dtype == np.uint8
    assert panel.shape[0] > 0
    assert panel.shape[1] == batch["img"].shape[3] * 2
    assert panel.shape[2] == 3


def test_custom_copy_paste_pipeline_runs(coco_root: Path):
    cfg = _config(coco_root, aug_name="cpa")
    datamodule = COCOJsonDataModule(cfg.dataset, project_root=Path.cwd())
    datamodule.setup("fit")

    sample = datamodule.train_dataset[0]

    assert "img" in sample
    assert sample["img"].shape[0] == 3
    assert "bboxes" in sample
    assert "masks" in sample


def test_summarize_validator_metrics_adds_benchmark_fields():
    summary = summarize_validator_metrics(
        task="segment",
        metrics={
            "metrics/mAP50-95(M)": 0.42,
            "metrics/mAP50(M)": 0.7,
            "fitness": 0.4,
        },
        speed={"preprocess": 1.5, "inference": 5.0, "postprocess": 0.5},
    )

    assert summary["benchmark/mAP50-95"] == pytest.approx(0.42)
    assert summary["benchmark/inference_ms_per_image"] == pytest.approx(5.0)
    assert summary["benchmark/mAP50-95_per_ms"] == pytest.approx(0.084)
    assert summary["benchmark/preprocess_ms_per_image"] == pytest.approx(1.5)
    assert summary["benchmark/mAP50_95_mask"] == pytest.approx(0.42)


def test_debug_builds_fast_dev_run_trainer(coco_root: Path, tmp_path: Path):
    cfg = _config(coco_root, aug_name="none")
    cfg.debug = True

    trainer = build_trainer(cfg, wandb_logger=None, default_root_dir=tmp_path)

    assert trainer.fast_dev_run == 1


def test_wandb_helpers_tolerate_dummy_experiment() -> None:
    class DummyExperiment:
        @property
        def summary(self):
            return lambda *args, **kwargs: None

    class DummyLogger:
        def __init__(self):
            self.logged = None
            self.experiment = DummyExperiment()

        def log_hyperparams(self, params):
            self.logged = params

    logger = DummyLogger()

    payload = {"model_name": "yolo26-seg.yaml", "debug": False}
    log_wandb_run_config(logger, payload)
    update_wandb_summary(logger, {"benchmark/mAP50-95": 0.5})

    assert logger.logged == payload

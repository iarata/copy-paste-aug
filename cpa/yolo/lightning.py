from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

import cv2
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import converter
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.optim import MuSGD
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import one_cycle, unwrap_model

from cpa.yolo.data import resolve_path

LOSS_NAMES: dict[str, tuple[str, ...]] = {
    "detect": ("box_loss", "cls_loss", "dfl_loss"),
    "segment": ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"),
}


def materialize_scaled_model_yaml(model_path: str | Path, scale: str | None) -> str:
    path = Path(model_path)
    if path.suffix not in {".yaml", ".yml"} or not scale:
        return str(path)

    scale = str(scale).strip().lower()
    if scale not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"Unsupported models.scale '{scale}'. Expected one of n, s, m, l, x.")

    basename = path.name
    if re.search(r"(yolo26|yoloe-26)[nslmx](?=-|\.yaml$|\.yml$)", basename):
        return str(path)

    if basename.startswith("yoloe-26"):
        scaled_name = basename.replace("yoloe-26", f"yoloe-26{scale}", 1)
    elif basename.startswith("yolo26"):
        scaled_name = basename.replace("yolo26", f"yolo26{scale}", 1)
    else:
        scaled_name = f"{path.stem}-{scale}{path.suffix}"

    if not path.exists():
        return str(path.with_name(scaled_name))

    generated_dir = Path.cwd() / ".model_cache"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_dir / scaled_name
    if not generated_path.exists():
        generated_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(generated_path)


def resolve_model_source(model_name: str | Path, scale: str | None, project_root: Path | None) -> str:
    model_source = str(model_name)
    candidate = Path(model_source)
    if project_root is not None and candidate.suffix in {".pt", ".yaml", ".yml"}:
        resolved = resolve_path(candidate, project_root)
        if resolved.exists():
            model_source = str(resolved)
    return materialize_scaled_model_yaml(model_source, scale)


def primary_map50_95_key(task: str, metrics: dict[str, Any]) -> str:
    if task == "segment" and "metrics/mAP50-95(M)" in metrics:
        return "metrics/mAP50-95(M)"
    if "metrics/mAP50-95(B)" in metrics:
        return "metrics/mAP50-95(B)"
    return "metrics/mAP50-95"


def summarize_validator_metrics(
    *,
    task: str,
    metrics: dict[str, Any],
    speed: dict[str, float] | None = None,
) -> dict[str, float]:
    summary: dict[str, float] = {}

    for key, value in metrics.items():
        if not isinstance(value, int | float):
            continue
        metric_name = (
            key.removeprefix("metrics/")
            .replace("(B)", "_box")
            .replace("(M)", "_mask")
            .replace("-", "_")
            .replace("/", "_")
        )
        summary[f"benchmark/{metric_name}"] = float(value)

    speed = speed or {}
    for name, value in speed.items():
        summary[f"benchmark/{name}_ms_per_image"] = float(value)

    primary_key = primary_map50_95_key(task, metrics)
    primary_value = float(metrics.get(primary_key, 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    summary["benchmark/mAP50-95"] = primary_value
    summary["benchmark/inference_ms_per_image"] = inference_ms
    summary["benchmark/mAP50-95_per_ms"] = primary_value / inference_ms if inference_ms > 0 else 0.0
    return summary


def run_validator_without_fusing_model(validator: Any, model: nn.Module) -> dict[str, Any]:
    """Run an Ultralytics validator against a live training model without mutating it.

    Ultralytics validators wrap nn.Module inputs in AutoBackend with `fuse=True`
    by default. For YOLO26 segmentation heads, fusing calls `Proto26.fuse()`,
    which drops the semantic branch used during training. We patch the validator's
    AutoBackend reference so benchmarking can reuse the live model safely.
    """

    class _NonFusingAutoBackend(AutoBackend):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["fuse"] = False
            super().__init__(*args, **kwargs)

    with patch("ultralytics.engine.validator.AutoBackend", _NonFusingAutoBackend):
        return validator(model=model)


def build_ultralytics_optimizer(
    model: nn.Module,
    *,
    name: str,
    lr: float,
    momentum: float,
    decay: float,
    iterations: int,
    nc: int,
) -> torch.optim.Optimizer:
    groups: list[Any] = [{}, {}, {}, {}]
    norm_layers = tuple(
        layer for layer in nn.__dict__.values() if isinstance(layer, type) and "Norm" in layer.__name__
    )

    if name == "auto":
        lr_fit = round(0.002 * 5 / (4 + nc), 6)
        name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10_000 else ("AdamW", lr_fit, 0.9)

    use_muon = name == "MuSGD"
    for module_name, module in unwrap_model(model).named_modules():
        for param_name, parameter in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if parameter.ndim >= 2 and use_muon:
                groups[3][fullname] = parameter
            elif "bias" in fullname:
                groups[2][fullname] = parameter
            elif isinstance(module, norm_layers) or "logit_scale" in fullname:
                groups[1][fullname] = parameter
            else:
                groups[0][fullname] = parameter

    if not use_muon:
        groups = [group.values() for group in groups[:3]]

    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optim_args = dict(lr=lr, momentum=momentum)
    elif name in {"SGD", "MuSGD"}:
        optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {name}")

    groups[2] = {"params": groups[2], **optim_args, "param_group": "bias"}
    groups[0] = {"params": groups[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
    groups[1] = {"params": groups[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}

    if use_muon:
        import re

        groups[3] = {
            "params": groups[3],
            **optim_args,
            "weight_decay": decay,
            "use_muon": True,
            "param_group": "muon",
        }
        pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg")
        split_groups = []
        for group in groups:
            params = group.pop("params")
            primary = [value for key, value in params.items() if pattern.search(key)]
            secondary = [value for key, value in params.items() if not pattern.search(key)]
            split_groups.extend([{"params": primary, **group, "lr": lr * 3}, {"params": secondary, **group}])
        groups = split_groups

    optimizer_cls = getattr(optim, name, partial(MuSGD, muon=0.2, sgd=1.0))
    optimizer = optimizer_cls(params=groups)
    print(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum})")
    return optimizer


class YOLO26LightningModule(L.LightningModule):
    def __init__(self, cfg: Any, names: dict[int, str], project_root: Path | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.names = names
        self.project_root = project_root
        self.task = cfg.dataset.task
        self.loss_names = LOSS_NAMES[self.task]
        self.max_wandb_samples = 4
        self._last_wandb_sample_epoch = -1
        self.resolved_model_source = resolve_model_source(
            cfg.models.name,
            getattr(cfg.models, "scale", None),
            project_root,
        )
        self.save_hyperparameters(
            {
                "model_name": cfg.models.name,
                "model_scale": getattr(cfg.models, "scale", None),
                "model_weights": getattr(cfg.models, "weights", None),
                "resolved_model_source": self.resolved_model_source,
                "task": self.task,
            }
        )

        ultra_model = YOLO(self.resolved_model_source)
        if getattr(cfg.models, "weights", None):
            weights = cfg.models.weights
            if project_root is not None and Path(weights).suffix in {".pt"}:
                weights_candidate = resolve_path(weights, project_root)
                weights = str(weights_candidate) if weights_candidate.exists() else weights
            ultra_model.model.load(weights)

        if ultra_model.task != self.task:
            raise ValueError(f"Model task '{ultra_model.task}' does not match dataset task '{self.task}'.")

        self.model = ultra_model.model
        self.model.names = names
        self.model.nc = len(names)
        self.model.args = get_cfg(overrides={"task": self.task})
        if getattr(self.model, "end2end", False):
            self.model.set_head_attr(max_det=300)

    def on_fit_start(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        experiment = self.logger.experiment
        experiment.define_metric("epoch")
        for pattern in ("train/*", "val/*", "benchmark/*", "samples/*", "lr-*"):
            experiment.define_metric(pattern, step_metric="epoch")

    def _preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch["img"] = batch["img"].float() / 255.0
        return batch

    def _class_color(self, class_id: int) -> tuple[int, int, int]:
        palette = (
            (235, 87, 87),
            (47, 128, 237),
            (39, 174, 96),
            (242, 153, 74),
            (155, 81, 224),
            (45, 156, 219),
            (111, 207, 151),
            (246, 189, 96),
        )
        return palette[class_id % len(palette)]

    def _render_sample_panel(self, batch: dict[str, Any], sample_idx: int) -> np.ndarray:
        image = batch["img"][sample_idx].detach().cpu().clamp(0, 1)
        image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        raw = image.copy()
        annotated = image.copy()
        height, width = annotated.shape[:2]

        sample_mask = batch["batch_idx"].detach().cpu() == sample_idx
        boxes = batch["bboxes"][sample_mask].detach().cpu().numpy()
        classes = batch["cls"][sample_mask].detach().cpu().numpy().astype(int).reshape(-1)

        if self.task == "segment" and "masks" in batch:
            overlap_mask = batch["masks"][sample_idx].detach().cpu().numpy().astype(np.int32)
            if overlap_mask.size:
                resized_mask = cv2.resize(overlap_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                overlay = annotated.copy()
                for instance_id, class_id in enumerate(classes, start=1):
                    mask = resized_mask == instance_id
                    if not np.any(mask):
                        continue
                    color = np.asarray(self._class_color(int(class_id)), dtype=np.uint8)
                    overlay[mask] = ((0.45 * overlay[mask]) + (0.55 * color)).astype(np.uint8)
                annotated = overlay

        for box, class_id in zip(boxes, classes, strict=False):
            x_center, y_center, box_width, box_height = box.tolist()
            x1 = max(int((x_center - box_width / 2) * width), 0)
            y1 = max(int((y_center - box_height / 2) * height), 0)
            x2 = min(int((x_center + box_width / 2) * width), width - 1)
            y2 = min(int((y_center + box_height / 2) * height), height - 1)
            color = self._class_color(int(class_id))
            label = self.names.get(int(class_id), str(int(class_id)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(raw, "input", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "targets",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return np.concatenate([raw, annotated], axis=1)

    def _maybe_log_wandb_samples(self, batch: dict[str, Any], split: str) -> None:
        if self._last_wandb_sample_epoch == self.current_epoch:
            return
        if self.trainer is None or self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        if not isinstance(self.logger, WandbLogger):
            return

        try:
            import wandb
        except ImportError:
            return

        count = min(int(batch["img"].shape[0]), self.max_wandb_samples)
        images = [
            wandb.Image(
                self._render_sample_panel(batch, sample_idx),
                caption=f"epoch={self.current_epoch} sample={sample_idx}",
            )
            for sample_idx in range(count)
        ]
        if not images:
            return

        self.logger.experiment.log(
            {
                "epoch": int(self.current_epoch),
                f"samples/{split}": images,
            }
        )
        self._last_wandb_sample_epoch = self.current_epoch

    def _should_run_validation_benchmark(self) -> bool:
        if self.trainer is None or self.trainer.sanity_checking or self.trainer.fast_dev_run:
            return False
        if not self.trainer.is_global_zero:
            return False
        interval = int(getattr(self.cfg.evaluation, "benchmark_every_n_epochs", 1))
        return interval > 0 and (int(self.current_epoch) + 1) % interval == 0

    def _run_validation_benchmark(self) -> dict[str, float]:
        if self.trainer is None or self.trainer.datamodule is None:
            return {}

        datamodule = self.trainer.datamodule
        output_dir = Path(self.trainer.default_root_dir) / "validation_benchmark" / f"epoch_{self.current_epoch:03d}"
        data_yaml = datamodule.write_data_yaml(output_dir / "coco_data.yaml")
        validator_cls = (
            COCOJsonSegmentationValidator if self.task == "segment" else COCOJsonDetectionValidator
        )
        validator = validator_cls(
            dataloader=datamodule.val_dataloader(),
            save_dir=output_dir,
            args=build_validator_args(data_yaml, self.cfg, int(datamodule.eval_batch_size)),
        )
        metrics = run_validator_without_fusing_model(validator, self.model)
        benchmark_metrics = summarize_validator_metrics(task=self.task, metrics=metrics, speed=validator.speed)
        (output_dir / "metrics.json").write_text(
            json.dumps({**metrics, **benchmark_metrics}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return benchmark_metrics

    def _log_loss_items(
        self,
        prefix: str,
        loss_total: torch.Tensor,
        loss_items: torch.Tensor,
        batch_size: int,
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        self.log(
            f"{prefix}/loss",
            loss_total,
            prog_bar=prefix == "val",
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=batch_size,
            sync_dist=True,
        )
        for name, value in zip(self.loss_names, loss_items, strict=False):
            self.log(
                f"{prefix}/{name}",
                value.detach(),
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _compute_loss(
        self,
        batch: dict[str, Any],
        *,
        prefix: str,
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_loss, loss_items = self.model.loss(batch)
        raw_loss = raw_loss.sum() if raw_loss.ndim > 0 else raw_loss
        loss_items = loss_items.detach().float()
        loss_total = loss_items.sum()

        if not torch.isfinite(raw_loss).all() or not torch.isfinite(loss_items).all():
            rendered = ", ".join(
                f"{name}={value.item():.6g}" for name, value in zip(self.loss_names, loss_items, strict=False)
            )
            raise FloatingPointError(
                f"Non-finite {prefix} loss at epoch={self.current_epoch} batch={batch_idx}: {rendered}"
            )

        return raw_loss, loss_total, loss_items

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._preprocess_batch(batch)
        loss, loss_total, loss_items = self._compute_loss(batch, prefix="train", batch_idx=batch_idx)
        self._log_loss_items("train", loss_total, loss_items, batch["img"].shape[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._preprocess_batch(batch)
        loss, loss_total, loss_items = self._compute_loss(batch, prefix="val", batch_idx=batch_idx)
        self._log_loss_items("val", loss_total, loss_items, batch["img"].shape[0], on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._maybe_log_wandb_samples(batch, split="val")
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("epoch", float(self.current_epoch), on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
        if not self._should_run_validation_benchmark():
            return

        benchmark_metrics = self._run_validation_benchmark()
        if benchmark_metrics:
            self.log_dict(
                benchmark_metrics,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )

    def configure_optimizers(self) -> dict[str, Any]:
        iterations = max(
            int(getattr(self.trainer, "estimated_stepping_batches", self.cfg.training.epochs)), 1
        )
        optimizer = build_ultralytics_optimizer(
            self.model,
            name=self.cfg.training.optimizer,
            lr=float(self.cfg.training.lr0),
            momentum=float(self.cfg.training.momentum),
            decay=float(self.cfg.training.weight_decay),
            iterations=iterations,
            nc=len(self.names),
        )
        if self.cfg.training.cos_lr:
            lr_lambda = one_cycle(1, float(self.cfg.training.lrf), int(self.cfg.training.epochs))
        else:

            def lr_lambda(epoch: int) -> float:
                return max(1 - epoch / int(self.cfg.training.epochs), 0) * (
                    1.0 - float(self.cfg.training.lrf)
                ) + float(self.cfg.training.lrf)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class COCOJsonDetectionValidator(DetectionValidator):
    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        if self.data.get("coco_eval", False) and len(model.names) == 80:
            self.is_coco = True
            self.class_map = converter.coco80_to_coco91_class()
            self.args.save_json |= self.args.val and not self.training


class COCOJsonSegmentationValidator(SegmentationValidator):
    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        if self.data.get("coco_eval", False) and len(model.names) == 80:
            self.is_coco = True
            self.class_map = converter.coco80_to_coco91_class()
            self.args.save_json |= self.args.val and not self.training


def build_validator_args(data_yaml: Path, cfg: Any, batch_size: int) -> dict[str, Any]:
    device = "0" if torch.cuda.is_available() else "cpu"
    return {
        "data": str(data_yaml),
        "imgsz": int(cfg.dataset.imgsz),
        "batch": int(batch_size),
        "conf": float(cfg.evaluation.conf),
        "iou": float(cfg.evaluation.iou),
        "max_det": int(cfg.evaluation.max_det),
        "task": cfg.dataset.task,
        "device": device,
        "split": cfg.evaluation.split,
        "plots": bool(cfg.evaluation.plots),
        "save_json": bool(cfg.evaluation.save_json),
        "save_txt": False,
        "save_conf": False,
        "val": True,
        "workers": int(cfg.dataset.num_workers),
    }


def evaluate_checkpoint(
    cfg: Any,
    *,
    checkpoint_path: str | Path,
    datamodule: Any,
    project_root: Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_yaml = datamodule.write_data_yaml(Path(output_dir) / "coco_data.yaml")
    module = YOLO26LightningModule.load_from_checkpoint(
        str(checkpoint),
        cfg=cfg,
        names=datamodule.names,
        project_root=project_root,
        map_location="cpu",
    )
    module.eval()

    validator_cls = (
        COCOJsonSegmentationValidator if cfg.dataset.task == "segment" else COCOJsonDetectionValidator
    )
    validator = validator_cls(
        dataloader=datamodule.val_dataloader(),
        save_dir=Path(output_dir),
        args=build_validator_args(data_yaml, cfg, int(datamodule.eval_batch_size)),
    )
    metrics = validator(model=module.model)
    metrics = {
        **metrics,
        **summarize_validator_metrics(task=cfg.dataset.task, metrics=metrics, speed=validator.speed),
    }

    metrics_path = Path(output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics
